# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import List

import numpy as np
import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn
from torch.nn import functional as F

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.tts.helpers.helpers import binarize_attention_parallel, get_mask_from_lengths
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.fastpitch import regulate_len
from nemo.collections.tts.modules.talknet import GaussianEmbedding, MaskedInstanceNorm1d, StyleResidual
from nemo.core import Exportable
from nemo.core.classes import ModelPT, PretrainedModelInfo, typecheck
from nemo.core.neural_types import MelSpectrogramType, NeuralType
from nemo.utils import logging


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg


class TalkNetDursModel(ModelPT):
    """TalkNet's durations prediction pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.embed = nn.Embedding(len(self.vocab.labels), cfg.d_char)
        self.model = instantiate(cfg.model)
        d_out = cfg.model.jasper[-1].filters
        self.proj = nn.Conv1d(d_out, 1, kernel_size=1)

    def forward(self, text, text_len):
        x, x_len = self.embed(text).transpose(1, 2), text_len
        y, _ = self.model(x, x_len)
        durs = self.proj(y).squeeze(1)
        return durs

    @staticmethod
    def _metrics(true_durs, true_text_len, pred_durs):
        loss = F.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
        mask = get_mask_from_lengths(true_text_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()

        durs_pred = pred_durs.exp() - 1
        durs_pred[durs_pred < 0.0] = 0.0
        durs_pred = durs_pred.round().long()

        acc = ((true_durs == durs_pred) * mask).sum().float() / mask.sum() * 100
        acc_dist_1 = (((true_durs - durs_pred).abs() <= 1) * mask).sum().float() / mask.sum() * 100
        acc_dist_3 = (((true_durs - durs_pred).abs() <= 3) * mask).sum().float() / mask.sum() * 100

        return loss, acc, acc_dist_1, acc_dist_3

    def training_step(self, batch, batch_idx):
        _, _, text, text_len, durs, *_ = batch
        pred_durs = self(text=text, text_len=text_len)
        loss, acc, acc_dist_1, acc_dist_3 = self._metrics(true_durs=durs, true_text_len=text_len, pred_durs=pred_durs,)
        train_log = {
            'train_loss': loss,
            'train_acc': acc,
            'train_acc_dist_1': acc_dist_1,
            'train_acc_dist_3': acc_dist_3,
        }
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        _, _, text, text_len, durs, *_ = batch
        pred_durs = self(text=text, text_len=text_len)
        loss, acc, acc_dist_1, acc_dist_3 = self._metrics(true_durs=durs, true_text_len=text_len, pred_durs=pred_durs,)
        val_log = {'val_loss': loss, 'val_acc': acc, 'val_acc_dist_1': acc_dist_1, 'val_acc_dist_3': acc_dist_3}
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

    @staticmethod
    def _loader(cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_talknet",
            location=(
                "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_talknet/versions/1.0.0rc1/files"
                "/talknet_durs.nemo"
            ),
            description=(
                "This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate durations "
                "values for English voice with an American accent."
            ),
            class_=cls,  # noqa
            aliases=["TalkNet-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models


class TalkNetPitchModel(ModelPT):
    """TalkNet's pitch prediction pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.embed = GaussianEmbedding(self.vocab, cfg.d_char)
        self.model = instantiate(cfg.model)
        d_out = cfg.model.jasper[-1].filters
        self.sil_proj = nn.Conv1d(d_out, 1, kernel_size=1)
        self.body_proj = nn.Conv1d(d_out, 1, kernel_size=1)
        self.f0_mean, self.f0_std = cfg.f0_mean, cfg.f0_std

    def forward(self, text, text_len, durs):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        y, _ = self.model(x, x_len)
        f0_sil = self.sil_proj(y).squeeze(1)
        f0_body = self.body_proj(y).squeeze(1)
        return f0_sil, f0_body

    def _metrics(self, true_f0, true_f0_mask, pred_f0_sil, pred_f0_body):
        sil_mask = true_f0 < 1e-5
        sil_gt = sil_mask.long()
        sil_loss = F.binary_cross_entropy_with_logits(input=pred_f0_sil, target=sil_gt.float(), reduction='none',)
        sil_loss *= true_f0_mask.type_as(sil_loss)
        sil_loss = sil_loss.sum() / true_f0_mask.sum()
        sil_acc = ((torch.sigmoid(pred_f0_sil) > 0.5).long() == sil_gt).float()  # noqa
        sil_acc *= true_f0_mask.type_as(sil_acc)
        sil_acc = sil_acc.sum() / true_f0_mask.sum()

        body_mse = F.mse_loss(pred_f0_body, (true_f0 - self.f0_mean) / self.f0_std, reduction='none')
        body_mask = ~sil_mask
        body_mse *= body_mask.type_as(body_mse)  # noqa
        body_mse = body_mse.sum() / body_mask.sum()  # noqa
        body_mae = ((pred_f0_body * self.f0_std + self.f0_mean) - true_f0).abs()
        body_mae *= body_mask.type_as(body_mae)  # noqa
        body_mae = body_mae.sum() / body_mask.sum()  # noqa

        loss = sil_loss + body_mse

        return loss, sil_acc, body_mae

    def training_step(self, batch, batch_idx):
        _, audio_len, text, text_len, durs, f0, f0_mask = batch
        pred_f0_sil, pred_f0_body = self(text=text, text_len=text_len, durs=durs)
        loss, sil_acc, body_mae = self._metrics(
            true_f0=f0, true_f0_mask=f0_mask, pred_f0_sil=pred_f0_sil, pred_f0_body=pred_f0_body,
        )
        train_log = {'train_loss': loss, 'train_sil_acc': sil_acc, 'train_body_mae': body_mae}
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        _, _, text, text_len, durs, f0, f0_mask = batch
        pred_f0_sil, pred_f0_body = self(text=text, text_len=text_len, durs=durs)
        loss, sil_acc, body_mae = self._metrics(
            true_f0=f0, true_f0_mask=f0_mask, pred_f0_sil=pred_f0_sil, pred_f0_body=pred_f0_body,
        )

        val_log = {'val_loss': loss, 'val_sil_acc': sil_acc, 'val_body_mae': body_mae}
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

    @staticmethod
    def _loader(cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_talknet",
            location=(
                "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_talknet/versions/1.0.0rc1/files"
                "/talknet_pitch.nemo"
            ),
            description=(
                "This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate pitch "
                "values for English voice with an American accent."
            ),
            class_=cls,  # noqa
            aliases=["TalkNet-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models


class TalkNetSpectModel(SpectrogramGenerator, Exportable):
    """TalkNet's mel spectrogram prediction pipeline."""

    @property
    def output_types(self):
        return OrderedDict({"mel-spectrogram": NeuralType(('B', 'D', 'T'), MelSpectrogramType())})

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.blanking = cfg.train_ds.dataset.blanking
        self.preprocessor = instantiate(cfg.preprocessor)
        self.embed = GaussianEmbedding(self.vocab, cfg.d_char)
        self.norm_f0 = MaskedInstanceNorm1d(1)
        self.res_f0 = StyleResidual(cfg.d_char, 1, kernel_size=3)
        self.model = instantiate(cfg.model)
        d_out = cfg.model.jasper[-1].filters
        self.proj = nn.Conv1d(d_out, cfg.n_mels, kernel_size=1)

    def forward(self, text, text_len, durs, f0):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        f0, f0_mask = f0.clone(), f0 > 0.0
        f0 = self.norm_f0(f0.unsqueeze(1), f0_mask)
        f0[~f0_mask.unsqueeze(1)] = 0.0
        x = self.res_f0(x, f0)
        y, _ = self.model(x, x_len)
        mel = self.proj(y)
        return mel

    @staticmethod
    def _metrics(true_mel, true_mel_len, pred_mel):
        loss = F.mse_loss(pred_mel, true_mel, reduction='none').mean(dim=-2)
        mask = get_mask_from_lengths(true_mel_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()
        return loss

    def training_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, durs, f0, f0_mask = batch
        mel, mel_len = self.preprocessor(audio, audio_len)
        pred_mel = self(text=text, text_len=text_len, durs=durs, f0=f0)
        loss = self._metrics(true_mel=mel, true_mel_len=mel_len, pred_mel=pred_mel)
        train_log = {'train_loss': loss}
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, durs, f0, f0_mask = batch
        mel, mel_len = self.preprocessor(audio, audio_len)
        pred_mel = self(text=text, text_len=text_len, durs=durs, f0=f0)
        loss = self._metrics(true_mel=mel, true_mel_len=mel_len, pred_mel=pred_mel)
        val_log = {'val_loss': loss}
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

    @staticmethod
    def _loader(cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    def parse(self, text: str, **kwargs) -> torch.Tensor:
        return torch.tensor(self.vocab.encode(text)).long().unsqueeze(0).to(self.device)

    def generate_spectrogram(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        assert hasattr(self, '_durs_model') and hasattr(self, '_pitch_model')

        if self.blanking:
            tokens = [
                AudioToCharWithDursF0Dataset.interleave(
                    x=torch.empty(len(t) + 1, dtype=torch.long, device=t.device).fill_(self.vocab.blank), y=t,
                )
                for t in tokens
            ]
            tokens = AudioToCharWithDursF0Dataset.merge(tokens, value=self.vocab.pad, dtype=torch.long)

        text_len = torch.tensor(tokens.shape[-1], dtype=torch.long).unsqueeze(0)
        durs = self._durs_model(tokens, text_len)
        durs = durs.exp() - 1
        durs[durs < 0.0] = 0.0
        durs = durs.round().long()

        # Pitch
        f0_sil, f0_body = self._pitch_model(tokens, text_len, durs)
        sil_mask = f0_sil.sigmoid() > 0.5
        f0 = f0_body * self._pitch_model.f0_std + self._pitch_model.f0_mean
        f0 = (~sil_mask * f0).float()

        # Spect
        mel = self(tokens, text_len, durs, f0)

        return mel

    def forward_for_export(self, tokens: torch.Tensor, text_len: torch.Tensor):
        durs = self._durs_model(tokens, text_len)
        durs = durs.exp() - 1
        durs[durs < 0.0] = 0.0
        durs = durs.round().long()

        # Pitch
        f0_sil, f0_body = self._pitch_model(tokens, text_len, durs)
        sil_mask = f0_sil.sigmoid() > 0.5
        f0 = f0_body * self._pitch_model.f0_std + self._pitch_model.f0_mean
        f0 = (~sil_mask * f0).float()

        # Spect
        x, x_len = self.embed(tokens, durs).transpose(1, 2), durs.sum(-1)
        f0, f0_mask = f0.clone(), f0 > 0.0
        f0 = self.norm_f0(f0.unsqueeze(1), f0_mask)
        f0[~f0_mask.unsqueeze(1)] = 0.0
        x = self.res_f0(x, f0)
        y, _ = self.model(x, x_len)
        mel = self.proj(y)

        return mel

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_talknet",
            location=(
                "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_talknet/versions/1.0.0rc1/files"
                "/talknet_spect.nemo"
            ),
            description=(
                "This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate female "
                "English voices with an American accent."
            ),
            class_=cls,  # noqa
            aliases=["TalkNet-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models


class TalkNet3Model(ModelPT):
    """TalkNet 3 pipeline"""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        typecheck.set_typecheck_enabled(enabled=False)
        cfg = self._cfg

        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.pitch_mean, self.pitch_std = float(cfg.pitch_mean), float(cfg.pitch_std)

        self.pitch_loss_scale, self.durs_loss_scale = cfg.pitch_loss_scale, cfg.durs_loss_scale

        self.learn_alignment = False
        if "learn_alignment" in cfg:
            self.learn_alignment = cfg.learn_alignment

        if self.learn_alignment:
            self.aligner = instantiate(cfg.alignment_module)
            self.forward_sum_loss = ForwardSumLoss()
            self.bin_loss = BinLoss()
            self.add_bin_loss = False
            self.bin_loss_scale = 0.0
            self.bin_loss_start_ratio = cfg.bin_loss_start_ratio
            self.bin_loss_warmup_epochs = cfg.bin_loss_warmup_epochs

        # TODO(Oktai): debug flags
        self.encoder_type = "transformer"
        self.decoder_type = "transformer"

        # Transformer-based encoder
        if self.encoder_type == "transformer":
            self.encoder = instantiate(cfg.encoder, n_embed=len(self.vocab.labels), padding_idx=self.vocab.pad)
            self.symbol_emb = self.encoder.word_emb
        # MLPMixer-based encoder or GMLP-based encoder
        elif self.encoder_type == "mlp_mixer" or self.encoder_type == "g_mlp":
            self.encoder = instantiate(cfg.encoder, num_tokens=len(self.vocab.labels))
            self.symbol_emb = self.encoder.to_embed
        # QN-based encoder
        elif self.encoder_type == "qn":
            self.encoder = instantiate(cfg.encoder)
            self.symbol_emb = nn.Embedding(len(self.vocab.labels), cfg.symbols_embedding_dim)
        else:
            raise NotImplementedError

        self.duration_predictor = instantiate(cfg.duration_predictor)
        self.pitch_predictor = instantiate(cfg.pitch_predictor)

        self.preprocessor = instantiate(cfg.preprocessor)
        self.pitch_emb = instantiate(cfg.pitch_emb)

        # Transformer-based decoder
        if self.decoder_type == "transformer":
            self.decoder = instantiate(cfg.decoder)
            d_out = self.decoder.d_model
        # MLPMixer-based decoder or GMLP-based decoder
        elif self.decoder_type == "mlp_mixer" or self.decoder_type == "g_mlp":
            self.decoder = instantiate(cfg.decoder)
            d_out = self.decoder.d_model
        # QN-based decoder
        elif self.decoder_type == "qn":
            self.decoder = instantiate(cfg.decoder)
            d_out = cfg.decoder.jasper[-1].filters
        else:
            raise NotImplementedError

        self.proj = nn.Linear(d_out, cfg.n_mel_channels)

    def _metrics(
        self,
        true_durs,
        true_text_len,
        pred_durs,
        true_pitch,
        pred_pitch,
        true_spect=None,
        pred_spect=None,
        true_spect_len=None,
        attn_logprob=None,
        attn_soft=None,
        attn_hard=None,
        attn_hard_dur=None,
    ):
        mask = get_mask_from_lengths(true_text_len)

        # dur loss and metrics
        durs_loss = F.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
        durs_loss = durs_loss * mask.float()
        durs_loss = durs_loss.sum() / mask.sum()

        durs_pred = pred_durs.exp() - 1
        durs_pred[durs_pred < 0.0] = 0.0
        durs_pred = durs_pred.round().long()

        acc = ((true_durs == durs_pred) * mask).sum().float() / mask.sum() * 100
        acc_dist_1 = (((true_durs - durs_pred).abs() <= 1) * mask).sum().float() / mask.sum() * 100
        acc_dist_3 = (((true_durs - durs_pred).abs() <= 3) * mask).sum().float() / mask.sum() * 100

        # spec loss
        pred_spect = pred_spect.transpose(1, 2)

        mel_loss = F.mse_loss(pred_spect, true_spect, reduction='none').mean(dim=-2)
        mel_mask = get_mask_from_lengths(true_spect_len)
        mel_loss = mel_loss * mel_mask.float()
        mel_loss = mel_loss.sum() / mel_mask.sum()

        loss = self.durs_loss_scale * durs_loss + mel_loss

        # aligner loss
        bin_loss, ctc_loss = None, None
        if self.learn_alignment:
            ctc_loss = self.forward_sum_loss(attn_logprob=attn_logprob, in_lens=true_text_len, out_lens=true_spect_len)
            loss = loss + ctc_loss
            if self.add_bin_loss:
                bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)
                loss = loss + self.bin_loss_scale * bin_loss
            true_pitch = average_pitch(true_pitch.unsqueeze(1), attn_hard_dur).squeeze(1)

        # pitch loss
        pitch_loss = F.mse_loss(pred_pitch, true_pitch, reduction='none')
        pitch_loss = (pitch_loss * mask).sum() / mask.sum()

        loss = loss + self.pitch_loss_scale * pitch_loss

        return loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss

    def forward(self, text, text_len, durs=None, pitch=None, spect=None, spect_len=None, attn_prior=None):
        if self.training:
            assert pitch is not None
            if not self.learn_alignment:
                assert durs is not None

        # Transformer-based encoder
        if self.encoder_type == "transformer":
            enc_out, enc_mask = self.encoder(input=text, conditioning=0)
        # MLPMixer-based encoder or GMLP-based encoder
        elif self.encoder_type == "mlp_mixer" or self.encoder_type == "g_mlp":
            enc_out, enc_len = self.encoder(text, text_len)
            enc_mask = get_mask_from_lengths(enc_len).unsqueeze(2)
        elif self.decoder_type == "qn":
            enc_out, enc_len = self.encoder(self.symbol_emb(text).transpose(1, 2), text_len)
            enc_mask = get_mask_from_lengths(enc_len).unsqueeze(2)
            enc_out = enc_out.transpose(1, 2)
        else:
            raise NotImplementedError

        attn_soft, attn_hard, attn_hard_dur, attn_logprob = None, None, None, None
        if self.learn_alignment:
            text_emb = self.symbol_emb(text)
            attn_soft, attn_logprob = self.aligner(
                spect,
                text_emb.permute(0, 2, 1),
                mask=get_mask_from_lengths(text_len).unsqueeze(-1) == 0,
                attn_prior=attn_prior,
            )
            attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            assert torch.all(torch.eq(attn_hard_dur.sum(dim=1), spect_len))

        log_durs_predicted = self.duration_predictor(enc_out, enc_mask)
        durs_predicted = torch.clamp(log_durs_predicted.exp() - 1, 0)

        pitch_predicted = self.pitch_predictor(enc_out, enc_mask)

        if not self.training:
            if self.learn_alignment and pitch is not None:
                pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
                pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
            else:
                pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))
        else:
            if self.learn_alignment:
                pitch = average_pitch(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))

        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if self.learn_alignment:
            len_regulated_enc_out, dec_lens = regulate_len(attn_hard_dur, enc_out)
        elif spect is None and durs is not None:
            len_regulated_enc_out, dec_lens = regulate_len(durs, enc_out)
        # Use predictions during inference
        elif spect is None:
            len_regulated_enc_out, dec_lens = regulate_len(durs_predicted, enc_out)
        else:
            raise NotImplementedError

        # Transformer-based decoder
        if self.decoder_type == "transformer":
            dec_out, _ = self.decoder(input=len_regulated_enc_out, seq_lens=dec_lens)
            pred_spect = self.proj(dec_out)
        # MLPMixer-based decoder or GMLP-based decoder
        elif self.decoder_type == "mlp_mixer" or self.decoder_type == "g_mlp":
            dec_out, _ = self.decoder(len_regulated_enc_out, dec_lens)
            pred_spect = self.proj(dec_out)
        # QN-based decoder
        elif self.decoder_type == "qn":
            dec_out, _ = self.decoder(len_regulated_enc_out.transpose(1, 2), dec_lens)
            pred_spect = self.proj(dec_out.transpose(1, 2))
        else:
            raise NotImplementedError

        return (
            pred_spect,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
        )

    def infer(self, text, text_len, durs=None, spect=None, spect_len=None, attn_prior=None, use_gt_durs=False):
        # Transformer-based encoder
        if self.encoder_type == "transformer":
            enc_out, enc_mask = self.encoder(input=text, conditioning=0)
        # MLPMixer-based encoder or GMLP-based encoder
        elif self.encoder_type == "mlp_mixer" or self.encoder_type == "g_mlp":
            enc_out, enc_len = self.encoder(text, text_len)
            enc_mask = get_mask_from_lengths(enc_len).unsqueeze(2)
        elif self.decoder_type == "qn":
            enc_out, enc_len = self.encoder(self.symbol_emb(text).transpose(1, 2), text_len)
            enc_mask = get_mask_from_lengths(enc_len).unsqueeze(2)
            enc_out = enc_out.transpose(1, 2)
        else:
            raise NotImplementedError

        attn_hard_dur = None
        if self.learn_alignment and use_gt_durs:
            text_emb = self.symbol_emb(text)
            attn_soft, _ = self.aligner(
                spect,
                text_emb.permute(0, 2, 1),
                mask=get_mask_from_lengths(text_len).unsqueeze(-1) == 0,
                attn_prior=attn_prior,
            )
            attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            assert torch.all(torch.eq(attn_hard_dur.sum(dim=1), spect_len))

        log_durs_predicted = self.duration_predictor(enc_out, enc_mask)
        durs_predicted = torch.clamp(log_durs_predicted.exp() - 1, 0)

        pitch_predicted = self.pitch_predictor(enc_out, enc_mask)
        pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))

        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if use_gt_durs:
            if durs is not None:
                len_regulated_enc_out, dec_lens = regulate_len(durs, enc_out)
            elif attn_hard_dur is not None:
                len_regulated_enc_out, dec_lens = regulate_len(attn_hard_dur, enc_out)
            else:
                raise NotImplementedError
        else:
            len_regulated_enc_out, dec_lens = regulate_len(durs_predicted, enc_out)

        # Transformer-based decoder
        if self.decoder_type == "transformer":
            dec_out, _ = self.decoder(input=len_regulated_enc_out, seq_lens=dec_lens)
            pred_spect = self.proj(dec_out)
        # QN-based decoder
        elif self.decoder_type == "qn":
            dec_out, _ = self.decoder(len_regulated_enc_out.transpose(1, 2), dec_lens)
            pred_spect = self.proj(dec_out.transpose(1, 2))
            # pred_spect = pred_spect.transpose(1, 2)
        # MLPMixer-based decoder or GMLP-based decoder
        elif self.decoder_type == "mlp_mixer" or self.decoder_type == "g_mlp":
            dec_out, _ = self.decoder(len_regulated_enc_out, dec_lens)
            pred_spect = self.proj(dec_out)
        else:
            raise NotImplementedError

        return pred_spect

    def on_train_epoch_start(self):
        if self.learn_alignment:
            bin_loss_start_epoch = np.ceil(self.bin_loss_start_ratio * self._trainer.max_epochs)

            # Add bin loss when current_epoch >= bin_start_epoch
            if not self.add_bin_loss and self.current_epoch >= bin_loss_start_epoch:
                logging.info(f"Using hard attentions after epoch: {self.current_epoch}")
                self.add_bin_loss = True

            if self.add_bin_loss:
                self.bin_loss_scale = min(
                    (self.current_epoch - bin_loss_start_epoch) / self.bin_loss_warmup_epochs, 1.0
                )

    def training_step(self, batch, batch_idx):
        attn_prior, durs = None, None
        if self.learn_alignment:
            audio, audio_len, text, text_len, attn_prior, pitch = batch
        else:
            audio, audio_len, text, text_len, durs, pitch, _ = batch

        spect, spect_len = self.preprocessor(input_signal=audio, length=audio_len)

        # pitch normalization
        zero_pitch_idx = pitch == 0
        pitch = (pitch - self.pitch_mean) / self.pitch_std
        pitch[zero_pitch_idx] = 0.0

        pred_spect, _, log_durs_pred, pitch_pred, attn_soft, attn_logprob, attn_hard, attn_hard_dur = self(
            text=text,
            text_len=text_len,
            durs=None if self.learn_alignment else durs,
            pitch=pitch,
            spect=spect if self.learn_alignment else None,
            spect_len=spect_len,
            attn_prior=attn_prior,
        )

        if durs is None:
            durs = attn_hard_dur

        loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss = self._metrics(
            pred_durs=log_durs_pred,
            pred_pitch=pitch_pred,
            true_durs=durs,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )

        train_log = {
            'train_loss': loss,
            'train_durs_loss': durs_loss,
            'train_pitch_loss': pitch_loss,
            'train_mel_loss': mel_loss,
            'train_durs_acc': acc,
            'train_durs_acc_dist_1': acc_dist_1,
            'train_durs_acc_dist_3': acc_dist_3,
            'train_ctc_loss': torch.tensor(1.0).to(durs_loss.device) if ctc_loss is None else ctc_loss,
            'train_bin_loss': torch.tensor(1.0).to(durs_loss.device) if bin_loss is None else bin_loss,
        }

        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        attn_prior, durs = None, None
        if self.learn_alignment:
            audio, audio_len, text, text_len, attn_prior, pitch = batch
        else:
            audio, audio_len, text, text_len, durs, pitch, _ = batch

        spect, spect_len = self.preprocessor(input_signal=audio, length=audio_len)

        # pitch normalization
        zero_pitch_idx = pitch == 0
        pitch = (pitch - self.pitch_mean) / self.pitch_std
        pitch[zero_pitch_idx] = 0.0

        pred_spect, _, log_durs_pred, pitch_pred, attn_soft, attn_logprob, attn_hard, attn_hard_dur = self(
            text=text,
            text_len=text_len,
            durs=None if self.learn_alignment else durs,
            pitch=pitch,
            spect=spect if self.learn_alignment else None,
            spect_len=spect_len,
            attn_prior=attn_prior,
        )

        if durs is None:
            durs = attn_hard_dur

        loss, durs_loss, acc, acc_dist_1, acc_dist_3, pitch_loss, mel_loss, ctc_loss, bin_loss = self._metrics(
            pred_durs=log_durs_pred,
            pred_pitch=pitch_pred,
            true_durs=durs,
            true_text_len=text_len,
            true_pitch=pitch,
            true_spect=spect,
            pred_spect=pred_spect,
            true_spect_len=spect_len,
            attn_logprob=attn_logprob,
            attn_soft=attn_soft,
            attn_hard=attn_hard,
            attn_hard_dur=attn_hard_dur,
        )

        val_log = {
            'val_loss': loss,
            'val_durs_loss': durs_loss,
            'val_pitch_loss': pitch_loss,
            'val_mel_loss': mel_loss,
            'val_durs_acc': acc,
            'val_durs_acc_dist_1': acc_dist_1,
            'val_durs_acc_dist_3': acc_dist_3,
            'val_ctc_loss': torch.tensor(1.0).to(durs_loss.device) if ctc_loss is None else ctc_loss,
            'val_bin_loss': torch.tensor(1.0).to(durs_loss.device) if bin_loss is None else bin_loss,
        }
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

    def parse(self, text: str, **kwargs) -> torch.Tensor:
        return torch.tensor(self.vocab.encode(text)).long().unsqueeze(0).to(self.device)

    @staticmethod
    def _loader(cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls):
        """Empty."""
        pass
