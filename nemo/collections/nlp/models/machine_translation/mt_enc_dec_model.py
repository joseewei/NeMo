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

from dataclasses import dataclass
import itertools
import time
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
import os
from tempfile import TemporaryDirectory, TemporaryFile

from youtokentome.youtokentome import BPE
from nemo.collections.nlp.modules.common.lm_utils import get_transformer

import omegaconf
from omegaconf.omegaconf import OmegaConf
from nemo.collections.nlp.modules.common.transformer.transformer_utils import get_nemo_transformer
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data as pt_data
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from sacrebleu import corpus_bleu
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.nlp.data import TranslationDataset
from nemo.collections.nlp.models.enc_dec_nlp_model import EncDecNLPModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.transformer import BeamSearchSequenceGenerator
from nemo.collections.nlp.modules.common.transformer.transformer import TransformerDecoderNM, TransformerEncoderNM
from nemo.core.classes.common import typecheck
from nemo.utils import logging, model_utils


__all__ = ['MTEncDecModel']


class MTEncDecModel(EncDecNLPModel):
    """
    Encoder-decoder machine translation model.
    """

    def __init__(self, cfg: MTEncDecModelConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        self.setup_enc_dec_tokenizers(cfg)

        super().__init__(cfg=cfg, trainer=trainer)

        # TODO: use get_encoder function with support for HF and Megatron
        # self.encoder = TransformerEncoderNM(
        #     vocab_size=self.encoder_vocab_size,
        #     hidden_size=cfg.encoder.hidden_size,
        #     num_layers=cfg.encoder.num_layers,
        #     inner_size=cfg.encoder.inner_size,
        #     max_sequence_length=cfg.encoder.max_sequence_length
        #     if hasattr(cfg.encoder, 'max_sequence_length')
        #     else 512,
        #     embedding_dropout=cfg.encoder.embedding_dropout if hasattr(cfg.encoder, 'embedding_dropout') else 0.0,
        #     learn_positional_encodings=cfg.encoder.learn_positional_encodings
        #     if hasattr(cfg.encoder, 'learn_positional_encodings')
        #     else False,
        #     num_attention_heads=cfg.encoder.num_attention_heads,
        #     ffn_dropout=cfg.encoder.ffn_dropout,
        #     attn_score_dropout=cfg.encoder.attn_score_dropout,
        #     attn_layer_dropout=cfg.encoder.attn_layer_dropout,
        #     hidden_act=cfg.encoder.hidden_act,
        #     mask_future=cfg.encoder.mask_future,
        #     pre_ln=cfg.encoder.pre_ln,
        # )
        encoder_cfg = OmegaConf.to_container(cfg.get('encoder'))
        encoder_cfg['vocab_size'] = self.encoder_vocab_size
        library = encoder_cfg.pop('library', 'nemo')
        model_name = encoder_cfg.pop('model_name', None)
        pretrained = encoder_cfg.pop('pretrained', False)
        self.encoder = get_transformer(
            library=library, model_name=model_name, pretrained=pretrained, config_dict=encoder_cfg
        )

        # TODO: user get_decoder function with support for HF and Megatron
        self.decoder = TransformerDecoderNM(
            vocab_size=self.decoder_vocab_size,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            inner_size=cfg.decoder.inner_size,
            max_sequence_length=cfg.decoder.max_sequence_length
            if hasattr(cfg.decoder, 'max_sequence_length')
            else 512,
            embedding_dropout=cfg.decoder.embedding_dropout if hasattr(cfg.decoder, 'embedding_dropout') else 0.0,
            learn_positional_encodings=cfg.decoder.learn_positional_encodings
            if hasattr(cfg.decoder, 'learn_positional_encodings')
            else False,
            num_attention_heads=cfg.decoder.num_attention_heads,
            ffn_dropout=cfg.decoder.ffn_dropout,
            attn_score_dropout=cfg.decoder.attn_score_dropout,
            attn_layer_dropout=cfg.decoder.attn_layer_dropout,
            hidden_act=cfg.decoder.hidden_act,
            pre_ln=cfg.decoder.pre_ln,
        )

        self.log_softmax = TokenClassifier(
            hidden_size=self.decoder.hidden_size,
            num_classes=self.decoder_vocab_size,
            activation=cfg.head.activation,
            log_softmax=cfg.head.log_softmax,
            dropout=cfg.head.dropout,
            use_transformer_init=cfg.head.use_transformer_init,
        )

        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.decoder.embedding,
            decoder=self.decoder.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.decoder.max_sequence_length,
            beam_size=cfg.beam_size,
            bos=self.decoder_tokenizer.bos_id,
            pad=self.decoder_tokenizer.pad_id,
            eos=self.decoder_tokenizer.eos_id,
            len_pen=cfg.len_pen,
            max_delta_length=cfg.max_generation_delta,
        )

        # tie weights of embedding and softmax matrices
        self.log_softmax.mlp.layer0.weight = self.decoder.embedding.token_embedding.weight

        # TODO: encoder and decoder with different hidden size?
        std_init_range = 1 / self.encoder.hidden_size ** 0.5
        self.apply(lambda module: transformer_weights_init(module, std_init_range))

        self.loss_fn = SmoothedCrossEntropyLoss(
            pad_id=self.decoder_tokenizer.pad_id, label_smoothing=cfg.label_smoothing
        )

        # self.training_perplexity = Perplexity(dist_sync_on_step=True)
        # self.eval_perplexity = Perplexity(compute_on_step=False)

    def filter_predicted_ids(self, ids):
        ids[ids >= self.decoder_tokenizer.vocab_size] = self.decoder_tokenizer.unk_id
        return ids

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        torch.nn.Module.forward method.
        Args:
            src: source ids
            src_mask: src mask (mask padding)
            tgt: target ids
            tgt_mask: target mask

        Returns:

        """
        src_hiddens = self.encoder(src, src_mask)
        if tgt is not None:
            tgt_hiddens = self.decoder(tgt, tgt_mask, src_hiddens, src_mask)
            log_probs = self.log_softmax(hidden_states=tgt_hiddens)
        else:
            log_probs = None
        beam_results = None
        if not self.training:
            beam_results = self.beam_search(encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask)
            beam_results = self.filter_predicted_ids(beam_results)
        return log_probs, beam_results

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels, _ = batch
        log_probs, _ = self(src_ids, src_mask, tgt_ids, tgt_mask)
        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        # training_perplexity = self.training_perplexity(logits=log_probs)
        tensorboard_logs = {
            'train_loss': train_loss,
            'lr': self._optimizer.param_groups[0]['lr'],
            # "train_ppl": training_perplexity,
        }
        return {'loss': train_loss, 'log': tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode):
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids = batch
        log_probs, beam_results = self(src_ids, src_mask, tgt_ids, tgt_mask)
        eval_loss = self.loss_fn(log_probs=log_probs, labels=labels).cpu().numpy()
        # self.eval_perplexity(logits=log_probs)
        translations = [self.decoder_tokenizer.ids_to_text(tr) for tr in beam_results.cpu().numpy()]
        np_tgt = tgt_ids.cpu().numpy()
        ground_truths = [self.decoder_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        num_non_pad_tokens = np.not_equal(np_tgt, self.decoder_tokenizer.pad_id).sum().item()
        tensorboard_logs = {f'{mode}_loss': eval_loss}
        return {
            f'{mode}_loss': eval_loss,
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
            'log': tensorboard_logs,
        }

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    @rank_zero_only
    def log_param_stats(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.trainer.logger.experiment.add_histogram(name + '_hist', p, global_step=self.global_step)
                self.trainer.logger.experiment.add_scalars(
                    name,
                    {'mean': p.mean(), 'stddev': p.std(), 'max': p.max(), 'min': p.min()},
                    global_step=self.global_step,
                )

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.eval_step(batch, batch_idx, 'val')

    def eval_epoch_end(self, outputs, mode):
        counts = np.array([x['num_non_pad_tokens'] for x in outputs])
        eval_loss = np.sum(np.array([x[f'{mode}_loss'] for x in outputs]) * counts) / counts.sum()
        # eval_perplexity = self.eval_perplexity.compute()
        translations = list(itertools.chain(*[x['translations'] for x in outputs]))
        ground_truths = list(itertools.chain(*[x['ground_truths'] for x in outputs]))
        # __TODO__ add target language so detokenizer can be lang specific.
        detokenizer = MosesDetokenizer()
        translations = [detokenizer.detokenize(sent.split()) for sent in translations]
        ground_truths = [detokenizer.detokenize(sent.split()) for sent in ground_truths]
        assert len(translations) == len(ground_truths)
        sacre_bleu = corpus_bleu(translations, [ground_truths], tokenize="13a")
        dataset_name = "Validation" if mode == 'val' else "Test"
        logging.info(f"\n\n\n\n{dataset_name} set size: {len(translations)}")
        logging.info(f"{dataset_name} Sacre BLEU = {sacre_bleu.score}")
        logging.info(f"{dataset_name} TRANSLATION EXAMPLES:".upper())
        for i in range(0, 3):
            ind = random.randint(0, len(translations) - 1)
            logging.info("    " + '\u0332'.join(f"EXAMPLE {i}:"))
            logging.info(f"    Prediction:   {translations[ind]}")
            logging.info(f"    Ground Truth: {ground_truths[ind]}")

        ans = {f"{mode}_loss": eval_loss, f"{mode}_sacreBLEU": sacre_bleu.score}  # , f"{mode}_ppl": eval_perplexity}
        ans['log'] = dict(ans)
        return ans

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.log_dict(self.eval_epoch_end(outputs, 'val'))

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')

    # def setup_training_data(self, train_data_config: Optional[DictConfig]):
    #     self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    # def setup_validation_data(self, val_data_config: Optional[DictConfig]):
    #     self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    # def setup_test_data(self, test_data_config: Optional[DictConfig]):
    #     self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        pass

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        pass

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        pass

    def train_dataloader(self):
        return self._setup_dataloader_from_config(cfg=self._cfg.train_ds)

    def validation_dataloader(self):
        return self._setup_dataloader_from_config(cfg=self._cfg.validation_ds)

    def test_dataloader(self):
        return self._setup_dataloader_from_config(cfg=self._cfg.test_ds)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        cached_dataset_fname = self.get_cached_dataset_fname(cfg.src_file_name, cfg.tgt_file_name, cfg.tokens_in_batch)
        logging.info(f'Loading from cached dataset {cached_dataset_fname}')
        dataset = pickle.load(open(cached_dataset_fname, 'rb'))
        dataset.reverse_lang_direction = cfg.get("reverse_lang_direction", False)
        if cfg.shuffle:
            sampler = pt_data.RandomSampler(dataset)
        else:
            sampler = pt_data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
        )

    @torch.no_grad()
    def translate(self, text: List[str], source_lang: str = 'de', target_lang: str = 'en') -> List[str]:
        """
        Translates list of sentences from source language to target language.
        Should be regular text, this method performs its own tokenization/de-tokenization
        Args:
            text: list of strings to translate
        Returns:
            list of translated strings
        """
        mode = self.training
        if source_lang != "None":
            tokenizer = MosesTokenizer(lang=source_lang)
            normalizer = MosesPunctNormalizer(lang=source_lang)
        if target_lang != "None":
            detokenizer = MosesDetokenizer(lang=target_lang)
        try:
            self.eval()
            res = []
            for txt in text:
                if source_lang != "None":
                    txt = normalizer.normalize(txt)
                    txt = tokenizer.tokenize(txt, escape=False, return_str=True)
                ids = self.encoder_tokenizer.text_to_ids(txt)
                ids = [self.encoder_tokenizer.bos_id] + ids + [self.encoder_tokenizer.eos_id]
                src = torch.Tensor(ids).long().to(self._device).unsqueeze(0)
                src_mask = torch.ones_like(src)
                src_embeddings = self.encoder._embedding(input_ids=src)
                src_hiddens = self.encoder._encoder(src_embeddings, src_mask)
                beam_results = self.beam_search(encoder_hidden_states=src_hiddens, encoder_input_mask=src_mask)
                beam_results = self.filter_predicted_ids(beam_results)
                translation_ids = beam_results.cpu()[0].numpy()
                translation = self.decoder_tokenizer.ids_to_text(translation_ids)
                if target_lang != "None":
                    translation = detokenizer.detokenize(translation.split())
                res.append(translation)
        finally:
            self.train(mode=mode)
        return res

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    def prepare_data(self) -> None:

        logging.info('Preprocessing datasets.')

        # preprocess train dataset
        self.preprocess_dataset(
            encoder_tokenizer_library=self._cfg.encoder_tokenizer.get('tokenizer_name', 'yttm'),
            encoder_tokenizer_model_name=self._cfg.encoder_tokenizer.get('model_name', None),
            encoder_vocab_size=self._cfg.encoder_tokenizer.get('vocab_size', 8192),
            encoder_bpe_dropout=self._cfg.encoder_tokenizer.get('bpe_dropout', 0.0),
            decoder_tokenizer_library=self._cfg.decoder_tokenizer.get('tokenizer_name', 'yttm'),
            decoder_tokenizer_model_name=self._cfg.decoder_tokenizer.get('model_name', None),
            decoder_vocab_size=self._cfg.decoder_tokenizer.get('vocab_size', 8192),
            decoder_bpe_dropout=self._cfg.decoder_tokenizer.get('bpe_dropout', 0.0),
            use_shared_tokenizer=self._cfg.get('use_shared_tokenizer', True),
            clean=self._cfg.train_ds.get('clean', False),
            src_fname=self._cfg.train_ds.get('src_file_name'),
            tgt_fname=self._cfg.train_ds.get('tgt_file_name'),
            out_dir=self._cfg.get('data_preprocessing_out_dir'),
            max_seq_length=self._cfg.encoder.get('max_sequence_length', 512),
            min_seq_length=self._cfg.encoder.get('min_sequence_length', 1),
            tokens_in_batch=self._cfg.train_ds.get('tokens_in_batch', 512),
        )

        # preprocess validation dataset
        self.preprocess_dataset(
            encoder_tokenizer_library=self._cfg.encoder_tokenizer.get('tokenizer_name', 'yttm'),
            encoder_tokenizer_model_name=self._cfg.encoder_tokenizer.get('model_name', None),
            encoder_vocab_size=self._cfg.encoder_tokenizer.get('vocab_size', 8192),
            encoder_bpe_dropout=self._cfg.encoder_tokenizer.get('bpe_dropout', 0.0),
            decoder_tokenizer_library=self._cfg.decoder_tokenizer.get('tokenizer_name', 'yttm'),
            decoder_tokenizer_model_name=self._cfg.decoder_tokenizer.get('model_name', None),
            decoder_vocab_size=self._cfg.decoder_tokenizer.get('vocab_size', 8192),
            decoder_bpe_dropout=self._cfg.decoder_tokenizer.get('bpe_dropout', 0.0),
            use_shared_tokenizer=self._cfg.get('use_shared_tokenizer', True),
            clean=self._cfg.validation_ds.get('clean', False),
            src_fname=self._cfg.validation_ds.get('src_file_name'),
            tgt_fname=self._cfg.validation_ds.get('tgt_file_name'),
            out_dir=self._cfg.get('data_preprocessing_out_dir'),
            max_seq_length=self._cfg.encoder.get('max_sequence_length', 512),
            min_seq_length=self._cfg.encoder.get('min_sequence_length', 1),
            tokens_in_batch=self._cfg.validation_ds.get('tokens_in_batch', 512),
        )

        # preprocess test dataset
        self.preprocess_dataset(
            encoder_tokenizer_library=self._cfg.encoder_tokenizer.get('tokenizer_name', 'yttm'),
            encoder_tokenizer_model_name=self._cfg.encoder_tokenizer.get('model_name', None),
            encoder_vocab_size=self._cfg.encoder_tokenizer.get('vocab_size', 8192),
            encoder_bpe_dropout=self._cfg.encoder_tokenizer.get('bpe_dropout', 0.0),
            decoder_tokenizer_library=self._cfg.decoder_tokenizer.get('tokenizer_name', 'yttm'),
            decoder_tokenizer_model_name=self._cfg.decoder_tokenizer.get('model_name', None),
            decoder_vocab_size=self._cfg.decoder_tokenizer.get('vocab_size', 8192),
            decoder_bpe_dropout=self._cfg.decoder_tokenizer.get('bpe_dropout', 0.0),
            use_shared_tokenizer=self._cfg.get('use_shared_tokenizer', True),
            clean=self._cfg.test_ds.get('clean', False),
            src_fname=self._cfg.test_ds.get('src_file_name'),
            tgt_fname=self._cfg.test_ds.get('tgt_file_name'),
            out_dir=self._cfg.get('data_preprocessing_out_dir'),
            max_seq_length=self._cfg.encoder.get('max_sequence_length', 512),
            min_seq_length=self._cfg.encoder.get('min_sequence_length', 1),
            tokens_in_batch=self._cfg.test_ds.get('tokens_in_batch', 512),
        )

    def get_cached_dataset_fname(self, src_fname: str, tgt_fname: str, num_tokens: int) -> str:
        cached_dataset_fname = f'{src_fname}_{tgt_fname}_batches.tokens.{num_tokens}.pkl'
        return cached_dataset_fname

    def preprocess_dataset(
        self,
        encoder_tokenizer_library: str = None,
        encoder_tokenizer_model_name: Optional[str] = None,
        encoder_vocab_size: Optional[int] = None,
        encoder_bpe_dropout: Optional[float] = 0.0,
        decoder_tokenizer_library: str = None,
        decoder_tokenizer_model_name: Optional[str] = None,
        decoder_vocab_size: Optional[int] = None,
        decoder_bpe_dropout: Optional[float] = 0.0,
        use_shared_tokenizer: bool = False,
        clean: bool = False,
        src_fname: str = None,
        tgt_fname: str = None,
        out_dir: str = None,
        max_seq_length: int = 512,
        min_seq_length: int = 1,
        tokens_in_batch: int = 8192,
    ) -> None:
        encoder_tokenizer = None
        decoder_tokenizer = None
        if encoder_tokenizer_library == 'yttm':
            encoder_tokenizer_bpe_model = None
            if use_shared_tokenizer:
                assert (
                    encoder_vocab_size == decoder_vocab_size
                ), "If using a shared tokenizer, then encoder vocab size and decoder vocab size should be the same."

                encoder_tokenizer_bpe_model = os.path.join(
                    out_dir, f'yttm_shared_tokenizer.{encoder_vocab_size}.BPE.model'
                )
                train_yttm_bpe(
                    data=[src_fname, tgt_fname], vocab_size=encoder_vocab_size, model=encoder_tokenizer_bpe_model,
                )
                encoder_tokenizer_model = encoder_tokenizer_bpe_model
            else:
                encoder_bpe_model = os.path.join(out_dir, f'yttm_encoder_tokenizer.{encoder_vocab_size}.BPE.model')
                train_yttm_bpe(data=[src_fname], vocab_size=encoder_vocab_size, model=encoder_bpe_model)
                encoder_tokenizer_model = encoder_bpe_model
            encoder_tokenizer = get_tokenizer(
                tokenizer_name='yttm', tokenizer_model=encoder_tokenizer_model, bpe_dropout=encoder_bpe_dropout
            )

        if decoder_tokenizer_library == 'yttm':
            decoder_tokenizer_bpe_model = None
            if not use_shared_tokenizer:
                decoder_bpe_model = os.path.join(out_dir, f'yttm_decoder_tokenizer.{decoder_vocab_size}.BPE.model')
                train_yttm_bpe(data=[tgt_fname], vocab_size=decoder_vocab_size, model=decoder_bpe_model)
                decoder_tokenizer_model = decoder_bpe_model
                decoder_tokenizer = get_tokenizer(
                    tokenizer_name='yttm', tokenizer_model=decoder_tokenizer_model, bpe_dropout=decoder_bpe_dropout
                )

        if use_shared_tokenizer:
            decoder_tokenizer = encoder_tokenizer

        tokens_in_batch = [int(item) for item in tokens_in_batch.split(',')]
        for num_tokens in tokens_in_batch:
            dataset = TranslationDataset(
                dataset_src=str(Path(src_fname).expanduser()),
                dataset_tgt=str(Path(tgt_fname).expanduser()),
                tokens_in_batch=num_tokens,
                clean=clean,
                max_seq_length=max_seq_length,
                min_seq_length=min_seq_length,
                max_seq_length_diff=max_seq_length,
                max_seq_length_ratio=max_seq_length,
                cache_ids=False,
                cache_data_per_node=False,
                use_cache=False,
            )
            print('Batchifying ...')
            dataset.batchify(encoder_tokenizer, decoder_tokenizer)
            start = time.time()
            cached_dataset_fname = self.get_cached_dataset_fname(src_fname, tgt_fname, tokens_in_batch)
            pickle.dump(dataset, open(os.path.join(out_dir, cached_dataset_fname, 'wb')))
            end = time.time()
            print(f'Took {end - start} time to pickle')
            start = time.time()
            dataset = pickle.load(open(os.path.join(out_dir, cached_dataset_fname, 'rb')))
            end = time.time()
            print(f'Took {end - start} time to unpickle')


def train_yttm_bpe(data: List[str], vocab_size: int, model: str):
    """Train YTTM BPE model.

    Args:
        data (List[str]): list of paths of text files to train on
        vocab_size (int): BPE vocab size
        model (str): path to save learned BPE model
    """
    if len(data) > 1:
        out_file = TemporaryFile(mode='w+')
        for filepath in data:
            with open(filepath) as in_file:
                for line in in_file:
                    out_file.write(line)
        BPE.train(data=out_file, vocab_size=vocab_size, model=model)
        out_file.close()

        # with TemporaryDirectory() as tmpdir:
        #     concat_path = os.path.join(tmpdir, 'concat_dataset.txt')
        #     with open(concat_path) as out_file:
        #         for filepath in data:
        #             with open(filepath) as in_file:
        #                 for line in in_file:
        #                     out_file.write(line)
    else:
        BPE.train(data=data[0], vocab_size=vocab_size, model=model)


# @dataclass
# class PreprocessDatasetConfig:
#     encoder_tokenizer_library: str = None
#     encoder_tokenizer_model_name: Optional[str] = None
#     encoder_vocab_size: Optional[int] = None
#     encoder_bpe_dropout: Optional[float] = 0.0
#     decoder_tokenizer_library: str = None
#     decoder_tokenizer_model_name: Optional[str] = None
#     decoder_vocab_size: Optional[int] = None
#     decoder_bpe_dropout: Optional[float] = 0.0
#     use_shared_tokenizer: bool = False
#     clean: bool = False
#     src_fname: str = None
#     tgt_fname: str = None
#     out_dir: str = None
#     max_seq_length: int = 512
#     min_seq_length: int = 1
#     tokens_in_batch: str = '8000,12000,16000,40000'

