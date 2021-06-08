# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
import pickle
import random
from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.parts.utils_funcs import list2str
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['TextNormalizationDataset', 'TextNormalizationTestDataset']


PUNCT_TYPE = "PUNCT"
PLAIN_TYPE = "PLAIN"
Instance = namedtuple('Instance', 'token_type un_normalized normalized')
tag_labels = {'O-I': 0, 'O-M': 1, 'B-I': 2, 'B-M': 3}


def load_files(file_paths: List[str]) -> List[List[Instance]]:
    res = []
    for file_path in file_paths:
        res.extend(load_file(file_path=file_path))
    return res


def load_file(file_path: str, max_sentence_length: Optional[int] = None) -> List[List[Instance]]:
    """
    load Google data file into list of sentences of instances
    """
    res = []
    sentence = []
    with open(file_path, 'r') as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if parts[0] == "<eos>":
                if max_sentence_length is None or len(sentence) < max_sentence_length:
                    res.append(sentence)
                sentence = []
            else:
                l_type, l_token, l_normalized = parts
                if l_type in [PUNCT_TYPE, PLAIN_TYPE]:
                    sentence.append(Instance(token_type=l_type, un_normalized=l_token, normalized=l_token))
                else:
                    sentence.append(Instance(token_type=l_type, un_normalized=l_token, normalized=l_normalized))
    return res


class TextNormalizationDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'context_ids': NeuralType(('B', 'T'), ChannelType()),
            'tag_ids': NeuralType(('B', 'T'), LabelsType()),
            'len_context': NeuralType(('B'), LengthsType()),
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'len_input': NeuralType(('B'), LengthsType()),
            'output_ids': NeuralType(('B', 'T'), LabelsType()),
            'len_output': NeuralType(('B'), LengthsType()),
            'l_context_ids': NeuralType(('B'), ChannelType()),
            'r_context_ids': NeuralType(('B'), ChannelType()),
            'example_id': NeuralType(('B'), ChannelType()),
        }

    def __init__(
        self,
        input_file: str,
        tokenizer_context: TokenizerSpec,
        tokenizer_encoder: TokenizerSpec,
        tokenizer_decoder: TokenizerSpec,
        max_sentence_length: Optional[int] = None,
        num_samples: int = -1,
        use_cache: bool = True,
    ):

        data_dir = os.path.dirname(input_file)

        context_vocab_size = getattr(tokenizer_context, "vocab_size", 0)
        encoder_vocab_size = getattr(tokenizer_encoder, "vocab_size", 0)
        decoder_vocab_size = getattr(tokenizer_decoder, "vocab_size", 0)
        filename = os.path.basename(input_file)
        features_pkl = os.path.join(
            data_dir,
            "cached_TextNormalizationDataset_{}_{}_{}_{}_maxlen_{}".format(
                filename,
                tokenizer_context.name,
                str(context_vocab_size),
                tokenizer_encoder.name,
                str(encoder_vocab_size),
                tokenizer_decoder.name,
                str(decoder_vocab_size),
                max_sentence_length,
            ),
        )

        self.tokenizer_context = tokenizer_context
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder
        instances = load_file(input_file, max_sentence_length=max_sentence_length)
        self.examples = [
            [(instance.token_type, instance.normalized) for instance in sentence] for sentence in instances
        ]
        if use_cache and os.path.exists(features_pkl):
            logging.info(f"loading from {features_pkl}")
            with open(features_pkl, "rb") as reader:
                self.features = pickle.load(reader)
        else:
            features = get_features(
                sentences=instances,
                tokenizer_context=tokenizer_context,
                tokenizer_encoder=tokenizer_encoder,
                tokenizer_decoder=tokenizer_decoder,
                mode="train",
            )

            self.features = features  # list of tuples of sent_ids, tag_ids, unnormalized_id, normalized_id, lefT_context_id, right_context_id,

            if use_cache:
                master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                if master_device:
                    logging.info("  Saving train features into cached file %s", features_pkl)
                    with open(features_pkl, "wb") as writer:
                        pickle.dump(self.features, writer)

                # wait until the master process writes to the processed data files
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            np.array(self.features[idx][0]),
            np.array(self.features[idx][1]),
            np.array(self.features[idx][2]),
            np.array(self.features[idx][3]),
            np.array(self.features[idx][4]),
            np.array(self.features[idx][5]),
            np.array(self.features[idx][6]),
        )

    def _collate_fn(self, batch):
        """collate batch of sent_ids, tag_ids, unnormalized_ids, normalized_ids l_context_id r_context_id
        """

        bs = len(batch)
        # initialize
        len_context = [0 for _ in range(bs)]
        len_encoder_input = [0 for _ in range(bs)]
        len_decoder_input = [0 for _ in range(bs)]
        l_context_ids = [0 for _ in range(bs)]
        r_context_ids = [0 for _ in range(bs)]
        example_ids = [0 for _ in range(bs)]
        # max length depends on batch, does not support predefined max length yet, where input might need to be truncated.
        max_length_sent = max([len(batch[i][0]) for i in range(bs)])
        max_length_input = max([len(batch[i][2]) for i in range(bs)])
        max_length_output = max([len(batch[i][3]) for i in range(bs)])

        sent_ids_padded = []
        tag_ids_padded = []
        unnormalized_ids_padded = []
        normalized_ids_padded = []

        for i in range(bs):
            sent_ids, tag_ids, unnormalized_ids, normalized_ids, l_context_id, r_context_id, example_id = batch[i]
            len_context[i] = len(sent_ids)
            len_encoder_input[i] = len(unnormalized_ids)
            len_decoder_input[i] = len(normalized_ids)
            l_context_ids[i] = l_context_id
            r_context_ids[i] = r_context_id
            example_ids[i] = example_id

            assert len(sent_ids) == len(tag_ids)
            if len(sent_ids) < max_length_sent:
                pad_width = max_length_sent - len(sent_ids)
                sent_ids_padded.append(
                    np.pad(sent_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_context.pad_id)
                )
                tag_ids_padded.append(np.pad(tag_ids, pad_width=[0, pad_width], constant_values=-1))
            else:
                sent_ids_padded.append(sent_ids)
                tag_ids_padded.append(tag_ids)

            if len(unnormalized_ids) < max_length_input:
                pad_width = max_length_input - len(unnormalized_ids)
                unnormalized_ids_padded.append(
                    np.pad(unnormalized_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_encoder.pad_id)
                )
            else:
                unnormalized_ids_padded.append(unnormalized_ids)

            if len(normalized_ids) < max_length_output:
                pad_width = max_length_output - len(normalized_ids)
                normalized_ids_padded.append(
                    np.pad(normalized_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_decoder.pad_id)
                )
            else:
                normalized_ids_padded.append(normalized_ids)

        return (
            torch.LongTensor(sent_ids_padded),
            torch.LongTensor(tag_ids_padded),
            torch.LongTensor(len_context),
            torch.LongTensor(unnormalized_ids_padded),
            torch.LongTensor(len_encoder_input),
            torch.LongTensor(normalized_ids_padded),
            torch.LongTensor(len_decoder_input),
            torch.LongTensor(np.asarray(l_context_ids)),
            torch.LongTensor(np.asarray(r_context_ids)),
            torch.LongTensor(np.asarray(example_ids)),
        )

    def evaluate(self, predictions):
        """
        Args:
            predictions: dict of example_id to tokens
        """

        num_samples = len(predictions)
        ids = predictions.keys()
        predictions = [" ".join(predictions[i]) for i in ids]
        gt = [" ".join([x[1] for x in self.examples[i]]) for i in ids]

        accuracy = accuracy_score(gt, predictions)
        return accuracy


class TextNormalizationTestDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'context_ids': NeuralType(('B', 'T'), ChannelType()),
            'len_context': NeuralType(('B'), LengthsType()),
            'example_id': NeuralType(('B'), ChannelType()),
        }

    def __init__(
        self,
        input_file: str,
        tokenizer_context: TokenizerSpec,
        tokenizer_encoder: TokenizerSpec,
        tokenizer_decoder: TokenizerSpec,
        max_sentence_length: Optional[int] = None,
        num_samples: int = -1,
        use_cache: bool = True,
    ):

        data_dir = os.path.dirname(input_file)

        context_vocab_size = getattr(tokenizer_context, "vocab_size", 0)
        encoder_vocab_size = getattr(tokenizer_encoder, "vocab_size", 0)
        decoder_vocab_size = getattr(tokenizer_decoder, "vocab_size", 0)
        filename = os.path.basename(input_file)
        features_pkl = os.path.join(
            data_dir,
            "cached_TextNormalizationTestDataset_{}_{}_{}_{}_maxlen_{}".format(
                filename,
                tokenizer_context.name,
                str(context_vocab_size),
                tokenizer_encoder.name,
                str(encoder_vocab_size),
                tokenizer_decoder.name,
                str(decoder_vocab_size),
                str(num_samples),
            ),
        )
        self.tokenizer_context = tokenizer_context
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder
        instances = load_file(input_file, max_sentence_length=max_sentence_length)
        self.examples = [
            [(instance.token_type, instance.normalized) for instance in sentence] for sentence in instances
        ]
        if use_cache and os.path.exists(features_pkl):
            logging.info(f"loading from {features_pkl}")
            with open(features_pkl, "rb") as reader:
                self.features = pickle.load(reader)
        else:
            features = get_features(
                sentences=instances,
                tokenizer_context=tokenizer_context,
                tokenizer_encoder=tokenizer_encoder,
                tokenizer_decoder=tokenizer_decoder,
                mode="test",
            )

            self.features = features  # list of tuples of sent_ids, tag_ids, unnormalized_id, normalized_id, lefT_context_id, right_context_id,

            if use_cache:
                master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                if master_device:
                    logging.info("  Saving train features into cached file %s", features_pkl)
                    with open(features_pkl, "wb") as writer:
                        pickle.dump(self.features, writer)

                # wait until the master process writes to the processed data files
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            np.array(self.features[idx][0]),
            np.array(self.features[idx][1]),
        )

    def _collate_fn(self, batch):
        """collate batch of sent_ids, tag_ids, unnormalized_ids, normalized_ids l_context_id r_context_id
        """

        bs = len(batch)
        # initialize
        len_context = [0 for _ in range(bs)]
        example_ids = [0 for _ in range(bs)]
        # max length depends on batch, does not support predefined max length yet, where input might need to be truncated.
        max_length_sent = max([len(batch[i][0]) for i in range(bs)])

        sent_ids_padded = []

        for i in range(bs):
            sent_ids, example_id = batch[i]
            len_context[i] = len(sent_ids)
            example_ids[i] = example_id

            if len(sent_ids) < max_length_sent:
                pad_width = max_length_sent - len(sent_ids)
                sent_ids_padded.append(
                    np.pad(sent_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_context.pad_id)
                )
            else:
                sent_ids_padded.append(sent_ids)

        return (
            torch.LongTensor(sent_ids_padded),
            torch.LongTensor(len_context),
            torch.LongTensor(np.asarray(example_ids)),
        )

    def evaluate(self, predictions):
        """
        Args:
            predictions: dict of example_id to tokens
        """

        num_samples = len(predictions)
        ids = predictions.keys()
        assert (
            num_samples == len(self.examples),
            "no. predictions should match original dataset length for test and inference",
        )
        predictions = [" ".join(predictions[i]) for i in ids]
        gt = [" ".join([x[1] for x in self.examples[i]]) for i in ids]

        accuracy = accuracy_score(gt, predictions)
        return accuracy


def get_features(
    sentences: List[List[Instance]],
    tokenizer_context: TokenizerSpec,
    tokenizer_encoder: TokenizerSpec,
    tokenizer_decoder: TokenizerSpec,
    mode: str,
) -> List[tuple]:
    """
    data processing from list of instances into tuples 
    ultimately (tokenized sentence ids with <s> and </s>, tag ids, [left context id, right_context_id, (tokenizer encoder ids with <s>), (tokenized decoder ids with </s>))])

    returns for now:
    List of (sent_ids: List, tag_ids: List, unnormalized_ids: List, normalized_ids: List, left_context_id:int, right_context_id:int)
    """

    def process_sentence(sentence: List[Instance], example_id: int):
        # unnormalized_ids: <BOS> <EOS>
        # normalized_ids <BOS>, word ids .., <EOS>
        # return list of unnormalized_ids, normalized_ids, l_context_id, r_context_id, sent_ids, tag_ids

        sent_ids = []  # list
        tag_ids = []  # list
        unnormalized_ids = []  # list of list
        normalized_ids = []  # list of list
        left_context_ids = []  # list of int
        right_context_ids = []  # list of int

        # start of sentence, needs to be only single token
        tokens = [tokenizer_context.bos_id]
        assert (len(tokens) == 1, "BOS should be tokenized to single token")
        sent_ids.extend(tokens)
        tag_ids.extend([tag_labels['O-I']] * len(tokens))

        for instance in sentence:
            if instance.token_type in [PLAIN_TYPE, PUNCT_TYPE]:
                tokens = tokenizer_context.text_to_ids(instance.un_normalized)
                sent_ids.extend(tokens)
                tag_ids.append(tag_labels['O-I'])
                if len(tokens) > 1:
                    tag_ids.extend([tag_labels['O-M']] * (len(tokens) - 1))
            else:
                # semiotic token
                tokens = tokenizer_context.text_to_ids(instance.un_normalized)
                left_context_ids.append(len(sent_ids) - 1)  # exclusive of this token
                sent_ids.extend(tokens)
                right_context_ids.append(len(sent_ids))  # exclusive of this token
                tag_ids.append(tag_labels['B-I'])
                if len(tokens) > 1:
                    tag_ids.extend([tag_labels['B-M']] * (len(tokens) - 1))
                unnormalized_ids.append(
                    [tokenizer_encoder.bos_id] + tokenizer_encoder.text_to_ids(instance.un_normalized)
                )
                normalized_ids.append(tokenizer_decoder.text_to_ids(instance.normalized) + [tokenizer_decoder.eos_id])

        # start of sentence, needs to be only single token
        tokens = [tokenizer_context.eos_id]
        assert (len(tokens) == 1, "EOS should be tokenized to single token")
        sent_ids.extend(tokens)
        tag_ids.extend([tag_labels['O-I']] * len(tokens))

        if mode == "train":
            features = [
                (
                    sent_ids,
                    tag_ids,
                    unnormalized_ids[i],
                    normalized_ids[i],
                    left_context_ids[i],
                    right_context_ids[i],
                    example_id,
                )
                for i in range(len(unnormalized_ids))
            ]
        elif mode == "test":
            features = [(sent_ids, example_id,)]

        return features

    features = []
    for example_id, sentence in enumerate(sentences):
        features.extend(process_sentence(sentence=sentence, example_id=example_id))
    return features
