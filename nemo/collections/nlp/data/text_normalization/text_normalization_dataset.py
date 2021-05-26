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
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.parts.utils_funcs import list2str
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['TextNormalizationDataset', 'tag_labels']


PUNCT_TYPE = "PUNCT"
PLAIN_TYPE = "PLAIN"
Instance = namedtuple('Instance', 'token_type un_normalized normalized')
tag_labels = {'<self>': 0, 'sil': 1, 'B-I': 2, 'B-M': 3}


def load_files(file_paths: List[str]) -> List[List[Instance]]:
    res = []
    for file_path in file_paths:
        res.extend(load_file(file_path=file_path))
    return res


def load_file(file_path: str) -> List[List[Instance]]:
    """
    load Google data file into list of sentences of instances
    """
    res = []
    sentence = []
    with open(file_path, 'r') as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if parts[0] == "<eos>":
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
        }

    def __init__(
        self,
        input_file: str,
        tokenizer_context: TokenizerSpec,
        tokenizer_encoder: TokenizerSpec,
        tokenizer_decoder: TokenizerSpec,
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
            "cached_{}_{}_{}_{}_{}".format(
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

        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        features = None
        if master_device and (not use_cache or not os.path.exists(features_pkl)):
            if num_samples == 0:
                raise ValueError("num_samples has to be positive", num_samples)

            instances = load_file(input_file)

            features = get_features(
                sentences=instances,
                tokenizer_context=tokenizer_context,
                tokenizer_encoder=tokenizer_encoder,
                tokenizer_decoder=tokenizer_decoder,
            )

            pickle.dump(features, open(features_pkl, "wb"))
            logging.info(f'features saved to {features_pkl}')

        # wait until the master process writes to the processed data files
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if features is None:
            features = pickle.load(open(features_pkl, 'rb'))
            logging.info(f'features restored from {features_pkl}')

        self.tokenizer_context = tokenizer_context
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder
        self.features = features  # list of tuples of sent_ids, tag_ids, unnormalized_id, normalized_id, lefT_context_id, right_context_id,

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
        # max length depends on batch, does not support predefined max length yet, where input might need to be truncated.
        max_length_sent = max([len(batch[i][0]) for i in range(bs)])
        max_length_input = max([len(batch[i][2]) for i in range(bs)])
        max_length_output = max([len(batch[i][3]) for i in range(bs)])

        sent_ids_padded = []
        tag_ids_padded = []
        unnormalized_ids_padded = []
        normalized_ids_padded = []

        for i in range(bs):
            sent_ids, tag_ids, unnormalized_ids, normalized_ids, l_context_id, r_context_id = batch[i]
            len_context[i] = len(sent_ids)
            len_encoder_input[i] = len(unnormalized_ids)
            len_decoder_input[i] = len(normalized_ids)
            l_context_ids[i] = l_context_id
            r_context_ids[i] = r_context_id

            assert len(sent_ids) == len(tag_ids)
            if len(sent_ids) < max_length_sent:
                pad_width = max_length_sent - len(sent_ids)
                sent_ids_padded.append(
                    np.pad(sent_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_context.pad_id)
                )
                tag_ids_padded.append(
                    np.pad(tag_ids, pad_width=[0, pad_width], constant_values=self.tokenizer_context.pad_id)
                )
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

        # sent_ids_padded = pad_sequence(sent_ids, batch_first=False, padding_value=self.tokenizer_context.pad_id)
        # tag_ids_padded = pad_sequence(tag_ids, batch_first=False, padding_value=-1)
        # unnormalized_ids_padded = pad_sequence(unnormalized_ids, batch_first=False, padding_value=self.tokenizer_encoder.pad_id)
        # normalized_ids_padded = pad_sequence(normalized_ids, batch_first=False, padding_value=self.tokenizer_encoder.pad_id)

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
        )


def get_features(
    sentences: List[List[Instance]],
    tokenizer_context: TokenizerSpec,
    tokenizer_encoder: TokenizerSpec,
    tokenizer_decoder: TokenizerSpec,
) -> List[tuple]:
    """
    data processing from list of instances into tuples 
    ultimately (tokenized sentence ids with <s> and </s>, tag ids, [left context id, right_context_id, (tokenizer encoder ids with <s>), (tokenized decoder ids with </s>))])

    returns for now:
    List of (sent_ids: List, tag_ids: List, unnormalized_ids: List, normalized_ids: List, left_context_id:int, right_context_id:int)
    """

    def process_sentence(sentence: List[Instance]):
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
        tag_ids.extend([tag_labels['<self>']] * len(tokens))

        for instance in sentence:
            if instance.token_type == PLAIN_TYPE:
                tokens = tokenizer_context.text_to_ids(instance.un_normalized)
                sent_ids.extend(tokens)
                tag_ids.extend([tag_labels['<self>']] * len(tokens))
            elif instance.token_type == PUNCT_TYPE:
                tokens = tokenizer_context.text_to_ids(instance.un_normalized)
                assert (len(tokens) == 1, "punctuation should be tokenized to single token")
                sent_ids.extend(tokens)
                tag_ids.extend([tag_labels['sil']] * len(tokens))
            else:
                # semiotic token
                tokens = tokenizer_context.text_to_ids(instance.un_normalized)
                # TODO rename left_context_ids and right_context_ids - to show it's a starting idx
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
        tag_ids.extend([tag_labels['<self>']] * len(tokens))

        features = [
            (sent_ids, tag_ids, unnormalized_ids[i], normalized_ids[i], left_context_ids[i], right_context_ids[i])
            for i in range(len(unnormalized_ids))
        ]

        if len(features) > 0:
            print([instance.un_normalized for instance in sentence])
            names = ['send_ids', 'tag_ids', 'unnormalized_ids', 'normalized_ids', 'left_context_ids', 'right_context_ids']
            for i, f in enumerate(features[0]):
                print(f'{names[i]}: {f}')
        return features

    features = []
    for sentence in sentences:
        features.extend(process_sentence(sentence))
    return features
