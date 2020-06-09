# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import os
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data as pt_data

from nemo.backends.pytorch import DataLayerNM
from nemo.collections.nlp.data import BertPretrainingDataset, BertPretrainingPreprocessedDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, DeviceType, LabelsType, MaskType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['GPT2DataLayer']


class GPT2DataLayer(TextDataLayer):
    """
    Data layer for masked language modeling task for text data.

    Args:
        tokenizer (TokenizerSpec): tokenizer
        dataset (str): directory or a single file with dataset documents
        max_seq_length (int): maximum allowed length of the text segments
        mask_probability (float): probability of masking input sequence tokens
        batch_size (int): batch size in segments
        short_seeq_prob (float): Probability of creating sequences which are
            shorter than the maximum length.
            Defaults to 0.1.
        shuffle (bool): whether to shuffle data or not. Default: False.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        input_ids:
            indices of tokens which constitute batches of masked text segments
        input_type_ids:
            tensor with 0's and 1's to denote the text segment type
        input_mask:
            bool tensor with 0s in place of tokens to be masked
        output_ids: indices of tokens which constitute batches of unmasked text segments
        output_mask: bool tensor with 0s in place of tokens to be masked
        labels: 0 or 1 for next sentence prediction classification
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            # "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            # "input_mask": NeuralType(('B', 'T'), ChannelType()),
            # "output_ids": NeuralType(('B', 'T'), LabelsType()),
            # "output_mask": NeuralType(('B', 'T'), MaskType()),
            # "labels": NeuralType(tuple('B'), LabelsType()),
        }

    def __init__(
        self,
        tokenizer,
        dataset_type,
        file_path,
        block_size,
        batch_size,
        overwrite_cache=False,
        shuffle=False,
        num_workers=0,
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'file_path': file_path,
            'block_size': block_size,
            'overwrite_cache': overwrite_cache,
        }

        super().__init__(dataset_type, dataset_params, batch_size=batch_size, shuffle=shuffle)
        self.tokenizer = tokenizer
        if self._placement == DeviceType.AllGpu:
            sampler = pt_data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = pt_data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            sampler=sampler,
        )

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader

    def _collate_fn(self, examples):
        """
        Data collator used for language modeling.
        - collates batches of tensors, honoring their tokenizer's pad_token
        - preprocesses batches for masked language modeling

        source: transformers/data/data_collator.py
        """

        def _tensorize_batch(examples):
            length_of_first = examples[0].size(0)
            are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
            if are_tensors_same_length:
                return torch.stack(examples, dim=0)
            else:
                if self.tokenizer.pad_token is None:
                    raise ValueError(
                        "You are attempting to pad samples but the tokenizer you are using"
                        f" ({self.tokenizer.__class__.__name__}) does not have one."
                    )
                return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_id)

        batch = _tensorize_batch(examples)
        import pdb

        pdb.set_trace()
        return batch
