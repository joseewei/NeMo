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

"""Pytorch Dataset for training Neural Machine Translation."""

from collections import OrderedDict

import numpy as np

from nemo.collections.nlp.data.data_utils.data_preprocessing import dataset_to_ids
from nemo.core import Dataset

__all__ = ['TranslationOneSideDataset']


class TranslationOneSideDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokens_in_batch=1024,
        clean=False,
        cache_ids=False,
        max_seq_length=512,
        min_seq_length=1,
    ):
        self.dataset = dataset
        self.tokens_in_batch = tokens_in_batch
        self.clean = clean
        self.cache_ids = cache_ids
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
    
    def batchify(self, tokenizer):
        self.pad_id = tokenizer.pad_id
        ids = dataset_to_ids(self.dataset, tokenizer, cache_ids=self.cache_ids)
        if self.clean:
            print('Cleaning data ...')
            ids = self.clean_data(ids, max_tokens=self.max_seq_length, min_tokens=self.min_seq_length)
        print('Packing data into batches ...')
        self.batch_sent_ids, self.batch_elem_lengths = self.pack_data_into_batches(ids)
        print('Padding batches ...')
        self.batches = self.pad_batches()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        ids = self.batches[idx]
        mask = (ids != self.pad_id).astype(np.int32)
        return ids, mask

    def pad_batches(self):
        """
        Augments source and target ids in the batches with padding symbol
        to make the lengths of all sentences in the batches equal.
        """

        batches = []
        for batch_elem_len, batch_sent_ids in zip(self.batch_elem_lengths, self.batch_sent_ids):
            batch = self.pad_id * np.ones((len(batch_sent_ids), batch_elem_len), dtype=np.int)
            for i, sentence in enumerate(batch_sent_ids):
                batch[i][: len(sentence)] = sentence
            batches.append(batch)
        return batches

    def pack_data_into_batches(self, ids):
        """
        Takes two lists of source and target sentences, sorts them, and packs
        into batches to minimize the use of padding tokens. Returns a list of
        batches where each batch contains indices of sentences included into it
        """
        batches = []
        curr_batch = []
        batch_elem_lengths = []
        len_of_longest_sent = -1
        for line_ids in ids:
            if len_of_longest_sent == -1:
                len_of_longest_sent = len(curr_batch)
            if len_of_longest_sent * (len(curr_batch) + 1) > self.tokens_in_batch:
                if not curr_batch:
                    raise ValueError(
                        f"The limitation on number of tokens in batch {self.tokens_in_batch} is too strong."
                        f"Several sentences contain {sent_len} tokens."
                    )
                batches.append(curr_batch)
                batch_elem_lengths.append(len_of_longest_sent)
                curr_batch = []
                len_of_longest_sent = -1
            curr_batch.append(line_ids)
            len_of_longest_sent = max(
                len_of_longest_sent, len(line_ids)
            )
        if curr_batch:
            batches.append(curr_batch)
            batch_elem_lengths.append(len_of_longest_sent)
        return batches, batch_elem_lengths

    def clean_data(self, ids, max_tokens=None, min_tokens=None):
        """
        Cleans source and target sentences to get rid of noisy data.
        Specifically, a pair of sentences is removed if
          -- either source or target is longer than *max_tokens*
          -- either source or target is shorter than *min_tokens*
          -- absolute difference between source and target is larger than
             *max_tokens_diff*
          -- one sentence is *max_tokens_ratio* times longer than the other
        """

        ids_ = []
        for i in range(len(ids)):
            len_ = len(ids[i])
            if (max_tokens is not None and len_ > max_tokens) or (min_tokens is not None and len_ < min_tokens):
                continue
            ids_.append(ids[i])
        return ids_
