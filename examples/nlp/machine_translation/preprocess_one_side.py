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

import argparse
import os
from pathlib import Path
import pickle
import time

from nemo.collections.nlp.data import TranslationOneSideDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT dataset pre-processing')
    parser.add_argument('--clean', action="store_true",
                help='Whether to clean dataset based on length diff')
    parser.add_argument('--bpe_dropout', type=float, default=0.0,
                help='Whether to share encoder/decoder tokenizers')
    parser.add_argument('--fname', type=str, required=True,
                help='Path to the source file')
    parser.add_argument('--out_fname', type=str, required=True,
                help='Path to store dataloader and tokenizer models')
    parser.add_argument('--max_seq_length', type=int, default=512,
                help='Max Sequence Length')
    parser.add_argument('--min_seq_length', type=int, default=1,
                help='Min Sequence Length')
    parser.add_argument('--tokens_in_batch', type=int, default=8000,
                help='# Tokens per batch per GPU')
    parser.add_argument('--tokenizer_model', type=str, default="8000,12000,16000,40000",
                help='# Tokens per batch per GPU')

    args = parser.parse_args()
    tokenizer = get_tokenizer(
        tokenizer_name='yttm',
        tokenizer_model=args.tokenizer_model,
        bpe_dropout=args.bpe_dropout
    )

    dataset = TranslationOneSideDataset(
        dataset=str(Path(args.fname).expanduser()),
        tokens_in_batch=args.tokens_in_batch,
        clean=args.clean,
        max_seq_length=args.max_seq_length,
        min_seq_length=args.min_seq_length,
        cache_ids=False,
    )
    print('Batchifying ...')
    dataset.batchify(tokenizer)
    start = time.time()
    pickle.dump(
        dataset,
        open(args.out_fname, 'wb')
    )
    end = time.time()
    print('Took %f time to pickle' % (end - start))
