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

from argparse import ArgumentParser

import torch

import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--text2translate", type=str, required=True, help="")
    parser.add_argument("--output", type=str, required=True, help="")
    parser.add_argument("--max_length", type=int, default=250, help="")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = nemo_nlp.models.TransformerMTModel.restore_from(restore_path=args.model)
    elif args.model.endswith(".ckpt"):
        logging.info("Attempting to initialize from .ckpt file")
        model = nemo_nlp.models.TransformerMTModel.load_from_checkpoint(checkpoint_path=args.model)
    
    model.replace_beam_with_sampling()
    if torch.cuda.is_available():
        model = model.cuda()

    logging.info(f"Translating: {args.text2translate}")
    with open(args.text2translate, 'r') as fin, open(args.output + '.src', 'w') as fout_src, open(args.output + '.tgt', 'w') as fout_tgt:
        lines = []
        for idx, line in enumerate(fin):
            if len(line.strip().split()) > args.max_length:
                continue
            lines.append(line.strip())
            if idx % 100 == 0 and idx !=0:
                translations = model.batch_translate(text=lines)
                if translations is None:
                    print('Warning! Translations returned None ...')
                    continue
                for tgt, src in zip(translations, lines):
                    fout_src.write(src + "\n")
                    fout_tgt.write(tgt + "\n")
                lines = []
    logging.info("all done")

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
