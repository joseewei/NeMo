import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, List, Optional

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
from nemo.collections.nlp.data.text_normalization.text_normalization_dataset import (
    PLAIN_TYPE,
    PUNCT_TYPE,
    Instance,
    load_file,
)


def main(args):
    sentences_instances = load_file(args.file)
    context = [[x.un_normalized for x in sentence] for sentence in sentences_instances]
    context_string = "\n".join([" ".join(sentence) for sentence in context])

    input = [
        [x.un_normalized for x in sentence if x.token_type not in [PLAIN_TYPE, PUNCT_TYPE]]
        for sentence in sentences_instances
    ]
    input_string = "\n".join([" ".join(sentence) for sentence in input])

    target = [
        [x.normalized for x in sentence if x.token_type not in [PLAIN_TYPE, PUNCT_TYPE]]
        for sentence in sentences_instances
    ]
    target_string = "\n".join([" ".join(sentence) for sentence in target])

    context_file = os.path.join(args.output_dir, "context.txt")
    input_file = os.path.join(args.output_dir, "input.txt")
    target_file = os.path.join(args.output_dir, "output.txt")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(context_file, 'w') as fp:
        fp.write(context_string)

    with open(input_file, 'w') as fp:
        fp.write(input_string)

    with open(target_file, 'w') as fp:
        fp.write(target_string)

    create_spt_model(
        data_file=context_file,
        vocab_size=args.context_vocab,
        sample_size=-1,
        do_lower_case=False,
        tokenizer_type='bpe',
        output_dir=args.output_dir + '/context_tok',
        bos=True,
        eos=True,
        pad=True,
    )
    create_spt_model(
        data_file=input_file,
        vocab_size=args.input_vocab,
        sample_size=-1,
        do_lower_case=False,
        tokenizer_type='char',
        output_dir=args.output_dir + '/input_tok',
        bos=True,
        eos=True,
        pad=True,
    )
    create_spt_model(
        data_file=target_file,
        vocab_size=args.target_vocab,
        sample_size=-1,
        do_lower_case=False,
        tokenizer_type='word',
        output_dir=args.output_dir + '/target_tok',
        bos=True,
        eos=True,
        pad=True,
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--file", type=str, help="training file path", default="/home/yzhang/data/nlp/text_norm/dataset/train.txt"
    )
    parser.add_argument("--context_vocab", type=int, help="context vocab size", default=32000)
    parser.add_argument("--target_vocab", type=int, help="target vocab size", default=1000)
    parser.add_argument("--input_vocab", type=int, help="input vocab size", default=1000)
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory for newly creately files",
        default="/home/yzhang/data/nlp/text_norm/dataset/tmp3",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
