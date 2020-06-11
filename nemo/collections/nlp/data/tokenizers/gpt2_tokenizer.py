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

from transformers import GPT2Tokenizer

from nemo.collections.nlp.data.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

__all__ = ['NemoGPT2Tokenizer']


class NemoGPT2Tokenizer(TokenizerSpec):
    def __init__(
        self,
        pretrained_model=None,
        vocab_file=None,
        merges_file=None,
        errors='replace',
        # special_tokens_dict=None
    ):
        if pretrained_model:
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.vocab_size = self.tokenizer.vocab_size
        # special_tokens_dict = {}
        # if self.tokenizer.unk_token is None:
        #     self.tokenizer.unk_token = "<|unk|>"
        #     special_tokens_dict["unk_token"] = "<|unk|>"
        # if self.tokenizer.bos_token is None:
        #     special_tokens_dict["bos_token"] = bos_token
        # if self.tokenizer.eos_token is None:
        #     special_tokens_dict["eos_token"] = eos_token

        # special_tokens_dict["pad_token"] = "<|pad|>"
        # self.tokenizer.add_special_tokens(special_tokens_dict)

    # def add_special_tokens(self, special_tokens_dict):
    #     """
    #     Add a dictionary of special tokens (eos, pad, cls...) to the encoder and link them
    #     to class attributes. If special tokens are NOT in the vocabulary, they are added
    #     to it (indexed starting from the last index of the current vocabulary).
    #     Using `add_special_tokens` will ensure your special tokens can be used in several ways:
    #     - special tokens are carefully handled by the tokenizer (they are never split)
    #     - you can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This makes it easy to develop model-agnostic training and fine-tuning scripts.
    #     When possible, special tokens are already registered for provided pretrained models (ex: BertTokenizer cls_token is already registered to be '[CLS]' and XLM's one is also registered to be '</s>')
    #     Args:
    #         special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
    #             [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
    #             ``additional_special_tokens``].
    #             Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).
    #     Returns:
    #         Number of tokens added to the vocabulary.
    #     """
    #     return self.tokenizer.add_special_tokens(special_tokens_dict)

    def text_to_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        text = self.tokens_to_text(tokens)
        return text

    @property
    def pad_id(self):
        return self.tokens_to_ids([self.tokenizer.pad_token])[0]

    @property
    def bos_id(self):
        return self.tokens_to_ids([self.tokenizer.bos_token])[0]

    @property
    def eos_token(self):
        return self.tokens_to_ids([self.tokenizer.eos_token])[0]

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def bos_token(self):
        return self.tokenizer.bos_token

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def max_len(self):
        return self.tokenizer.max_len
