# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2019 The Google Research Authors.
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
"""
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/data_utils.py
"""

from nemo import logging

__all__ = ['InputExamplePM']

class InputExamplePM(object):
    """An example for training/inference."""

    def __init__(
        self,
        user_utterance,
        system_utterance,
        user_frames,
        system_frames_next,
        tokenizer,
    ):
        """Constructs an InputExample.

        Args:
          max_seq_length: The maximum length of the sequence. Sequences longer than
            this value will be truncated.
          service_schema: A ServiceSchema object wrapping the schema for the service
            corresponding to this example.
          example_id: Unique identifier for the example, like: 'train-1_00000-00-Restaurants_1'
          example_id_num: dialogue_id and turn_id combined and service id combined into a list of ints,
            like: [1, 0, 0, 18]
          is_real_example: Indicates if an example is real or used for padding in a
            minibatch.
          tokenizer (Tokenizer): such as NemoGPT2Tokenizer
        """
        user_special_token = '<|user|>'
        sys_special_token = '<|system|>'

        user_tokens = user_special_token + tokenizer.text_to_tokens(user_utterance)
        self.user_token_ids = tokenizer.tokens_to_ids(user_tokens)
        system_tokens = sys_special_token + tokenizer.text_to_tokens(user_utterance)
        self.system_token_ids = tokenizer.tokens_to_ids(system_tokens)

        token_type_ids = len(user_token_ids) * [tokenizer.tokens_to_ids(user_special_token)] +
                         len(system_token_ids) * [tokenizer.tokens_to_ids(sys_special_token)] +

        import pdb; pdb.set_trace()

        print()




    