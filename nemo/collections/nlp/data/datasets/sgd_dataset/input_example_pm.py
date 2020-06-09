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
        self, user_utterance, system_utterance, user_frames, system_frames_next, tokenizer):
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
        context_start = '<|context|>'
        context_end = '<|endofcontext|>'
        belief_start = '<|belief|>'
        belief_end = '<|endofbelief|>'
        action_start = '<|action|>'
        action_end = '<|endofaction|>'

        user_tokens =  [user_special_token] + tokenizer.text_to_tokens(user_utterance)
        system_tokens =  [sys_special_token] + tokenizer.text_to_tokens(system_utterance)
        
        dialogue_history = [context_start] + system_tokens + user_tokens + [context_end]
        self.token_ids = tokenizer.tokens_to_ids(dialogue_history)
        self.token_type_ids = (len(user_tokens) + 1) * [tokenizer.tokens_to_ids(user_special_token)] + \
            (len(system_tokens) + 1) * [tokenizer.tokens_to_ids(sys_special_token)]

        assert len(self.token_type_ids) == len(dialogue_history)
        self.dialogue_history = tokenizer.tokens_to_text(dialogue_history)

        # extract belief state - during inference use SGD tracker instead
        dialogue_belief = belief_start
        for service in user_frames:
            belief = service + ', '

            state = user_frames[service]['state']
            active_intent = 'intent = ' + self.split_intent(state['active_intent']) + ', '
          
            slots = ''
            for slot_name, slot_values in state['slot_values'].items():
                slots += slot_name + ' = ' + ','.join(slot_values) + ', '

            requested_slots = 'requested_slots = '
            requested_slots += ','.join(state['requested_slots']) if len(state['requested_slots']) > 0 else 'none'
            
            belief += active_intent + slots + requested_slots + belief_end

            dialogue_belief += belief + belief_end

        # add belief to token_ids and token_type_ids
        belief_tokens_ids = tokenizer.tokens_to_ids(tokenizer.text_to_tokens(dialogue_belief))
        self.token_ids += belief_tokens_ids
        self.token_type_ids += len(belief_tokens_ids) * [tokenizer.tokens_to_ids(belief_start)]

        print('\n', dialogue_belief)

        """
        {'Restaurants_1': 
        {'actions': [{'act': 'INFORM', 'canonical_values': ['American'], 'slot': 'cuisine', 'values': ['American']}], 
         'service': 'Restaurants_1', 'slots': [{'exclusive_end': 34, 'slot': 'cuisine', 'start': 26}],
         'state': {'active_intent': 'FindRestaurants', 'requested_slots': [], 
         'slot_values': {'city': ['San Jose'], 'cuisine': ['American']}}}
        }

        <|belief|> Restaurants_1, intent = FindRestaurants, cuisine = American, city = San Jose, requested_slots = none <|endofbelief|>
        for service in system_frames_next:
        service + active_intent + slots + requested_slots

        {'Restaurants_1': {'actions': [{'act': 'REQUEST', 'canonical_values': [], 'slot': 'city', 'values': []}], 'service': 'Restaurants_1', 'slots': []}}
        """
        print (system_frames_next)
        '''
        add system action
        possible system acts
        INFORM, REQUEST, CONFIRM, OFFER, NOTIFY_SUCCESS, NOTIFY_FAILURE, INFORM_COUNT, OFFER_INTENT, REQ_MORE, GOODBYE 
        '''
        system_acts = action_start
        for service in system_frames_next:
            system_act = ''
            for act in system_frames_next[service]['actions']:
                
                act_name = (act['act'].lower() + ' ').replace('_', ' ') # turn NOTIFY_SUCCESS into notify success
                act_slots = act['slot'] + ' ' + ','.join(act['values']) + ', '
                system_act += (act_name + act_slots).strip()

            # remove trailing comma
            system_acts += ' ' + system_act[:-1] if len(system_act) > 0 else system_act
            system_acts = system_acts.strip()
        system_acts += action_end
        
        print ('\n-------->', system_acts)       
        import pdb; pdb.set_trace()
        print()   

    def split_intent(self, intent):
        reformatted_intent = ''
        for ch in intent:
            if ch.isupper():
                reformatted_intent += ' ' + ch.lower()
            else:
                reformatted_intent += ch
        return reformatted_intent.strip()
