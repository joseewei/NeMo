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
        self, user_utterance, system_utterance, system_utterance_next, user_frames, system_frames_next, tokenizer):
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
        response_start = "<|response|>"
        response_end = "<|endofresponse|>"

        # process dialogue_history and add it to token_ids and to token_type_ids
        system_tokens = [tokenizer.bos_token, context_start]
        system_tokens +=  [sys_special_token] + tokenizer.text_to_tokens(system_utterance) if len(system_utterance) > 0 else []
        user_tokens =  [user_special_token] + tokenizer.text_to_tokens(user_utterance) + [context_end]
        
        dialogue_history_tokens =  system_tokens + user_tokens
        self.token_ids = tokenizer.tokens_to_ids(dialogue_history_tokens)
        self.token_type_ids = len(user_tokens) * [tokenizer.tokens_to_ids(user_special_token)] + \
            len(system_tokens) * [tokenizer.tokens_to_ids(sys_special_token)]

        assert len(self.token_type_ids) == len(dialogue_history_tokens)
        self.dialogue_history = tokenizer.tokens_to_text(dialogue_history_tokens)

        # extract belief state - during inference use SGD tracker instead
        dialogue_belief = belief_start + ' '
        for service in user_frames:
            belief = service + ', '

            state = user_frames[service]['state']
            active_intent = 'intent = ' + self.split_intent(state['active_intent']) + ', '
          
            slots = ''
            for slot_name, slot_values in state['slot_values'].items():
                slots += slot_name + ' = ' + ','.join(slot_values) + ', '

            requested_slots = 'requested_slots = '
            requested_slots += ','.join(state['requested_slots']) if len(state['requested_slots']) > 0 else 'none'
            
            belief += active_intent + slots + requested_slots

            dialogue_belief += belief
        dialogue_belief += ' ' + belief_end
        self.dialogue_belief = dialogue_belief
        # add dialogue belief to token_ids and token_type_ids
        belief_tokens_ids = tokenizer.tokens_to_ids(tokenizer.text_to_tokens(dialogue_belief))
        self.token_ids += belief_tokens_ids
        self.token_type_ids += len(belief_tokens_ids) * [tokenizer.tokens_to_ids(belief_start)]
        
        # process system action, possible system acts: 
        # INFORM, REQUEST, CONFIRM, OFFER, NOTIFY_SUCCESS, NOTIFY_FAILURE, INFORM_COUNT, OFFER_INTENT, REQ_MORE, GOODBYE 
        self.use_external_service = {}
        self.service_results = {}
        self.service_call = {}
        system_acts = action_start
        for service in system_frames_next:
            # add DB/service to lexilize the response
            if 'service_call' in system_frames_next[service]:
                self.use_external_service[service] = True
                self.service_call[service] = system_frames_next[service]['service_call']
                self.service_results[service] = system_frames_next[service]['service_results']
            else:
                self.use_external_service[service] = False
                self.service_call[service] = None
                self.service_results[service] = None

            system_act = ''
            for act in system_frames_next[service]['actions']:
                # turn NOTIFY_SUCCESS into notify success
                act_name = (act['act'].lower() + ' ').replace('_', ' ') 
                act_slots = act['slot'] + ' ' + '|'.join(act['values']) + ', '
                system_act += act_name + act_slots

            # remove trailing comma
            system_act = system_act.strip()
            system_acts += ' ' + system_act[:-1] if len(system_act) > 0 else system_act 
            system_acts = system_acts.strip()
        system_acts += ' ' + action_end

        self.system_acts = system_acts
        self.delex_system_acts = self.delexilize(system_acts, system_frames_next)
        # add delex system acts to token_ids and token_type_ids
        delex_system_acts_tokens_ids = tokenizer.tokens_to_ids(tokenizer.text_to_tokens(self.delex_system_acts))
        self.token_ids += delex_system_acts_tokens_ids
        self.token_type_ids += len(delex_system_acts_tokens_ids) * [tokenizer.tokens_to_ids(action_start)]

        # process system response
        self.response = response_start + system_utterance_next + response_end
        self.delex_response = self.delexilize(self.response, system_frames_next)
        # add delex system response to token_ids and token_type_ids
        delex_response_tokens_ids = tokenizer.tokens_to_ids(tokenizer.text_to_tokens(self.delex_response + tokenizer.eos_token))
        self.token_ids += delex_response_tokens_ids
        self.token_type_ids += len(delex_response_tokens_ids) * [tokenizer.tokens_to_ids(response_start)]

        assert len(self.token_ids) == len(self.token_type_ids)
        # TODO add bos and eos tokens
        print ('-->', tokenizer.tokens_to_text(tokenizer.ids_to_tokens(self.token_ids)))
        print()
        print(self.dialogue_history)
        print(self.dialogue_belief)
        print(self.delex_system_acts)
        print(self.delex_response)

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

    def delexilize(self, uttr, frame):
        """
        Delexilizes utterance
        Args:
            uttr (str): An agent utterance
            frame (dict): A dialogue frame is the SGD format
        Returns:
            uttr (str): delexilized utterance
        Example:
            I see that at 71 Saint Peter there is a good American restaurant which is in San Jose.
            I see that at [value_restaurant_name] there is a good [value_cuisine] restaurant which is in [value_city].
        """
        # delex slot values found in actions
        for v in frame.values():
            if 'actions' in v:
                for action in v['actions']:
                    for slot_value in action['values']:
                        if slot_value in uttr:
                            uttr = uttr.replace(slot_value, '[value_' + action['slot'] + ']')

        # delex slot_values from DB search results
        for v in frame.values():
            if 'service_results' in v:
                for service_result in v['service_results']:
                    for slot_name, slot_value in service_result.items():
                        uttr = uttr.replace(slot_value, '[value_' + slot_name + ']')
        return uttr


    # def remove_action_slots_from_uttr(self, uttr, frame):
    #     """
    #     Delexilizes utterance
    #     Args:
    #         uttr (str): An agent utterance
    #         frame (dict): A dialogue frame is the SGD format
    #     Returns:
    #         uttr (str): delexilized utterance
    #     Example:
    #         I see that at 71 Saint Peter there is a good American restaurant which is in San Jose.
    #         I see that at [value_restaurant_name] there is a good [value_cuisine] restaurant which is in [value_city].
    #     """
    #     # delex slot values found in actions
    #     for v in frame.values():
    #         if 'actions' in v:
    #             for action in v['actions']:
    #                 for slot_value in action['values']:
    #                     if slot_value in uttr:
    #                         uttr = uttr.replace(slot_value, '[value_' + action['slot'] + ']')
    #     return uttr

        
    # def remove_db_results_from_uttr(self, uttr, frame):
    #     # delex slot_values from DB search results
    #     for v in frame.values():
    #         if 'service_results' in v:
    #             for service_result in v['service_results']:
    #                 for slot_name, slot_value in service_result.items():
    #                     uttr = uttr.replace(slot_value, '[value_' + slot_name + ']')
    #     return uttr