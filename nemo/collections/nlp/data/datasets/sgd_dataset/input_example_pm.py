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

from nemo.utils import logging
import re

__all__ = ['InputExamplePM']


class InputExamplePM(object):
    """An example for training/inference."""

    def __init__(
        self, user_utterance, system_utterance, system_utterance_next, user_frames, system_frames_next, tokenizer
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
        system_tokens += (
            [sys_special_token] + tokenizer.text_to_tokens(system_utterance) if len(system_utterance) > 0 else []
        )
        user_tokens = [user_special_token] + tokenizer.text_to_tokens(user_utterance) + [context_end]

        dialogue_history_tokens = system_tokens + user_tokens
        self.token_ids = tokenizer.tokens_to_ids(dialogue_history_tokens)
        self.token_type_ids = len(user_tokens) * [tokenizer.tokens_to_ids(user_special_token)] + len(system_tokens) * [
            tokenizer.tokens_to_ids(sys_special_token)
        ]

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
        self.use_external_service = False
        self.service_results = None
        self.service_call = None
        
        system_acts = ''
        for service in system_frames_next:
            # add DB/service to lexilize the response
            if 'service_call' in system_frames_next[service]:
                self.use_external_service = True
                self.service_call = system_frames_next[service]['service_call']
                self.service_results = system_frames_next[service]['service_results']

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

        self.system_acts = system_acts
        query_db = 'query false, '
        if self.service_results:
            query_db = 'query true, '
        delex_system_acts = self.delexilize(system_acts, system_frames_next, user_frames)
        self.delex_system_acts = action_start + ' ' + query_db + delex_system_acts + ' ' + action_end
        
        # add delex system acts to token_ids and token_type_ids
        delex_system_acts_tokens_ids = tokenizer.tokens_to_ids(tokenizer.text_to_tokens(self.delex_system_acts))
        self.token_ids += delex_system_acts_tokens_ids
        self.token_type_ids += len(delex_system_acts_tokens_ids) * [tokenizer.tokens_to_ids(action_start)]

        # process system response
        self.response = response_start + ' ' + system_utterance_next + ' ' + response_end
        self.delex_response = self.delexilize(self.response, system_frames_next, user_frames)

        # delex number of DB matches from response
        digits_in_response = [int(d) for d in re.findall(r'\d+', self.delex_response)]
        if len(digits_in_response) > 0:
            logging.info('digits remaining in response: %s', self.delex_response)
            with open('/home/ebakhturina/Desktop/responses_with_digits', 'a') as f:
                f.write(self.delex_response + '\n')
       
        # add delex system response to token_ids and token_type_ids
        # create labels for lm task
        delex_response_tokens_ids = tokenizer.tokens_to_ids(
            tokenizer.text_to_tokens(self.delex_response + tokenizer.eos_token)
        )
        self.ignore_index = -100
        self.labels_lm = [self.ignore_index] * len(self.token_ids)
        self.labels_lm += [self.ignore_index] + delex_response_tokens_ids[1:]
        self.token_ids += delex_response_tokens_ids
        self.token_type_ids += len(delex_response_tokens_ids) * [tokenizer.tokens_to_ids(response_start)]

        assert len(self.token_ids) == len(self.token_type_ids)
        assert len(self.token_ids) == len(self.labels_lm)
        # TODO add DB
        self.input_text = tokenizer.tokens_to_text(tokenizer.ids_to_tokens(self.token_ids))
        logging.debug(self.input_text)
        logging.debug(self.dialogue_history)
        logging.debug(self.dialogue_belief)
        logging.debug(self.delex_system_acts)
        logging.debug(self.delex_response)

    def _remove_star_rating(text):
        replacement = "[star rating]"
        star_rating_regeex = re.compile(r'\d.\d stars')
        mo = star_rating_regeex.search(text)
        if mo is not None:
            logging.debug(f'Original text: {text}')
            text = text.replace(mo.group(), replacement)
            logging.debug(f'Remove rating: {text}')

    def split_intent(self, intent):
        reformatted_intent = ''
        for ch in intent:
            if ch.isupper():
                reformatted_intent += ' ' + ch.lower()
            else:
                reformatted_intent += ch
        return reformatted_intent.strip()

    def delexilize(self, uttr, system_frame, user_frames):
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
            
        found_values = []
        # delex slot values found in actions
        for service in system_frame.values():
            if 'actions' in service:
                for action in service['actions']:
                    for slot_value in action['values']:
                        if slot_value in uttr:
                            slot_name = action['slot']
                            found_values.append((slot_value, slot_name, len(slot_value)))
                            # # spaces added to make sure we're not replacing parts of values
                            # uttr = uttr.replace(' ' + slot_value + ' ', ' [value_' + action['slot'] + '] ')

        # delex slot_values from DB search results
        for service in system_frame.values():
            if 'service_results' in service:
                for service_result in service['service_results']:
                    for slot_name, slot_value in service_result.items():
                        found_values.append((slot_value, slot_name, len(slot_value)))
                        # uttr = uttr.replace(' ' + slot_value + ' ', ' [value_' + slot_name + '] ')
        
        # delex slot_values from user state
        for service in user_frames.values():
            if 'state' in service:
                for slot_name, slot_values in service['state']['slot_values'].items():
                    for slot_value in slot_values:
                        found_values.append((slot_value, slot_name, len(slot_value)))
                        # uttr = uttr.replace(slot_value, '[value_' + slot_name + ']')

        found_values = sorted(found_values, key = lambda x: x[2], reverse=True)
        
        for match in found_values:
            slot_value, slot_name, _ = match
            uttr = uttr.replace(slot_value, '[value_' + slot_name + ']')
        return uttr

    # def delexilize(self, uttr, system_frame, user_frames):
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
    #     def _delex(uttr, system_frame, user_frame):
    #         found_values = []
    #         # delex slot values found in actions
    #         for service in system_frame.values():
    #             if 'actions' in service:
    #                 for action in service['actions']:
    #                     for slot_value in action['values']:
    #                         if slot_value in uttr:
    #                             found_values.append(slot_value, slot_name, len(slot_value))
    #                             # # spaces added to make sure we're not replacing parts of values
    #                             # uttr = uttr.replace(' ' + slot_value + ' ', ' [value_' + action['slot'] + '] ')

    #         # delex slot_values from DB search results
    #         for service in system_frame.values():
    #             if 'service_results' in service:
    #                 for service_result in service['service_results']:
    #                     for slot_name, slot_value in service_result.items():
    #                         found_values.append(slot_value, slot_name, len(slot_value))
    #                         # uttr = uttr.replace(' ' + slot_value + ' ', ' [value_' + slot_name + '] ')
            
    #         # delex slot_values from user state
    #         for service in user_frame.values():
    #             if 'state' in service:
    #                 for slot_name, slot_values in service['state']['slot_values'].items():
    #                     for slot_value in slot_values:
    #                         found_values.append(slot_value, slot_name, len(slot_value))
    #                         # uttr = uttr.replace(' ' + slot_value + ' ', ' [value_' + slot_name + '] ')
            
    #         return uttr
        
    #     # separate all punctuation and words with space
    #     uttr = _delex(uttr, system_frame, user_frames)
    #     uttr = uttr.replace('. ', ' . ').replace(', ', ' , ').replace('? ', ' ? ').replace('! ', ' ! ')
    #     uttr = _delex(uttr, system_frame, user_frames)
    #     return uttr

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
