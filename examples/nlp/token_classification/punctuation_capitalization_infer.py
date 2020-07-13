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

import argparse
import os

import numpy as np

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.nm.data_layers import BertTokenClassificationInferDataLayer, PunctuationCapitalizationDataLayer
from nemo.collections.nlp.utils.data_utils import get_vocab
from nemo.collections.nlp.utils.callback_utils import get_classification_report
from nemo.collections.nlp.nm.trainables import TokenClassifier

# Parsing arguments
parser = argparse.ArgumentParser(description='Punctuation and capitalization detection inference')
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--punct_num_fc_layers", default=3, type=int)
parser.add_argument("--capit_num_fc_layers", default=2, type=int)
parser.add_argument("--part_sent_num_fc_layers", default=1, type=int)
parser.add_argument(
    "--pretrained_model_name",
    default="bert-base-uncased",
    type=str,
    help="Name of the pre-trained model",
)
parser.add_argument(
    "--add_part_sent_head",
    action='store_true',
    help="Whether to a head to BeRT that would be responsible for detecting whether the sentence is partial or not.",
)
parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
parser.add_argument(
    "--tokenizer_model",
    default=None,
    type=str,
    help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
)
parser.add_argument(
    "--tokenizer",
    default="nemobert",
    type=str,
    choices=["nemobert", "sentencepiece"],
    help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
)
parser.add_argument("--vocab_file", default=None, type=str, help="Path to the vocab file.")
parser.add_argument(
    "--do_lower_case",
    action='store_true',
    help="Whether to lower case the input text. True for uncased models, False for cased models. "
    + "Only applicable when tokenizer is build with vocab file",
)
parser.add_argument("--none_label", default='O', type=str)
parser.add_argument(
    "--queries",
    action='append',
    default=[
        'we bought four shirts from the ' + 'nvidia gear store in santa clara',
        'nvidia is a company',
        'can i help you',
        'how are you',
        'how\'s the weather today',
        'okay',
        'we bought four shirts one mug and ten thousand titan rtx graphics cards the more you buy the more you save',
        "what is the weather in",
        "what is the name of", 
        "the next flight is going to be at", 
        "why are they",
        "how many",
        "hell",
        "hello",
        "nice to see you how can i help you",
        "find me the distance",
        "what is the destination place",
        "yeah you sweat too much you get dehydrated real fast",
        "yeah",
        "you find yourself constantly drinking like",
        "right",
        "i mean don't get me wrong it's good to drink liquid cause eighty five supposedly eighty five percent of your body's liquid but you know just a hassle",
        "yep yeah",
        "just a very big hassle in the summer too much heat you know  Like i'm about to step out after our phone conversation and go take a little stroll",
        "yeah i think i'm gonna walk i'm gonna walk you know like thirty blocks or something",
        "wow",
        "but walk at a fast pace",
        "that's a good walk",
        "you know but at a fast pace",
        "i do walk in the city sometimes just for work",
        "oh yeah",
        "you know just to go to clients",
        "yeah they make you work yeah over there you walk a lot",
        "yeah instead of taking the subway sometimes i'll walk",
        "yeah sounds good",
        "you know and you walk like a mile two miles",
        "it's good then you build a stamina like that",
        "really builds some stamina like that you're always fit",
        "right",
        "and nobody call you a weakling or nothing like that",
        "hi find me a restaurant nearby"
    ],
    help="Example: --queries 'san francisco' --queries 'la'",
)
parser.add_argument(
    "--ground_truth_queries",
    action='append',
    default=[
        'We bought four shirts from the Nvidia gear store in Santa Clara.',
        'Nvidia is a company.',
        'Can I help you?',
        'How are you?',
        'How\'s the weather today?',
        'Okay.',
        'We bought four shirts, one mug and ten thousand Titan Rtx graphics cards. The more you buy, the more you save.',
        "What is the weather in",
        "What is the name of", 
        "The next flight is going to be at", 
        "Why are they",
        "How many",
        "Hell",
        "Hello",
        "Nice to see you. How can I help you?",
        "Find me the distance.",
        "What is the destination place?",
        "Yeah, you sweat too much. You get dehydrated real fast.",
        "Yeah.",
        "You find yourself constantly drinking, like",
        "Right.",
        "I mean, don't get me wrong, it's good to drink liquid cause eighty five, supposedly, eighty five percent of your body's liquid. But, you know just a hassle.",
        "Yep, yeah.",
        "Just a very big hassle in the summer. Too much heat, you know. Like, I'm about to step out after our phone conversation and go take a little stroll.",
        "Yeah, I think I'm gonna walk. I'm gonna walk, you know, like thirty blocks or something.",
        "Wow.",
        "But walk at a fast pace.",
        "That's a good walk.",
        "You know, but at a fast pace.",
        "I do walk in the city sometimes just for work.",
        "Oh, yeah.",
        "You know, just to go to clients.",
        "Yeah, they make you work. Yeah, over there you walk a lot.",
        "Yeah, instead of taking the subway, sometimes I'll walk.",
        "Yeah, sounds good.",
        "You know, and you walk like a mile, two miles.",
        "It's good. Then you build a stamina like that.",
        "Really builds some stamina like that. You're always fit.",
        "Right.",
        "And nobody call you a weakling or nothing like that.",
        "Hi. Find me a restaurant nearby."
    ],
    help="Example: --queries 'san francisco' --queries 'la'",
)
parser.add_argument(
    "--add_brackets",
    action='store_false',
    help="Whether to take predicted label in brackets or \
                    just append to word in the output",
)
parser.add_argument("--checkpoint_dir", default='output/checkpoints', type=str)
parser.add_argument(
    "--labels_dict_dir",
    default='data_dir',
    type=str,
    help='Path to directory with punct_label_ids.csv, capit_label_ids.csv and part_sent_label_ids.csv(optional) files. ' +
    'These files are generated during training when the datalayer is created',
)
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--eval_file_prefix', default='dev', type=str)
parser.add_argument('--mode', default='examples', choices=['examples', 'file', 'interactive'])
parser.add_argument('--batch_size', default=8, type=int)

args = parser.parse_args()

if not os.path.exists(args.checkpoint_dir):
    raise ValueError(f'Checkpoints folder not found at {args.checkpoint_dir}')

if args.mode == 'file':
    nf = nemo.core.NeuralModuleFactory(log_dir=os.path.join(args.checkpoint_dir, 'infer' + args.data_dir.replace('/', '_')))
else:
    nf = nemo.core.NeuralModuleFactory(log_dir=None)

punct_labels_dict_path = os.path.join(args.labels_dict_dir, 'punct_label_ids.csv')
capit_labels_dict_path = os.path.join(args.labels_dict_dir, 'capit_label_ids.csv')

if not os.path.exists(punct_labels_dict_path) or not os.path.exists(capit_labels_dict_path):
    raise ValueError ('--labels_dict_dir should contain punct_label_ids.csv and capit_label_ids.csv generated during training')

punct_labels_dict = get_vocab(punct_labels_dict_path)
capit_labels_dict = get_vocab(capit_labels_dict_path)
punct_label_ids = {v:k for k, v in punct_labels_dict.items()}
capit_label_ids = {v:k for k, v in capit_labels_dict.items()}
part_sent_labels_dict = None

if args.add_part_sent_head:
    part_sent_labels_dict_path = os.path.join(args.labels_dict_dir, 'part_sent_label_ids.csv')
    if not os.path.exists(part_sent_labels_dict_path):
        raise ValueError ('--labels_dict_dir should contain part_sent_label_ids.csv generated during training')
    part_sent_labels_dict = get_vocab(part_sent_labels_dict_path)

model = nemo_nlp.nm.trainables.get_huggingface_model(
    bert_config=args.bert_config, pretrained_model_name=args.pretrained_model_name
)
logging.info(f'Number of weights: {model.num_weights}')

tokenizer = nemo.collections.nlp.data.tokenizers.get_tokenizer(
    tokenizer_name=args.tokenizer,
    pretrained_model_name=args.pretrained_model_name,
    tokenizer_model=args.tokenizer_model,
)

hidden_size = model.hidden_size

punct_classifier = TokenClassifier(
            hidden_size=hidden_size,
            num_classes=len(punct_label_ids),
            num_layers=1,
            name='Punctuation',
        )
capit_classifier = TokenClassifier(
            hidden_size=hidden_size, num_classes=len(capit_label_ids), num_layers=1, name='Capitalization'
        )
def concatenate(lists):
        return np.concatenate([t.cpu() for t in lists])

if args.mode == 'examples':
    data_layer = BertTokenClassificationInferDataLayer(
    queries=args.queries, tokenizer=tokenizer, max_seq_length=args.max_seq_length, batch_size=1
    )
    input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask = data_layer()
    hidden_states = model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
    punct_logits = punct_classifier(hidden_states=hidden_states)
    capit_logits = capit_classifier(hidden_states=hidden_states)

    logits = [punct_logits, capit_logits]
    # if args.add_part_sent_head:
    #     logits.append(part_sent_logits)

    evaluated_tensors = nf.infer(tensors=logits + [subtokens_mask], checkpoint_dir=args.checkpoint_dir)

    if args.add_part_sent_head:
        punct_logits, capit_logits, part_sent_logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]
        part_sent_preds = 0
    else:
        punct_logits, capit_logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]

    punct_preds = np.argmax(punct_logits, axis=2)
    capit_preds = np.argmax(capit_logits, axis=2)

    correct = 0
    wrong = 0
    for i, query in enumerate(args.queries):
        punct_pred = punct_preds[i][subtokens_mask[i] > 0.5]
        capit_pred = capit_preds[i][subtokens_mask[i] > 0.5]
        words = query.strip().split()
        if len(punct_pred) != len(words) or len(capit_pred) != len(words):
            raise ValueError('Pred and words must be of the same length')

        output = ''
        for j, w in enumerate(words):
            punct_label = punct_labels_dict[punct_pred[j]]
            capit_label = capit_labels_dict[capit_pred[j]]
            if capit_label != args.none_label:
                w = w.capitalize()
            output += w
            if punct_label != args.none_label:
                output += punct_label
            output += ' '
        if output.strip() == args.ground_truth_queries[i].strip():
            correct += 1
        else:
            wrong += 1
            print(f'Query: {query}')
            print(f'Combined: {output.strip()}')
            print(f'Gr truth: {args.ground_truth_queries[i].strip()}\n')
    logging.info(f'Number of correct predictions: {correct}')
    logging.info(f'Number of wrong predictions: {wrong}')
elif args.mode == 'interactive':
    while True:
        logging.info("Type your text, use STOP to exit and RESTART to start a new dialogue.")
        query = input()
        if query == "STOP":
            logging.info("===================== Exiting ===================")
            break
        data_layer = BertTokenClassificationInferDataLayer(
        queries=[query], tokenizer=tokenizer, max_seq_length=args.max_seq_length, batch_size=1
        )
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask = data_layer()
        hidden_states = model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        punct_logits = punct_classifier(hidden_states=hidden_states)
        capit_logits = capit_classifier(hidden_states=hidden_states)

        logits = [punct_logits, capit_logits]

        evaluated_tensors = nf.infer(tensors=logits + [subtokens_mask], checkpoint_dir=args.checkpoint_dir)

        punct_logits, capit_logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]

        punct_preds = np.argmax(punct_logits, axis=2)
        capit_preds = np.argmax(capit_logits, axis=2)
        i = 0
        punct_pred = punct_preds[i][subtokens_mask[i] > 0.5]
        capit_pred = capit_preds[i][subtokens_mask[i] > 0.5]
        words = query.strip().split()
        if len(punct_pred) != len(words) or len(capit_pred) != len(words):
            raise ValueError('Pred and words must be of the same length')

        output = ''
        for j, w in enumerate(words):
            punct_label = punct_labels_dict[punct_pred[j]]
            capit_label = capit_labels_dict[capit_pred[j]]
            if capit_label != args.none_label:
                w = w.capitalize()
            output += w
            if punct_label != args.none_label:
                output += punct_label
            output += ' '
        
        print(f'Query: {query}')
        print(f'Combined: {output.strip()}')
else:
    text_file = f'{args.data_dir}/text_{args.eval_file_prefix}.txt'
    label_file = f'{args.data_dir}/labels_{args.eval_file_prefix}.txt'

    data_layer = PunctuationCapitalizationDataLayer(
            tokenizer=tokenizer,
            text_file=text_file,
            label_file=label_file,
            pad_label=args.none_label,
            punct_label_ids=punct_label_ids,
            capit_label_ids=capit_label_ids,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size
        )

    data = data_layer()
    hidden_states = model(input_ids=data.input_ids, token_type_ids=data.input_type_ids, attention_mask=data.input_mask)
    punct_logits = punct_classifier(hidden_states=hidden_states)
    capit_logits = capit_classifier(hidden_states=hidden_states)

    logits = [punct_logits, capit_logits]

    evaluated_tensors = nf.infer(tensors=logits + [data.punct_labels, data.capit_labels, data.subtokens_mask], checkpoint_dir=args.checkpoint_dir)

    def _combine(words, punct_pred_or_label, capit_pred_or_label, add_punct=True, add_capit=True):
            output = ''
            for j, w in enumerate(words):
                punct_label = punct_labels_dict[punct_pred_or_label[j]]
                capit_label = capit_labels_dict[capit_pred_or_label[j]]

                if add_capit:
                    if capit_label != args.none_label:
                        w = w.capitalize()
                output += w
                if add_punct:
                    if punct_label != args.none_label:
                        output += punct_label
                output += ' '
            return output

    punct_logits, capit_logits, punct_labels, capit_labels, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]

    punct_preds = np.argmax(punct_logits, axis=2)
    capit_preds = np.argmax(capit_logits, axis=2)
  
    logging.info(f'\n {get_classification_report(punct_labels[subtokens_mask>0.5], punct_preds[subtokens_mask>0.5], label_ids=punct_label_ids)}')
    logging.info(f'\n {get_classification_report(capit_labels[subtokens_mask>0.5], capit_preds[subtokens_mask>0.5], label_ids=capit_label_ids)}')
#     correct = 0
#     wrong = 0
    
#     file_for_errors_path = os.path.join(args.data_dir, args.checkpoint_dir.replace('/', '-') + '.txt')
#     file_for_errors = open(file_for_errors_path, 'w')
#     with open(text_file, 'r') as f:
#         for i, line in enumerate(f):
#             punct_pred = punct_preds[i][subtokens_mask[i] > 0.5]
#             punct_label = punct_labels[i][subtokens_mask[i] > 0.5]
#             capit_pred = capit_preds[i][subtokens_mask[i] > 0.5]
#             capit_label = capit_labels[i][subtokens_mask[i] > 0.5]
#             words = line.strip().split()
#             if len(punct_pred) != len(words) or len(capit_pred) != len(words):
#                 print(f'{i} skipped')
#                 continue
#                 # raise ValueError('Pred and words must be of the same length')
            
#             prediction = _combine(words, punct_pred, capit_pred, add_capit=False).strip()
#             ground_truth = _combine(words, punct_label, capit_label, add_capit=False).strip()

#             if prediction == ground_truth:
#                 correct += 1
#             else:
#                 wrong += 1
#                 file_for_errors.write('Pred: ' + prediction + '\n')
#                 file_for_errors.write('True: ' + ground_truth + '\n\n')

#     logging.info(f'Incorrect examples saved to : {file_for_errors_path}')
# if args.mode != 'interactive':
#     logging.info(f'Number of correct predictions: {correct}')
#     logging.info(f'Number of wrong predictions: {wrong}')
        
