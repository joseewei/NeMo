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


'''
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/train_and_predict.py
'''

import argparse
import math
import os
import re
import time

import numpy as np
import torch
from multiwoz_evaluate import score
from nltk.translate.bleu_score import corpus_bleu
from transformers.tokenization_bert import BasicTokenizer

import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.data.datasets.sgd_dataset.data_processor as data_processor
from nemo.collections.nlp.callbacks.lm_gpt2_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data.datasets.sgd_dataset.schema_processor import SchemaPreprocessor
from nemo.core import (
    Backend,
    CheckpointCallback,
    EvaluatorCallback,
    NeuralModuleFactory,
    SimpleLossLoggerCallback,
    WandbCallback,
)
from nemo.utils import logging
from nemo.utils.lr_policies import get_lr_policy

# Parsing arguments
parser = argparse.ArgumentParser(description='Schema_guided_dst')

# BERT based utterance encoder related arguments
parser.add_argument(
    "--max_seq_length",
    default=1024,
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate for BERT representations.")
parser.add_argument(
    "--pretrained_model_name",
    default="gpt2",
    type=str,
    help="Name of the pre-trained model",
    choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
)
# Hyperparameters and optimization related flags.
parser.add_argument(
    "--checkpoint_dir",
    default=None,
    type=str,
    help="The folder containing the checkpoints for the model to continue training",
)
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation.")
parser.add_argument("--num_epochs", default=80, type=int, help="Total number of training epochs to perform.")

parser.add_argument("--optimizer_kind", default="adam_w", type=str)
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--lr_policy", default="PolynomialDecayAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument(
    "--lr_warmup_proportion",
    default=0.1,
    type=float,
    help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10% of training.",
)
parser.add_argument("--grad_norm_clip", type=float, default=1, help="Gradient clipping")
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--num_gpus", default=1, type=int)

# Input and output paths and other flags.
parser.add_argument(
    "--task_name",
    default="sgd_single_domain",
    type=str,
    choices=data_processor.FILE_RANGES.keys(),
    help="The name of the task to train.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Directory for the downloaded SGD data, which contains the dialogue files"
    " and schema files of all datasets (eg train, dev)",
)
parser.add_argument(
    "--work_dir",
    type=str,
    default="output/SGD",
    help="The output directory where the model checkpoints will be written.",
)
parser.add_argument(
    "--dialogues_example_dir",
    type=str,
    default="dialogues_example_dir",
    help="Directory where preprocessed SGD dialogues are stored.",
)
parser.add_argument(
    "--no_overwrite_dial_files",
    action="store_false",
    help="Whether to generate a new file saving the dialogue examples.",
    dest="overwrite_dial_files",
)
parser.add_argument("--no_shuffle", action="store_true", help="Whether to shuffle training data")
parser.add_argument("--no_time_to_log_dir", action="store_true", help="whether to add time to work_dir or not")
parser.add_argument(
    "--eval_dataset",
    type=str,
    default="dev_test",
    choices=["dev", "test", "dev_test"],
    help="Dataset splits for evaluation.",
)
parser.add_argument(
    "--save_epoch_freq",
    default=1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)
parser.add_argument(
    "--save_step_freq",
    default=-1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)
parser.add_argument("--train_step_freq", default=25, type=int, help="Print training metrics every given iteration.")
parser.add_argument("--eval_step_freq", default=25, type=int, help="Print evaluation metrics every given iteration.")
parser.add_argument(
    "--loss_log_freq", default=-1, type=int, help="Frequency of logging loss values, '-1' - at the end of the epoch",
)

parser.add_argument(
    "--eval_epoch_freq", default=1, type=int, help="Frequency of evaluation",
)
parser.add_argument(
    "--enable_pin_memory", action="store_true", help="Enables the pin_memory feature of Pytroch's DataLoader",
)
parser.add_argument(
    "--debug_mode", action="store_true", help="Enables debug mode with more info on data preprocessing and evaluation",
)

parser.add_argument(
    "--checkpoints_to_keep", default=1, type=int, help="The number of last checkpoints to keep",
)
parser.add_argument(
    "--wandb_project", default=None, type=str, help='Project name for tracking with Weights and Biases'
)
parser.add_argument(
    "--wandb_exp_name", default=None, type=str, help='Experiment name for tracking with Weights and Biases'
)
parser.add_argument("--gpt2_checkpoint", default=None, type=str, help='Path to pretrained GPT-2 checkpoint')
parser.add_argument("--mode", type=str, choices=['train', 'infer'], default='train')
### GPT-2 args
parser.add_argument("--vocab_size", default=-1, type=int, help="Vocabulary size")


args = parser.parse_args()
logging.info(args)

if args.debug_mode:
    logging.setLevel("DEBUG")

if args.task_name == "multiwoz":
    schema_config = {
        "MAX_NUM_CAT_SLOT": 9,
        "MAX_NUM_NONCAT_SLOT": 4,
        "MAX_NUM_VALUE_PER_CAT_SLOT": 47,
        "MAX_NUM_INTENT": 1,
    }
else:
    schema_config = {
        "MAX_NUM_CAT_SLOT": 6,
        "MAX_NUM_NONCAT_SLOT": 12,
        "MAX_NUM_VALUE_PER_CAT_SLOT": 12,
        "MAX_NUM_INTENT": 4,
    }

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

nf = NeuralModuleFactory(
    backend=Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    checkpoint_dir=args.checkpoint_dir,
    files_to_copy=[__file__],
    add_time_to_log_dir=not args.no_time_to_log_dir,
)

ATTR_TO_SPECIAL_TOKEN = {
    'bos_token': '<|bos|>',
    'eos_token': '<|eos|>',
    'pad_token': '<|pad|>',
    'additional_special_tokens': [
        '<|context|>',
        "<|endofcontext|>",
        '<|user|>',
        '<|system|>',
        "<|belief|>",
        "<|endofbelief|>",
        "<|action|>",
        "<|endofaction|>",
        "<|response|>",
        "<|endofresponse|>",
    ],
}

SPECIAL_TOKENS = ['<|bos|>', '<|eos|>', '<|pad|>'] + ATTR_TO_SPECIAL_TOKEN['additional_special_tokens']

MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

MODEL_NAME = 'gpt2'
gpt2_model = nemo_nlp.nm.trainables.huggingface.GPT2LM(pretrained_model_name=MODEL_NAME,)

gpt2_tokenizer = nemo_nlp.data.NemoGPT2Tokenizer(
    pretrained_model=MODEL_NAME
)  # , special_tokens_dict=ATTR_TO_SPECIAL_TOKEN#bos_token=['<|bos|>'], eos_token=['<|eos|>']

# TODO move to HF utils
def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.tokenizer.encoder)
    num_added_tokens = tokenizer.tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN
    )  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
    logging.info('%s special tokens added', num_added_tokens)
    tokenizer.vocab_size += num_added_tokens
    logging.info('%s new vocab_size', tokenizer.vocab_size)
    print(model)


args.vocab_size = gpt2_tokenizer.vocab_size
logging.info(f'Vocab size: {args.vocab_size}')
add_special_tokens_(gpt2_model, gpt2_tokenizer)

if args.gpt2_checkpoint:
    gpt2_model.restore_from(args.gpt2_checkpoint)
    logging.info(f'GPT2 was restored from {args.gpt2_checkpoint}')

args.max_seq_length = min(args.max_seq_length, gpt2_tokenizer.max_len)
schema_config["MAX_SEQ_LENGTH"] = args.max_seq_length
# Run SGD preprocessor to generate and store schema embeddings
schema_preprocessor = SchemaPreprocessor(data_dir=args.data_dir, schema_config=schema_config)


dialogues_processor = data_processor.SGDDataProcessor(
    task_name=args.task_name,
    data_dir=args.data_dir,
    dialogues_example_dir=args.dialogues_example_dir,
    tokenizer=gpt2_tokenizer,
    schema_emb_processor=schema_preprocessor,
    overwrite_dial_files=args.overwrite_dial_files,
    pm_max_seq_length=args.max_seq_length,
    mode='PM',
)


def create_pipeline(dataset_split):
    datalayer = nemo_nlp.nm.data_layers.GPT2DataLayer(
        tokenizer=gpt2_tokenizer,
        dataset_split=dataset_split,
        dialogues_processor=dialogues_processor,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle if dataset_split == 'train' else False,
        pin_memory=args.enable_pin_memory,
    )

    steps_per_epoch = math.ceil(len(datalayer) / (args.batch_size * args.num_gpus))

    data = datalayer()
    loss = gpt2_model(input_ids=data.token_ids, token_type_ids=data.token_type_ids, labels=data.labels_lm)
    return datalayer, loss, steps_per_epoch


_, train_loss, steps_per_epoch = create_pipeline('train')
eval_datalayer, eval_loss, eval_steps_per_epoch = create_pipeline('dev')


logging.info("steps per epoch: %s", steps_per_epoch)

# callback which prints training loss and perplexity once in a while
train_callback = SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=args.train_step_freq,
    print_func=lambda x: logging.info(f'Loss:{str(round(x[0].item(), 3))}'),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer,
)

ckpt_callback = CheckpointCallback(
    folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq, checkpoints_to_keep=1
)

callbacks = [train_callback, ckpt_callback]

if args.wandb_project is not None:
    wand_callback = WandbCallback(
        train_tensors=[train_loss],
        wandb_name=args.wandb_exp_name,
        wandb_project=args.wandb_project,
        update_freq=args.loss_log_freq if args.loss_log_freq > 0 else steps_per_epoch,
        args=args,
    )
    callbacks.append(wand_callback)

eval_callback = EvaluatorCallback(
    eval_tensors=[eval_loss],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=args.eval_step_freq,
    wandb_name=args.wandb_exp_name,
    wandb_project=args.wandb_project,
)
callbacks.append(eval_callback)

lr_policy_fn = get_lr_policy(
    args.lr_policy, total_steps=args.num_epochs * steps_per_epoch, warmup_ratio=args.lr_warmup_proportion
)


if args.mode == 'train':
    nf.train(
        tensors_to_optimize=[train_loss],
        callbacks=callbacks,
        lr_policy=lr_policy_fn,
        optimizer=args.optimizer_kind,
        optimization_params={
            "num_epochs": args.num_epochs,
            "lr": args.learning_rate,
            "eps": 1e-6,
            "weight_decay": args.weight_decay,
            "grad_norm_clip": args.grad_norm_clip,
        },
    )
else:
    logging.info('Doing inference')
    start_time = time.time()
    dev_size = len(eval_datalayer._dataset)

    def _add_space_before_after_special_tok(special_tokens, text):
        for special_tok in special_tokens:
            text = text.replace(special_tok, ' ' + special_tok + ' ')
        return text

    def extract_action_respose(text):
        """
        text: list of str
        Returns:
            actions (list of str)
            response (list of str)
        """
        # to make sure phrases like 'restaurant_name' are treated as a single work for BLEU calculation
        text = ' '.join(text).replace(' _ ', '_').replace('[ ', '[').replace(' ]', ']').split()
        actions = None
        try:
            response_start_idx = text.index('<|response|>')
            # -1 to remove <|endofaction|> token
            actions = text[: response_start_idx - 1]

            response_end_idx = text.index('<|endofresponse|>')
            response = text[response_start_idx + 1 : response_end_idx]
        except:
            # TODO check examples that fail on the regular processing: nlg doesn't generate all special tokens
            if actions is None:
                actions = text
            response = text
        return actions, response

    class ERRScorer:
        """
        ERR = (p + q) / M, where
            M - total number of slots in the dialog act
            p - the number of missing slots,
            q - the number of redundant slots in the given realisation
        """

        def __init__(self):
            self.total = 0
            self.redundant = 0
            self.missing = 0

        def score_response(self, actions, response):
            actions = ' '.join(actions)
            response = ' '.join(response)
            match = r'\[[a-z]*_[a-z]*_?[a-z]+]'
            action_slots = re.findall(match, actions)
            response_slots = re.findall(match, response)

            self.total += len(action_slots)
            self.missing += len(set(action_slots) - set(response_slots))
            self.redundant += len(set(response_slots) - set(action_slots))

        def get_err(self):
            return round((self.missing + self.redundant) / self.total * 100, 2)

    bt = BasicTokenizer(never_split=SPECIAL_TOKENS)
    err_scorer = ERRScorer()
    refs_responses = []
    refs_actions = []
    hyps_responses = []
    hyps_actions = []

    batch_size = 1

    prompts = []
    token_type_ids_pr = []
    prompt_lens = []

    k = 0
    batch_id = 1
    for idx, sample in enumerate(eval_datalayer._dataset):
        token_ids = sample['token_ids']
        labels_lm = sample['labels_lm']

        labels_lm = np.array(labels_lm)
        labels = labels_lm[labels_lm != -100]
        labels_text = gpt2_tokenizer.ids_to_text(labels)
        labels_ids = gpt2_tokenizer.text_to_ids(labels_text)
        labels_tokens = gpt2_tokenizer.text_to_tokens(labels_text)

        # delete everything passed start action token - so that the model generates both action and response
        prompt_ids = np.array(token_ids)[labels_lm == -100]
        logging.debug(f'Prompt: {gpt2_tokenizer.ids_to_text(prompt_ids)}')
        token_type_ids = sample['token_type_ids'][: prompt_ids.shape[0]]
        prompt_len = len(prompt_ids)
        prompt_lens.append(prompt_len)

        ref = gpt2_tokenizer.ids_to_text(token_ids[prompt_len:])
        ref = _add_space_before_after_special_tok(SPECIAL_TOKENS, ref)
        # use basic tokenizer: split by space, split punctuation, don't tokenizer special tokens
        ref = bt.tokenize(ref)

        ref_actions, ref_response = extract_action_respose(ref)
        refs_actions.append([ref_actions])
        refs_responses.append([ref_response])

        prompts.append(prompt_ids)
        token_type_ids_pr.append(token_type_ids)

        is_last = idx + 1 == len(eval_datalayer)
        if idx == batch_size * batch_id - 1 or is_last:
            max_prompt_len = max(prompt_lens)

            decoder_start_token_ids = []
            attention_masks = []

            # # pad from the left
            for i, prompt in enumerate(prompts):
                decoder_start_token_ids.append(prompt[0])
                padding = np.array([gpt2_tokenizer.pad_id] * (max_prompt_len - len(prompt)))
                prompts[i] = np.append(padding, prompt)
                token_type_ids_pr[i] = np.append(padding, token_type_ids_pr[i])
                attention_mask = prompts[i] != gpt2_tokenizer.pad_id
                attention_masks.append(attention_mask)

            prompts = torch.tensor(prompts, dtype=torch.long, device=gpt2_model._device)
            generated_text = gpt2_model.generate(
                input_ids=prompts,
                # token_type_ids=token_type_ids,
                # decoder_start_token_id=decoder_start_token_ids,
                # attention_mask=attention_masks,
                pad_token_id=gpt2_tokenizer.pad_id,
            )

            """
            for bs = 1 for the single domain task
            BLEU: 16.18
            MultiWOZ eval: 16.21
            BLEU: 16.18
            ERR: 50.48
            Evaluation complete in 1284.3471043109894
            
            Process finished with exit code 0
            """

            """
            # doesn't fully supported by HF
            token_type_ids = torch.tensor(token_type_ids_pr, dtype=torch.long, device=gpt2_model._device)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long, device=gpt2_model._device)
            
            generated_text = gpt2_model.generate(
                input_ids=prompts,
                token_type_ids=token_type_ids,
                decoder_start_token_id=decoder_start_token_ids,
                attention_mask=attention_masks,
                pad_token_id=gpt2_tokenizer.pad_id
            )
            BLEU 12.76 with bs=1 with padding left - TODO - padding shouldn't matter for bs=1 but still BLEU is lower
            than without padding
            """

            for i, gen_text in enumerate(generated_text):
                text = gpt2_tokenizer.ids_to_text(gen_text[prompt_lens[i] :])
                text = bt.tokenize(_add_space_before_after_special_tok(SPECIAL_TOKENS, text))
                hyp_actions, hyp_response = extract_action_respose(text)
                hyps_actions.append(hyp_actions)
                hyps_responses.append(hyp_response)
                err_scorer.score_response(refs_actions[i][0], hyp_response)

            prompts = []
            token_type_ids_pr = []
            prompt_lens = []
            refs_actions = []
            batch_id += 1

    assert len(hyps_responses) == len(eval_datalayer)
    bleu = round(corpus_bleu(refs_responses, hyps_responses) * 100, 2)
    print(f'BLEU: {bleu}')

    bleu_multiwoz_eval = round(score(hyps_responses, refs_responses) * 100, 2)
    print(f'MultiWOZ eval:', bleu_multiwoz_eval)
    print(f'BLEU: {bleu}')

    print(f'ERR: {err_scorer.get_err()}')

    refs = [' '.join(r[0]) for r in refs_responses]
    hyps = [' '.join(h) for h in hyps_responses]

    with open(os.path.join(args.work_dir, 'generated_text.txt'), 'w') as f:
        f.write('BLEU:' + str(bleu) + '\n')
        f.write('REF:\n')
        f.write('\n'.join(refs))
        f.write('\n')
        f.write('GENERATED TEXT:\n')
        f.write('\n'.join(hyps))

    with open(os.path.join(args.work_dir, 'ref_hyp_comparison.txt'), 'w') as f:
        f.write('BLEU:' + str(bleu) + '\n')
        assert len(refs) == len(hyps)
        for i in range(len(refs)):
            f.write('REF:' + refs[i] + '\n')
            f.write('GEN:' + hyps[i] + '\n\n')

    print(f'Evaluation complete in {time.time() - start_time}')
