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
"""

To pretrain BERT on raw uncased text dataset run
python bert_pretraining.py \
--amp_opt_level "O0" \
--train_data path_to/wikitext-2/train.txt \
--eval_data path_to/wikitext-2/valid.txt \
--work_dir outputs/bert_lm \
--batch_size 64 \
--lr 0.01 \
--lr_policy CosineAnnealing \
--lr_warmup_proportion 0.05 \
--optimizer novograd \
--beta1 0.95 \
--beta2 0.25 \
--tokenizer sentence-piece \
--vocab_size 3200 \
--hidden_size 768 \
--intermediate_size 3072 \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--hidden_act "gelu" \
--save_step_freq 200 \
data_text \
--dataset_name wikitext-2 \
--num_epochs 10 \
--sample_size 10000000 \
--mask_probability 0.15 \
--short_seq_prob 0.1 \

To pretrain BERT large on preprocessed dataset,
download and preprocess dataset from here:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/
Run the script:
./data/create_datasets_from_start.sh
and extract data into train_data and eval_data

Then run BERT large on dataset with a sequence length of 512 and a maximum of 80 masked tokens per sequence
python -m torch.distributed.launch --nproc_per_node=8 bert_pretraining.py \
--batch_size 8 \
--config_file bert_config.json
--train_data train_data \
--eval_data eval_data \
--save_step_freq 200 \
--num_gpus 8 \
--amp_opt_level "O1" \
--lr_policy SquareRootAnnealing \
--beta1 0.9 \
--beta2 0.999 \
--lr_warmup_proportion 0.01 \
--optimizer adam_w \
--weight_decay 0.01 \
--lr 0.4375e-4 \
data_preprocessed \
--max_predictions_per_seq 80 \
--num_iters 2285714  

BERT base uncased trained with 2285714 iterations on a DGX1 with 8 V100 GPUs with AMP O1 optimization
should finish in 200 hours and yield EM/F1 of 82.74/89.79 on SQuADv1.1 and 71.24/74.32 on SQuADv2.0.
On GLUE benchmark MRPC task the model achieves accuracy/F1 od 86.52/90.53.

BERT large uncased trained with 2285714 iterations on a DGX1 with 8 V100 GPUs with AMP O1 optimization
should finish in 410 hours and yield EM/F1 of 85.79/92.28 on SQuADv1.1 and 80.17/83.32 on SQuADv2.0.
On GLUE benchmark MRPC task the model achieves accuracy/F1 od 88.7/91.96.

More information about BERT pretraining can be found at 
https://nvidia.github.io/NeMo/nlp/bert_pretraining.html

Pretrained BERT models and model configuration files can be found at 
https://ngc.nvidia.com/catalog/models/nvidia:bertlargeuncasedfornemo
https://ngc.nvidia.com/catalog/models/nvidia:bertbaseuncasedfornemo
https://ngc.nvidia.com/catalog/models/nvidia:bertbasecasedfornemo

"""
import argparse
import math
import os
import sys

from transformers import BertConfig

import nemo.backends.pytorch.common as nemo_common
import nemo.backends.pytorch.common.losses
import nemo.collections.nlp as nemo_nlp
import nemo.core as nemo_core
from nemo.utils import logging
from nemo.collections.nlp.data.datasets.lm_bert_dataset import BERTPretrainingDataDesc
from nemo.utils.lr_policies import get_lr_policy

parser = argparse.ArgumentParser(description='BERT pretraining')
parser.add_argument(
    "--local_rank", default=None, type=int, help="Automatically set when using Multi-GPU with torch.distributed."
)
parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs to use.")
parser.add_argument("--train_data", required=True, type=str, help="path to training dataset.")
parser.add_argument("--config_file", default=None, type=str, help="The BERT model config")
parser.add_argument("--eval_data", required=True, type=str, help="path to evaluation dataset.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size per worker for each model pass.")
parser.add_argument(
    "--batches_per_step",
    default=1,
    type=int,
    help="Number of gradient accumulation steps per iteration before parameters are updated.",
)
parser.add_argument("--lr", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument(
    "--lr_policy",
    default=None,
    type=str,
    choices=[
        "WarmupHoldPolicy",
        "SquareAnnealing",
        "SquareRootAnnealing",
        "CosineAnnealing",
        "WarmupAnnealing",
        "InverseSquareRootAnnealing",
        "PolynomialDecayAnnealing",
        "PolynomialHoldDecayAnnealing",
    ],
    help="Learning rate policy.",
)
parser.add_argument(
    "--lr_warmup_proportion", default=0.05, type=float, help="Warm up proportion of total training iterations."
)
parser.add_argument(
    "--optimizer",
    default="novograd",
    type=str,
    choices=["novograd", "adam", "sgd", "adam_w", "fused_novograd", "fused_adam", "fused_lamb"],
    help="Optimizer algorithm for training.",
)
parser.add_argument(
    "--beta1",
    default=0.95,
    type=float,
    help="Only needed for specific optimizers. Exponential decay rates for the 1st moment of optimizers, e.g. *adam*, *novograd*, *lamb*.",
)
parser.add_argument(
    "--beta2",
    default=0.25,
    type=float,
    help="Only needed for specific optimizers. Exponential decay rates for the 2nd moment of optimizers, e.g. *adam*, *novograd*, *lamb*.",
)
parser.add_argument(
    "--amp_opt_level",
    default="O0",
    type=str,
    choices=["O0", "O1", "O2"],
    help="Automatic Mixed Precision optimization level. For further information visit https://nvidia.github.io/apex/amp.html.",
)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay parameter of the optimizer.")
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--vocab_size", default=3200, type=int)
parser.add_argument("--hidden_size", default=768, type=int)
parser.add_argument("--intermediate_size", default=3072, type=int)
parser.add_argument("--num_attention_heads", default=12, type=int)
parser.add_argument("--num_hidden_layers", default=12, type=int)
parser.add_argument("--hidden_act", default="gelu", type=str)
parser.add_argument("--gradient_predivide", action="store_true", default=False, help="use gradient predivide")
parser.add_argument("--only_mlm_loss", action="store_true", default=False, help="use only masked language model loss")
parser.add_argument(
    "--load_dir",
    default=None,
    type=str,
    help="Directory with weights and optimizer checkpoints. Used for resuming training.",
)
parser.add_argument(
    "--bert_checkpoint",
    default=None,
    type=str,
    help="Path to BERT encoder weights file. Used for encoder initialization for finetuning.",
)
parser.add_argument(
    "--work_dir", default="outputs/bert_lm", type=str, help="Output directory for checkpoints, logs etc."
)
parser.add_argument("--grad_norm_clip", type=float, default=-1, help="gradient clipping")
parser.add_argument("--save_epoch_freq", default=1, type=int, help="Save checkpoints every given epoch.")
parser.add_argument("--save_step_freq", default=100, type=int, help="Save checkpoints every given iteration.")
parser.add_argument("--train_step_freq", default=25, type=int, help="Print training metrics every given iteration.")
parser.add_argument("--eval_step_freq", default=25, type=int, help="Print evaluation metrics every given iteration.")
parser.add_argument("--no_shuffle", action="store_true", help="Whether to shuffle training data")
parser.add_argument("--num_epochs", default=10, type=int, help="Number of training epochs.")
parser.add_argument("--num_iters", default=-1, type=int, help="Number of training steps.")
parser.add_argument("--sample_size", default=1e7, type=int, help="Data sample size.")
parser.add_argument(
    "--mask_probability",
    default=0.15,
    type=float,
    help="Probability of masking a token in the input text during data processing.",
)
parser.add_argument(
    "--short_seq_prob",
    default=0.1,
    type=float,
    help="Probability of having a sequence shorter than the maximum sequence length `max_seq_length` in data processing.",
)
parser.add_argument(
    "--dataset_name", default="wikitext-2", choices=["wikitext-2"], type=str, help="Dataset name."
)
parser.add_argument(
    "--tokenizer",
    default="nemogpt2",
    type=str,
    choices=["nemogpt2"],
    help="Text tokenizer type.",
)
parser.add_argument(
    "--pretrained_model_name",
    default="gpt2",
    type=str,
    help="Name of the pre-trained model",
)

args = parser.parse_args()
logging.info(f'{args}')

nf = nemo_core.NeuralModuleFactory(
    backend=nemo_core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
    add_time_to_log_dir=False,
)

    # TODO gpt2 special tokens?
    # special_tokens = nemo_nlp.data.get_bert_special_tokens('bert')

ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens':
                         ['<context>', "<|endofcontext|>", 
                         '<|user|>', '<|system|>', 
                         "<|belief|>", "<|endofbelief|>",
                         "<|action|>""<|endofaction|>", 
                         "<|response|>", "<|endofresponse|>"]}

SPECIAL_TOKENS = ['<|bos|>', '<|eos|>', '<|pad|>'] + ATTR_TO_SPECIAL_TOKEN['additional_special_tokens']

MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

gpt2_model = nemo_nlp.nm.trainables.huggingface.GPT2LM(
    pretrained_model_name=args.pretrained_model_name,
)
tokenizer = nemo_nlp.data.NemoGPT2Tokenizer(pretrained_model=args.pretrained_model_name,
bos_token=['<|bos|>'],
eos_token=['<|eos|>'])

# TODO move to HF utils
def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
    logging.info('%s special tokens added', num_added_tokens)
    tokenizer.vocab_size += num_added_tokens

args.vocab_size = tokenizer.vocab_size
import pdb; pdb.set_trace()
add_special_tokens_(gpt2_model, tokenizer)



# classifier = nemo_nlp.nm.trainables.BertTokenClassifier(
#     model.hidden_size, num_classes=args.vocab_size, activation=args.hidden_act, log_softmax=True
# )

# if args.bert_checkpoint is not None:
#     bert_model.restore_from(args.bert_checkpoint)

# """ create necessary modules for the whole translation pipeline, namely
# data layers, BERT encoder, and MLM and NSP loss functions
# """


args.max_seq_length = min(args.max_seq_length, tokenizer.max_len)


def create_pipeline(data_file, train=True):
    data_layer = nemo_nlp.nm.data_layers.GPT2DataLayer(
            dataset_type=nemo_nlp.data.LineByLineTextDataset,
            tokenizer=tokenizer,
            file_path=data_file,
            block_size=args.max_seq_length,
            batch_size=args.batch_size, 
            shuffle=not args.no_shuffle if train else False,
        )

    steps_per_epoch = math.ceil(len(data_layer) / (args.batch_size * args.num_gpus * args.batches_per_step))

    input_ids = data_layer()
    loss = gpt2_model(input_ids=input_ids)
    return loss, steps_per_epoch
    

train_loss, steps_per_epoch = create_pipeline(
    data_file=args.train_data
)

eval_loss, eval_steps_per_epoch = create_pipeline(
    data_file=args.eval_data, train=False
)


logging.info("steps per epoch: %s", steps_per_epoch)
# callback which prints training loss and perplexity once in a while
train_callback = nemo_core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=args.train_step_freq,
    print_func=lambda x: logging.info(f'Loss:{str(round(x[0].item(), 3))}'),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer,
)

ckpt_callback = nemo_core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    epoch_freq=args.save_epoch_freq,
    load_from_folder=args.load_dir,
    step_freq=args.save_step_freq,
)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[eval_loss],
    user_iter_callback=nemo_nlp.callbacks.lm_bert_callback.eval_iter_callback,
    user_epochs_done_callback=nemo_nlp.callbacks.lm_bert_callback.eval_epochs_done_callback,
    eval_step=args.eval_step_freq,
)

# define learning rate decay policy
if args.lr_policy is not None:
    if args.num_iters < 0:
        lr_policy_fn = get_lr_policy(
            args.lr_policy, total_steps=args.num_epochs * steps_per_epoch, warmup_ratio=args.lr_warmup_proportion
        )
    else:
        lr_policy_fn = get_lr_policy(
            args.lr_policy, total_steps=args.num_iters, warmup_ratio=args.lr_warmup_proportion
        )
else:
    lr_policy_fn = None


# define and launch training algorithm (optimizer)
optimization_params = {
    "lr": args.lr,
    "betas": (args.beta1, args.beta2),
    "weight_decay": args.weight_decay,
}

if args.num_iters < 0:
    optimization_params['num_epochs'] = args.num_epochs
else:
    optimization_params['max_steps'] = args.num_iters

if args.grad_norm_clip >= 0:
    optimization_params['grad_norm_clip'] = args.grad_norm_clip

nf.train(
    tensors_to_optimize=[train_loss],
    lr_policy=lr_policy_fn,
    callbacks=[train_callback, ckpt_callback, eval_callback],
    optimizer=args.optimizer,
    batches_per_step=args.batches_per_step,
    gradient_predivide=args.gradient_predivide,
    optimization_params=optimization_params,
)
