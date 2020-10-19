import re
import csv
from transformers import AdamW, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, GPT2Model, GPT2Config, GPT2Tokenizer
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.utils.checkpoint as checkpoint
import torch
import numpy as np
import json
from time import time
from random import choice, randint
import math
import functools
print("Importing dependencies: ")

import os
import faiss
import nlp
import ast

import pprint


class MSMarcoNLGenDatasetS2S(Dataset):
    def __init__(
        self, examples_array, make_doc_fun=None, extra_answer_threshold=3, document_cache=None, training=True
    ):
        self.training = training
        self.data = examples_array
        self.make_doc_function = make_doc_fun
        self.document_cache = {} if document_cache is None else document_cache
        # assert not (make_doc_fun is None and document_cache is None)
        # make index of specific question-answer pairs from multi-answers
        # if self.training:
        #     self.qa_id_list = [
        #         (i, j)
        #         for i, qa in enumerate(self.data)
        #         for j, (a, sc) in enumerate(zip(qa["answers"]["text"], qa["answers"]["score"]))
        #         if j == 0 or sc >= extra_answer_threshold
        #     ]
        # else:
        # self.qa_id_list = [(i, 0) for i in range(self.data.num_rows)]

    def __len__(self):
        # return len(self.qa_id_list)
        return len(self.data)

    def make_example(self, idx):
        example = self.data[idx]
        question = example["query"]

        x = ast.literal_eval(example["wellFormedAnswers"])
        x = [n.strip() for n in x]

        wellFormedAnswer = x[0]

        x = ast.literal_eval(example["answers"])
        x = [n.strip() for n in x]
        ans_span = x[0]

        context = "\<P>"+ans_span

        in_st = "question: {} context: {}".format(
            question.lower().replace(" --t--", "").strip(), context.lower().strip(),
        )

        out_st = wellFormedAnswer

        return (in_st, out_st)

    def __getitem__(self, idx):
        return self.make_example(idx)


def make_qa_s2s_model(model_name, from_file=None, device="cuda:0"):
    if(model_name != "gpt2"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        configuration = GPT2Config()
        model = GPT2Model(configuration)
    if from_file is not None:
        # has model weights, optimizer, and scheduler states
        param_dict = torch.load(from_file)
        model.load_state_dict(param_dict["model"])
    return tokenizer, model


def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360, device="cuda:0"):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(
        q_ls, max_length=max_len, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(
        max_len, max_a_len), pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks["input_ids"]).to(device),
        torch.LongTensor(a_toks["attention_mask"]).to(device),
    )
    lm_labels = a_ids[:, 1:].contiguous().clone()
    lm_labels[a_mask[:, 1:].contiguous() == 0] = -100
    model_inputs = {
        "input_ids": q_ids,
        "attention_mask": q_mask,
        "decoder_input_ids": a_ids[:, :-1].contiguous(),
        "lm_labels": lm_labels,
    }
    return model_inputs


def train_qa_s2s_epoch(model, dataset, tokenizer, optimizer, scheduler, args, e=0, curriculum=False):
    model.train()
    # make iterator
    if curriculum:
        train_sampler = SequentialSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch_inputs in enumerate(epoch_iterator):
        pre_loss = model(**batch_inputs)[0]
        # print("Pre loss: ", ([int(x) for x in pre_loss.shape]))
        loss = pre_loss.sum() / pre_loss.shape[0]
        loss.backward()
        # optimizer
        if step % args.backward_freq == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        # some printing within the epoch
        loc_loss += loss.item()
        loc_steps += 1
        if step % args.print_freq == 0 or step == 1:
            print(
                "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                    e, step, len(dataset) // args.batch_size, loc_loss /
                    loc_steps, time() - st_time,
                )
            )
            loc_loss = 0
            loc_steps = 0


def eval_qa_s2s_epoch(model, dataset, tokenizer, args):
    model.eval()
    # make iterator
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # print()
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    with torch.no_grad():
        for step, batch_inputs in enumerate(epoch_iterator):
            #print("I am here")
            pre_loss = model(**batch_inputs)[0]
            # print("Pre losdd: ", ([int(x) for x in pre_loss.shape]))
            loss = pre_loss.sum() / pre_loss.shape[0]
            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                print(
                    "{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                        step, len(dataset) // args.batch_size, loc_loss /
                        loc_steps, time() - st_time,
                    )
                )
    print("Total \t L: {:.3f} \t -- {:.3f}".format(loc_loss /
                                                   loc_steps, time() - st_time,))


def train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args):
    s2s_optimizer = AdamW(qa_s2s_model.parameters(),
                          lr=s2s_args.learning_rate, eps=1e-8)
    s2s_scheduler = get_linear_schedule_with_warmup(
        s2s_optimizer,
        num_warmup_steps=400,
        num_training_steps=(s2s_args.num_epochs + 1) *
        math.ceil(len(s2s_train_dset) / s2s_args.batch_size),
    )
    for e in range(s2s_args.num_epochs):
        train_qa_s2s_epoch(
            qa_s2s_model,
            s2s_train_dset,
            qa_s2s_tokenizer,
            s2s_optimizer,
            s2s_scheduler,
            s2s_args,
            e,
            curriculum=(e == 0),
        )
        m_save_dict = {
            "model": qa_s2s_model.state_dict(),
            "optimizer": s2s_optimizer.state_dict(),
            "scheduler": s2s_scheduler.state_dict(),
        }
        # print("Saving model {}".format(s2s_args.model_save_name))
        eval_qa_s2s_epoch(qa_s2s_model, s2s_valid_dset,
                          qa_s2s_tokenizer, s2s_args)
        torch.save(m_save_dict, "{}_{}.pth".format(
            s2s_args.model_save_name, e))


# generate answer from input "question: ... context: <p> ..."
def qa_s2s_generate(
    question_doc,
    qa_s2s_model,
    qa_s2s_tokenizer,
    num_answers=1,
    num_beams=None,
    min_len=64,
    max_len=256,
    do_sample=False,
    temp=1.0,
    top_p=None,
    top_k=None,
    max_input_length=512,
    device="cuda:0",
):
    model_inputs = make_qa_s2s_batch(
        [(question_doc, "A")], qa_s2s_tokenizer, max_input_length, device=device,)
    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    generated_ids = qa_s2s_model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        min_length=min_len,
        max_length=max_len,
        do_sample=do_sample,
        early_stopping=True,
        num_beams=1 if do_sample else n_beams,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=qa_s2s_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=num_answers,
        decoder_start_token_id=qa_s2s_tokenizer.bos_token_id,
    )
    return [qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]


compileDataset = False

if compileDataset:
    print("Working on Dataset preparation: ")

    train_data = nlp.load_dataset("ms_marco", name="v2.1")["train"]
    eval_data = nlp.load_dataset("ms_marco", name="v2.1")["validation"]

    train_data = train_data.filter(
        lambda example: len(example["wellFormedAnswers"]) > 0)
    eval_data = eval_data.filter(
        lambda example: len(example["wellFormedAnswers"]) > 0)

    # print(f"The length of eval data is "{len(eval_data)})
    # print(f"The length of train data is "{len(train_data)})
    headers = train_data.column_names

    print("Compiling Train dataset in csv...")
    with open("./train_dataset.csv", "a+") as file:
        csv_writer = csv.DictWriter(file, fieldnames=headers)
        csv_writer.writeheader()
        for i in range(len(train_data)):
            # print(train_data[i])
            csv_writer.writerow(train_data[i])

    print("Completed Train Dataset")
    print("Compiling Test dataset in csv...")
    print(eval_data.column_names)
    print(len(eval_data))
    headers = eval_data.column_names
    with open("./test_dataset.csv", "a+") as file:
        csv_writer = csv.DictWriter(file, fieldnames=headers)
        csv_writer.writeheader()
        for i in range(len(eval_data)):
            # print(test_data[i])
            csv_writer.writerow(eval_data[i])

    print("Completed Test Dataset")

file_train_data = nlp.load_dataset("csv", data_files="train_dataset.csv")
file_test_data = nlp.load_dataset("csv", data_files="test_dataset.csv")

# pp = pprint.PrettyPrinter(indent=4)
#
# pp.pprint(type(file_train_data["train"]))
# pp.pprint(file_train_data["train"][0])
# # pp.pprint(file_train_data["train"][1])
# # pp.pprint(file_train_data["train"][2])
# # pp.pprint(file_train_data["train"][3])
# pp.pprint(file_train_data["train"][0])
#
#
# print(type(file_train_data["train"][0]["wellFormedAnswers"]))
#
# x = ast.literal_eval(file_train_data["train"][0]["wellFormedAnswers"])
# x = [n.strip() for n in x]
# print(x)
# print(x[0])
#
# print(type(file_train_data["train"][0]["query"]))
# print(file_train_data["train"][0]["query"])
#
#
# print("answer spans:::::::::::")
#
# print(type(file_train_data["train"][0]["answers"]))
# print(file_train_data["train"][0]["answers"][0])
#
#
# x = ast.literal_eval(file_train_data["train"][0]["answers"])
# x = [n.strip() for n in x]
# print(x)
# print(x[0])






###############
# MSMarcoNLGen seq2seq model training
###############


print("Starting training...")
bs = 16 # int(input("Enter batch size: "))
maxLength = 1024 # int(input("Enter max length value: "))
# training loop proper


class ArgumentsS2S():
    def __init__(self, bs):
        self.batch_size = bs
        self.backward_freq = 16
        self.max_length = maxLength
        self.print_freq = 100
        self.model_save_name = "msmarco_bart_wellformedans"
        self.learning_rate = 2e-4
        self.num_epochs = 3
        self.truncation = True


s2s_args = ArgumentsS2S(bs)

s2s_train_dset = MSMarcoNLGenDatasetS2S(file_train_data["train"])
s2s_valid_dset = MSMarcoNLGenDatasetS2S(file_test_data["train"], training=False)

qa_s2s_tokenizer, pre_model = make_qa_s2s_model(
    model_name="facebook/bart-large",
    from_file=None,
    device="cuda:0"
)
qa_s2s_model = torch.nn.DataParallel(pre_model)
print("Starting Training...")
train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer,
             s2s_train_dset, s2s_valid_dset, s2s_args)