# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import faiss
import numpy as np
import streamlit as st
import torch
import transformers
from lfqa_utils import *
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer


@st.cache(allow_output_mutation=True)
def load_models():
    qar_tokenizer = AutoTokenizer.from_pretrained("yjernite/retribert-base-uncased")
    qar_model = AutoModel.from_pretrained("yjernite/retribert-base-uncased").to("cuda:0")
    _ = qar_model.eval()

    s2s_tokenizer = AutoTokenizer.from_pretrained("yjernite/bart_eli5")
    s2s_model = AutoModelForSeq2SeqLM.from_pretrained("yjernite/bart_eli5").to("cuda:0")
    # save_dict = torch.load("seq2seq_models/eli5_bart_model_blm_2.pth")
    # s2s_model.load_state_dict(save_dict["model"])
    _ = s2s_model.eval()

    return (qar_tokenizer, qar_model, s2s_tokenizer, s2s_model)


@st.cache(allow_output_mutation=True)
def load_indexes(qar_model, qar_tokenizer):
    marvel_snippets = load_data("/home/vgetselevich/data/marvel/wiki/", 100)

    # prepare IR index
    if not os.path.isfile('marvel_passages_reps_32_l-8_h-768_b-512-512.dat'):
        print("*** Generating dense index ***")
        make_qa_dense_index_text_chunks(
            qar_model,
            qar_tokenizer,
            # wiki40b_snippets,
            marvel_snippets,
            device='cuda:0',
            index_name='marvel_passages_reps_32_l-8_h-768_b-512-512.dat',
        )

    faiss_res = faiss.StandardGpuResources()
    #    wiki40b_passages = nlp.load_dataset(path="wiki_snippets", name="wiki40b_en_100_0")["train"]
    marvel_passage_reps = np.memmap(
        "marvel_passages_reps_32_l-8_h-768_b-512-512.dat",
        dtype="float32",
        mode="r",
        shape=(len(marvel_snippets), 128),
    )
    wiki40b_index_flat = faiss.IndexFlatIP(128)
    wiki40b_gpu_index_flat = faiss.index_cpu_to_gpu(faiss_res, 1, wiki40b_index_flat)
    wiki40b_gpu_index_flat.add(marvel_passage_reps)  # TODO fix for larger GPU

    return (marvel_snippets, wiki40b_gpu_index_flat)


qar_tokenizer, qar_model, s2s_tokenizer, s2s_model = load_models()
passages, gpu_dense_index = load_indexes(qar_model, qar_tokenizer)


def make_support(question, source="wiki40b", method="dense", n_results=10):
    if source == "none":
        support_doc, hit_lst = (" <P> ".join(["" for _ in range(11)]).strip(), [])
    else:
        support_doc, hit_lst = query_qa_dense_index(
            question, qar_model, qar_tokenizer, passages, gpu_dense_index, n_results
        )

    support_list = [
        #       (res["article_title"], res["section_title"].strip(), res["score"], res["passage_text"]) for res in hit_lst
        ('Title', 'Section title', res["score"], res["text"])
        for res in hit_lst
    ]
    question_doc = "question: {} context: {}".format(question, support_doc)
    return question_doc, support_list


@st.cache(hash_funcs={torch.Tensor: (lambda _: None), transformers.tokenization_bart.BartTokenizer: (lambda _: None)})
def answer_question(
    question_doc, s2s_model, s2s_tokenizer, min_len=64, max_len=256, sampling=False, n_beams=2, top_p=0.95, temp=0.8
):
    with torch.no_grad():
        answer = qa_s2s_generate(
            question_doc,
            s2s_model,
            s2s_tokenizer,
            num_answers=1,
            num_beams=n_beams,
            min_len=min_len,
            max_len=max_len,
            do_sample=sampling,
            temp=temp,
            top_p=top_p,
            top_k=None,
            max_input_length=1024,
            device="cuda:0",
        )[0]
    return (answer, support_list)


st.title("Marvel Universe Question Answering")

# Start sidebar
# header_html = "<img src='https://huggingface.co/front/assets/huggingface_logo.svg'>"
# header_html = "<img src='iron_man.png'>"
header_html = "<img width='150' height='200' src='https://vignette.wikia.nocookie.net/marvelcinematicuniverse/images/f/f2/Iron_Man_Armor_-_Mark_LXXXV.png/revision/latest/top-crop/width/360/height/450?cb=20190401222437'>"
header_full = """
<html>
  <head>
    <style>
      .img-container {
        padding-left: 70px;
        padding-right: 70px;
        padding-top: 50px;
        padding-bottom: 50px;
        background-color: #f0f3f9;
      }
    </style>
  </head>
  <body>
    <span class="img-container"> <!-- Inline parent element -->
      %s
    </span>
  </body>
</html>
""" % (
    header_html,
)
st.sidebar.markdown(
    header_full, unsafe_allow_html=True,
)

# Generative QA on Marvel Universe
description = """
---
This demo presents generative and extractive answers to Marvel Universe content questions.
First dense IR model fetches a set of relevant snippets from Marvel wiki docs given the question,
and then Bart  model use them as a context to generate and answer and
extractive QA model creates an alternative answer.
"""
st.sidebar.markdown(description, unsafe_allow_html=True)

action_list = [
    "Answer the question",
    "View the retrieved document only",
    "View the most similar ELI5 question and answer",
    "Show me everything, please!",
]
demo_options = st.sidebar.checkbox("Demo options")
if demo_options:
    action_st = st.sidebar.selectbox("", action_list, index=3,)
    action = action_list.index(action_st)
    show_type = st.sidebar.selectbox("", ["Show full text of passages", "Show passage section titles"], index=0,)
    show_passages = show_type == "Show full text of passages"
else:
    action = 3
    show_passages = True

retrieval_options = st.sidebar.checkbox("Retrieval options")
if retrieval_options:
    retriever_info = """
    ### Information retriever options

    The **sparse** retriever uses ElasticSearch, while the **dense** retriever uses max-inner-product search between a question and passage embedding
    trained using the [ELI5](https://arxiv.org/abs/1907.09190) questions-answer pairs.
    The answer is then generated by sequence to sequence model which takes the question and retrieved document as input.
    """
    st.sidebar.markdown(retriever_info)
    wiki_source = st.sidebar.selectbox("Which Wikipedia format should the model use?", ["wiki40b", "none"])
    index_type = st.sidebar.selectbox("Which Wikipedia indexer should the model use?", ["dense", "sparse", "mixed"])
else:
    wiki_source = "wiki40b"
    index_type = "dense"

sampled = "beam"
n_beams = 8
min_len = 64
max_len = 256
top_p = None
temp = None
generate_options = st.sidebar.checkbox("Generation options")
if generate_options:
    generate_info = """
    ### Answer generation options

    The sequence-to-sequence model was initialized with [BART](https://huggingface.co/facebook/bart-large)
    weights and fine-tuned on the ELI5 QA pairs and retrieved documents. You can use the model for greedy decoding with
    **beam** search, or **sample** from the decoder's output probabilities.
    """
    st.sidebar.markdown(generate_info)
    sampled = st.sidebar.selectbox("Would you like to use beam search or sample an answer?", ["beam", "sampled"])
    min_len = st.sidebar.slider(
        "Minimum generation length", min_value=8, max_value=256, value=64, step=8, format=None, key=None
    )
    max_len = st.sidebar.slider(
        "Maximum generation length", min_value=64, max_value=512, value=256, step=16, format=None, key=None
    )
    if sampled == "beam":
        n_beams = st.sidebar.slider("Beam size", min_value=1, max_value=8, value=2, step=None, format=None, key=None)
    else:
        top_p = st.sidebar.slider(
            "Nucleus sampling p", min_value=0.1, max_value=1.0, value=0.95, step=0.01, format=None, key=None
        )
        temp = st.sidebar.slider(
            "Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.01, format=None, key=None
        )
        n_beams = None

# start main text
questions_list = [
    "<MY QUESTION>",
    "Who is iron man?",
    "who is Tony Stark?",
    "who directed iron man?",
]
question_s = st.selectbox(
    "What would you like to ask? ---- select <MY QUESTION> to enter a new query", questions_list, index=1,
)
if question_s == "<MY QUESTION>":
    question = st.text_input("Enter your question here:", "")
else:
    question = question_s

if st.button("Show me!"):
    if action in [0, 1, 3]:
        if index_type == "mixed":
            _, support_list_dense = make_support(question, source=wiki_source, method="dense", n_results=10)
            _, support_list_sparse = make_support(question, source=wiki_source, method="sparse", n_results=10)
            support_list = []
            for res_d, res_s in zip(support_list_dense, support_list_sparse):
                if tuple(res_d) not in support_list:
                    support_list += [tuple(res_d)]
                if tuple(res_s) not in support_list:
                    support_list += [tuple(res_s)]
            support_list = support_list[:10]
            question_doc = "<P> " + " <P> ".join([res[-1] for res in support_list])
        else:
            question_doc, support_list = make_support(question, source=wiki_source, method=index_type, n_results=10)
    if action in [0, 3]:
        # Geenerative QA
        answer, support_list = answer_question(
            question_doc,
            s2s_model,
            s2s_tokenizer,
            min_len=min_len,
            max_len=int(max_len),
            sampling=(sampled == "sampled"),
            n_beams=n_beams,
            top_p=top_p,
            temp=temp,
        )

        # Extractive QA
        ex_answer_span, ex_answer_sent = extractive_qa(question, question_doc.replace("<P> ", ""))

        st.markdown("### The model generated answer is:")
        st.write(answer)

        st.markdown("### The model extractive answer is:")
        st.write(ex_answer_span + " - " + ex_answer_sent)

    if action in [0, 1, 3] and wiki_source != "none":
        st.markdown("--- \n ### The model is drawing information from the following Marvel snippets:")
        for i, res in enumerate(support_list):
            wiki_url = "https://en.wikipedia.org/wiki/{}".format(res[0].replace(" ", "_"))
            sec_titles = res[1].strip()
            if sec_titles == "":
                sections = "[{}]({})".format(res[0], wiki_url)
            else:
                sec_list = sec_titles.split(" & ")
                sections = " & ".join(
                    ["[{}]({}#{})".format(sec.strip(), wiki_url, sec.strip().replace(" ", "_")) for sec in sec_list]
                )
            st.markdown(
                "{0:02d} - **Article**: {1:<18} <br>  _Section_: {2}".format(i + 1, res[0], sections),
                unsafe_allow_html=True,
            )
            if show_passages:
                st.write(
                    '> <span style="font-family:arial; font-size:10pt;">' + res[-1] + "</span>", unsafe_allow_html=True
                )
