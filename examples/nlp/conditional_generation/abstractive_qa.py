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

import nlp
from lfqa_utils import *


def full_test():
    # eli5 = nlp.load_dataset('eli5')
    wiki40b_snippets = nlp.load_dataset('wiki_snippets', name='wiki40b_en_100_0')['train']

    # print(eli5['test_eli5'][12345])
    # print(wiki40b_snippets[8991855])

    # load models
    # dense IR model
    qar_tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased')
    qar_model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to('cuda:0')
    _ = qar_model.eval()

    # generative model
    qa_s2s_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
    qa_s2s_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')
    _ = qa_s2s_model.eval()

    # prepare IR index
    if not os.path.isfile('marvel_passages_reps_32_l-8_h-768_b-512-512.dat'):
        print("*** Generating dense index ***")
        make_qa_dense_index(
            qar_model,
            qar_tokenizer,
            wiki40b_snippets,
            device='cuda:0',
            index_name='marvel_passages_reps_32_l-8_h-768_b-512-512.dat',
        )

    faiss_res = faiss.StandardGpuResources()
    wiki40b_passage_reps = np.memmap(
        'marvel_passages_reps_32_l-8_h-768_b-512-512.dat',
        dtype='float32',
        mode='r',
        shape=(wiki40b_snippets.num_rows, 128),
    )

    wiki40b_index_flat = faiss.IndexFlatIP(128)
    wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 1, wiki40b_index_flat)
    wiki40b_gpu_index.add(wiki40b_passage_reps)

    # run examples
    questions = []
    answers = []

    for question in questions:
        # create support document with the dense index
        doc, res_list = query_qa_dense_index(
            question, qar_model, qar_tokenizer, wiki40b_snippets, wiki40b_gpu_index, device='cuda:0'
        )
        # concatenate question and support document into BART input
        question_doc = "question: {} context: {}".format(question, doc)
        # generate an answer with beam search
        answer = qa_s2s_generate(
            question_doc,
            qa_s2s_model,
            qa_s2s_tokenizer,
            num_answers=1,
            num_beams=8,
            min_len=64,
            max_len=256,
            max_input_length=1024,
            device="cuda:0",
        )[0]
        questions += [question]
        answers += [answer]

    df = pd.DataFrame({'Question': questions, 'Answer': answers,})
    df.style.set_properties(**{'text-align': 'left'})


def short_test():
    qa_s2s_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
    qa_s2s_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')
    _ = qa_s2s_model.eval()

    # concatenate question and support document into BART input
    # question = "who is iron man?"
    # question = "who is Tony stark?"
    # question = "who directed iron man?"
    question = "How did Stark escape from Titan?"
    # question = "Who is Anthony Stark daughter?"

    # doc = "Iron Man is a 2008 American superhero film based on the Marvel Comics character of the same name. Produced by Marvel Studios and distributed \
    #    by Paramount Pictures. It is the first film in the Marvel Cinematic Universe. It was directed by Jon Favreau from a screenplay \
    #    by the writing teams of Mark Fergus and Hawk Ostby, and Art Marcum and Matt Holloway, and stars Robert Downey Jr. as Tony Stark / Iron Man \
    #    alongside Terrence Howard, Jeff Bridges, Shaun Toub, and Gwyneth Paltrow. In the film, following his escape from captivity by a terrorist group, \
    #    world famous industrialist and master engineer Tony Stark builds a mechanized suit of armor and becomes the superhero Iron Man."
    # doc = "Anthony Edward Stark is a fictional character portrayed by Robert Downey Jr in the Marvel Cinematic Universe (MCU) film " \
    #      "franchise—based on the Marvel Comics character of the same name—commonly known by his alter ego, Iron Man. In 2018, when Thanos " \
    #      "and the Black Order invaded Earth in their conquest to acquire the six Infinity Stones, Stark, Doctor Strange, and Spider-Man " \
    #      "convened to battle Thanos on Titan with the help of the Guardians of the Galaxy. When Stark was held at Thanos' mercy, " \
    #      "Doctor Strange surrendered the Time Stone for Stark's life. After the Snap, Stark and Nebula remained the sole survivors on Titan. " \
    #      "Stark and Nebula used the Benatar to escape Titan, but were stranded in space as the ship was damaged. They were rescued by " \
    #      "Captain Marvel, who brought them back to Earth. In the five years after the Snap, Stark chose to retire from being Iron Man, " \
    #      "marrying Potts and having a daughter, Morgan Stark. When Stark discovered the key to travel through time, he rejoined the Avengers " \
    #      "to undo the Snap, traveling back in time to retrieve the Scepter and then to regain the Tesseract. During the Battle of Earth, " \
    #      "Stark heroically sacrificed himself to eliminate Thanos and his armies, who traveled through time to collect the Infinity Stones, " \
    #      "saving the universe from decimation, and leaving behind a legacy as one of Earth's most revered superheroes."

    doc = (
        "After the Snap, Stark and Nebula remained the sole survivors on Titan. "
        "Stark and Nebula used the Benatar to escape Titan, but were stranded in space as the ship was damaged. They were rescued by "
        "Captain Marvel, who brought them back to Earth. In the five years after the Snap, Stark chose to retire from being Iron Man, "
        "marrying Potts and having a daughter, Morgan Stark. When Stark discovered the key to travel through time, he rejoined the Avengers "
        "to undo the Snap, traveling back in time to retrieve the Scepter and then to regain the Tesseract. During the Battle of Earth, "
        "Stark heroically sacrificed himself to eliminate Thanos and his armies, who traveled through time to collect the Infinity Stones, "
        "saving the universe from decimation, and leaving behind a legacy as one of Earth's most revered superheroes."
    )

    question_doc = "question: {} context: {}".format(question, doc)
    # generate an answer with beam search
    answers = qa_s2s_generate(
        question_doc,
        qa_s2s_model,
        qa_s2s_tokenizer,
        num_answers=2,
        num_beams=4,
        min_len=30,
        max_len=150,
        max_input_length=1024,
        device="cuda:0",
    )
    print(answers)


if __name__ == '__main__':
    short_test()
    # full_test()
