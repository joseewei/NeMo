from lfqa_utils import *
import socket

HOST = '10.110.42.102'
PORT = 25556


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


def load_indexes(qar_model, qar_tokenizer):
    snippets = load_data("/home/vgetselevich/data/minecraft/", 100)

    # prepare IR index
    if not os.path.isfile('minecraft_passages_reps_32_l-8_h-768_b-512-512.dat'):
        print("*** Generating dense index ***")
        make_qa_dense_index_text_chunks(
            qar_model,
            qar_tokenizer,
            snippets,
            device='cuda:0',
            index_name='minecraft_passages_reps_32_l-8_h-768_b-512-512.dat',
        )

    faiss_res = faiss.StandardGpuResources()
    #    wiki40b_passages = nlp.load_dataset(path="wiki_snippets", name="wiki40b_en_100_0")["train"]
    passage_reps = np.memmap(
        "minecraft_passages_reps_32_l-8_h-768_b-512-512.dat",
        dtype="float32",
        mode="r",
        shape=(len(snippets), 128),
    )
    wiki40b_index_flat = faiss.IndexFlatIP(128)
    wiki40b_gpu_index_flat = faiss.index_cpu_to_gpu(faiss_res, 1, wiki40b_index_flat)
    wiki40b_gpu_index_flat.add(passage_reps)  # TODO fix for larger GPU

    return (snippets, wiki40b_gpu_index_flat)


def get_answer(question, qar_tokenizer, qar_model, qa_s2s_tokenizer, qa_s2s_model, snippets, gpu_dense_index, use_eqa=True):
    # create support document with the dense index
    doc, res_list = query_qa_dense_index(
        question, qar_model, qar_tokenizer, snippets, gpu_dense_index, n_results=5, device='cuda:0'
    )
    # concatenate question and support document into BART input
    question_doc = "question: {} context: {}".format(question, doc)
    # generate an answer with beam search
    gen_answer = qa_s2s_generate(
        question_doc,
        qa_s2s_model,
        qa_s2s_tokenizer,
        num_answers=1,
        num_beams=8,
        min_len=10,
        max_len=50,
        max_input_length=1024,
        device="cuda:0",
    )[0]

    print(question)
    print("Generative Answer: " + gen_answer)
    print("Context: " + doc.replace("<P> ", ""))

    ex_answer_span = ''

    if use_eqa:
        ex_answer_span, ex_answer_context = extractive_qa(question, doc.replace("<P> ", ""))
        print("Extractive Answer: " + ex_answer_span)

    return (gen_answer, ex_answer_span)


def main():
    qar_tokenizer, qar_model, qa_s2s_tokenizer, qa_s2s_model = load_models()
    snippets, gpu_dense_index = load_indexes(qar_model, qar_tokenizer)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))

        print(f"IR + QA server is running on: {HOST}:{PORT}")
        s.listen()

        while True:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while True:
                    data = conn.recv(1024).decode()
                    if not data:
                        break

                    gen_answer, ex_answer_span = get_answer(data,  qar_tokenizer, qar_model, qa_s2s_tokenizer,
                                                            qa_s2s_model, snippets, gpu_dense_index, use_eqa=False)
                    if ex_answer_span:
                        conn.sendall(ex_answer_span.encode('utf-8'))
                    else:
                        conn.sendall(gen_answer.encode('utf-8'))

                    # response = f'Extractive: {ex_answer_span} Generative: {gen_answer}'
                    # conn.sendall(response.encode('utf-8'))



if __name__ == '__main__':
    main()
