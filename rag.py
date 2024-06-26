# import numpy as np

from rank_bm25 import BM25Okapi
import jieba,json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

device = "cuda"

# rerank_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
# rerank_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
# rerank_model.cuda()

model_path = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    # trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    torch_dtype="auto",
    device_map="auto",
    # trust_remote_code=True
    # attn_implementation="flash_attention_2"
)


# 对长文本进行切分
def split_text_fixed_size(text, chunk_size, overlap_size):
    new_text = []
    for i in range(0, len(text), chunk_size):
        if i == 0:
            new_text.append(text[0:chunk_size])
        else:
            new_text.append(text[i - overlap_size:i + chunk_size])
            # new_text.append(text[i:i + chunk_size])
    return new_text


def get_rank_index(max_score_page_idxs_, questions_, pdf_content_):
    pairs = []
    for idx in max_score_page_idxs_:
        pairs.append([questions_[query_idx]["question"], pdf_content_[idx][0]['page_content']])

    inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    max_score = scores.cpu().numpy().argmax()
    index = max_score_page_idxs_[max_score]

    return max_score, index


def read_data(query_data_path, knowledge_data_path):
    with open(query_data_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    with open(knowledge_data_path, 'r', encoding='utf-8') as f:
        knowledge_data = json.load(f)

    return questions, knowledge_data


def qwen_preprocess(tokenizer_, kenowledge, question):
    """
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me who you are."},
        {"role": "assistant", "content": "I am a large language model named Qwen..."}
    ]
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},

        {"role": "user", "content": f"""You are an intelligent assistant. Please summarize the content of the knowledge base to answer the question. Please list the data in the knowledge base and answer in detail. When all knowledge base content is irrelevant to the question, your answer must include the sentence "The answer you are looking for is not found in the knowledge base!" Answers need to consider chat history.
      Here is the knowledge base:
      {kenowledge}
      The above is the knowledge base.
      Here is the question:
      {question}"""},
    ]

    text = tokenizer_.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs_ = tokenizer_([text], return_tensors="pt").to(device)

    input_ids = tokenizer_.encode(text, return_tensors='pt')
    attention_mask_ = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    # print(model_inputs)
    # sys.exit()
    return model_inputs_, attention_mask_


if __name__ == '__main__':

    questions, content = read_data(query_data_path=r"questions.json",
                              knowledge_data_path=r'all_parsed_urls.json')
    content = split_text_fixed_size(content[0][0]['page_content'], chunk_size=2048, overlap_size=300)
    content_words = [jieba.lcut(x) for x in content]
    bm25 = BM25Okapi(content_words)


    sent_model = SentenceTransformer(
        r'all-MiniLM-L6-v2'
    )

    question_sentences = [x['question'] for x in questions]
    pdf_content_sentences = [x for x in content]

    question_embeddings = sent_model.encode(question_sentences, normalize_embeddings=True)
    pdf_embeddings = sent_model.encode(pdf_content_sentences, normalize_embeddings=True)

    for query_idx in range(len(questions)):

        doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
        bm25_score_page_idxs = doc_scores.argsort()[-10:]

        score = question_embeddings[query_idx] @ pdf_embeddings.T
        ste_score_page_idxs = score.argsort()[-10:]

        # out of memory,so
        import gc
        gc.collect()
        torch.cuda.empty_cache()


        max_score_page_idx = bm25_score_page_idxs[0]
        # questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)
        # questions[query_idx]['reference'] = pdf_content[max_score_page_idxs]['page']

        # bm25_score, bm25_index = get_rank_index(bm25_score_page_idxs, questions, content)
        # ste_score, ste_index = get_rank_index(ste_score_page_idxs, questions, content)
        #
        # max_score_page_idx = 0
        # if ste_score >= bm25_score:
        #     questions[query_idx]['reference'] = 'page_' + str(ste_index + 1)
        #     max_score_page_idx = ste_index
        # else:
        #     questions[query_idx]['reference'] = 'page_' + str(bm25_index + 1)
        #     max_score_page_idx = bm25_index

        model_inputs, attention_mask = qwen_preprocess(
            tokenizer, content[max_score_page_idx], questions[query_idx]["question"]
        )

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # print(response)
        # answer = ask_glm(pdf_content[max_score_page_idx]['content'], questions[query_idx]["question"])
        print(f'question: {questions[query_idx]["question"]}, answer: {response}')
