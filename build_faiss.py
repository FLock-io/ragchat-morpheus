import faiss
import json
import numpy as np
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from sentence_transformers import SentenceTransformer


dimension = 384
index = faiss.IndexFlatL2(dimension)

def load_all_urls_to_documents(urls):
    """The input txt file should contains one url per line.

    Args:
        url_file (str): Path to txt file containing all urls to be indexed
    """


    loader = AsyncChromiumLoader(urls)
    html = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p", "li", "div", "a"])
    documents = [doc.to_json()["kwargs"] for doc in docs_transformed]
    return documents


def split_text_fixed_size(text, chunk_size, overlap_size):
    new_text = []
    for i in range(0, len(text), chunk_size):
        if i == 0:
            new_text.append(text[0:chunk_size])
        else:
            new_text.append(text[i - overlap_size:i + chunk_size])
            # new_text.append(text[i:i + chunk_size])
    return new_text


if __name__ == '__main__':

    pdf_content =[]
    text = load_all_urls_to_documents(
        ["https://mor.org/about"]
    )
    documents = split_text_fixed_size(text, chunk_size=2048, overlap_size=300)
    with open("all_parsed_urls.json", "w") as f:
        json.dump(documents, f)
    # for chunk_text in new_text:
    #     pdf_content.append({
    #         'content': chunk_text
    #     })
    # sent_model = SentenceTransformer(
    #     r'infgrad/stella-base-zh-v3-1792d'
    # )
    # content_sentences = [x['content'] for x in pdf_content]
    # sent_embeddings = sent_model.encode(content_sentences, normalize_embeddings=True)
    # index.add(np.array(sent_embeddings))
