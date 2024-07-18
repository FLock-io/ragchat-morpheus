
from modules.embedding_model import OllamaEmbed
from modules.vector_db import ChromaDB
from modules.rerank_model import BaseRank,XinferenceRerank
from modules.QwenChat import QwenChat



def main():
    while True:
        history = []
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        print("\n\n> Question:")
        print(query)

        ## retrieval

        emb = OllamaEmbed()

        vectordb = ChromaDB(directory=r"db")
        vectordb.set_collection_name(name="my-collection", embedding_fn=emb.langchain_default_concept())
        knb = vectordb.query([query], 3, {'app_id': 'morpheus'}, False)

        ## rerank
        rerank_model = BaseRank()
        index = rerank_model.text_pair_sort(query, knb)
        knb_rerank = [knb[i] for i in index]

        ## chat stream
        llm_chat = QwenChat()
        result = llm_chat.stream_chat(prompt=query, context='\n'.join(knb_rerank), history=history)
        for chunk in result:
            print(chunk, end="", flush=True)
        history.append({
            "role": "user",
            "content": query
        })
        history.append({
            "role": "assistant",
            "content": result
        })
        # print(answer)


if __name__ == "__main__":
    main()
