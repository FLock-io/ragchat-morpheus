import os


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
        from modules.embedding_model import OllamaEmbed
        emb = OllamaEmbed()
        from modules.vector_db import ChromaDB
        vectordb = ChromaDB(directory=r"db")
        vectordb.set_collection_name(name="my-collection", embedding_fn=emb.langchain_default_concept())

        knb = vectordb.query([query], 3, {}, False)

        ## rerank

        ## chat stream
        from modules.QwenChat import QwenChat

        llm_chat = QwenChat()
        result = llm_chat.stream_chat(prompt=query, context='\n'.join(knb), history=history)
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
