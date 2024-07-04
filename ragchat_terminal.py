import os
from embedchain import App

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
app = App.from_config(config_path="config.yaml")


def main():
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        print("\n\n> Question:")
        print(query)
        answer = app.query(query, session_id='user1')

        print(answer)
        # for chunk in answer:
        #     print(chunk, end="", flush=True)

        # Print the relevant sources used for the answer
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)


if __name__ == "__main__":
    main()
