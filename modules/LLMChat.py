import requests
from typing import List, Optional
from ollama import Client

DEFAULT_PROMPT = """
You are a Q&A expert system. Your responses must always be rooted in the context provided for each query. Here are some guidelines to follow:

1. Refrain from explicitly mentioning the context provided in your response.
2. The context should silently guide your answers without being directly acknowledged.
3. Do not use phrases such as 'According to the context provided', 'Based on the context, ...' etc.

Context information:
----------------------
{context}
----------------------

Query: {question}
Answer:
"""  # noqa:E501


class AnswerResult:
    history: List[List[str]] = []
    llm_output: Optional[dict] = None


class OllamaChat():
    def __init__(self, server_url="http://ollama:11434/api/chat", model_name="llama3:8b"):
        super().__init__()
        self.prompt_template = DEFAULT_PROMPT
        self.client = Client(host=server_url)
        self.model_name = model_name

    def stream_chat(self, prompt: str, context: str, history: List = []):

        # if len(history) > 0:
        #     for q, a in history:
        #         msg_history.append({"role": "user", "content": q})
        #         msg_history.append({"role": "assistant", "content": a})
        history.append({"role": "user",
                        "content": self.prompt_template.replace("{context}", context).
                       replace("{question}", prompt)})
        chat_response = self.client.chat(
            model=self.model_name,
            stream=True,
            messages=history)

        for chunk in chat_response:
            if chunk["done"] == True:
                break
            else:
                result = chunk["message"]["content"]
                if result is None:
                    result = ""
                yield result

    def chat(self, prompt: str, context: str, history: List = []):

        history.append({"role": "user",
                        "content": self.prompt_template.replace("{context}", context).
                       replace("{question}", prompt)})
        chat_response = self.client.chat(
            model=self.model_name,
            messages=history)

        return chat_response["message"]["content"]

