import requests
from typing import List, Optional

DEFAULT_PROMPT_TEMPLATE = """Reference Information: {context}
Please answer the question based on the above reference information. The previous reference information may be useful or not; you need to select the content that might be relevant to the question to provide a basis for your answer.
The answer must be faithful to the original text, concise but without losing information, and do not fabricate anything if there is no relevant content in the reference information.
Please respond in English.
Question: {question}
"""


class AnswerResult:
    history: List[List[str]] = []
    llm_output: Optional[dict] = None


class QwenChat():
    def __init__(self):
        super().__init__()
        self.prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.server_url = "http://localhost:11434/api/generate"
        self.model_name = "qwen2:1.5b"

    async def stream_chat(self, prompt: str, history: List[List[str]] = [], **kw):

        msg_history = []
        # msg_history.append({"role": "system", "content": self.config["system"]})
        if len(history) > 0:
            for q, a in history:
                msg_history.append({"role": "user", "content": q})
                msg_history.append({"role": "assistant", "content": a})
        msg_history.append({"role": "user", "content": prompt})
        headers = {'Content-Type': 'application/json'}
        chat_response = requests.post(self.server_url,
                                      headers=headers,
                                      json=msg_history)
        history += [[]]
        for chunk in chat_response:
            choices = chunk.choices[0]
            if chunk.choices[0].finish_reason == "stop":
                break
            else:
                r = choices.delta.content
                if r is None:
                    r = ""
                history[-1] = [prompt, r]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": r}
                yield answer_result

