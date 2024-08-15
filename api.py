import json
import time
import os
import sys

# if linux
if os.name == 'posix':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import Optional, List, Dict
from pydantic import BaseModel, Field

from starlette.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request
from modules.embedding_model import OllamaEmbed
from modules.vector_db import ChromaDB
from modules.rerank_model import BaseRank, XinferenceRerank
from modules.LLMChat import OllamaChat
from utils.load_config import from_config

config_data = from_config("config.yaml")
LLM_MODEL = config_data["llm"]["config"]["model"]
LLM_MODEL_URL = config_data["llm"]["config"]["base_url"]
EMB_MODEL = config_data["embedding"]["config"]["model"]
EMB_MODEL_URL = config_data["embedding"]["config"]["base_url"]
RERANK_MODEL = config_data["rerank"]["config"]["model"]
VECTORDB_NAME = config_data["vectordb"]["config"]["collection_name"]
VECTORDB_NAME_DIR = config_data["vectordb"]["config"]["dir"]

app = FastAPI(title="OpenAI-compatible API")


# data models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


async def _resp_async_generator(request, query, knb_rerank, history):
    ## chat stream
    llm_chat = OllamaChat(server_url=LLM_MODEL_URL, model_name=LLM_MODEL)
    result = llm_chat.stream_chat(prompt=query, context='\n'.join(knb_rerank), history=history)

    for i, token in enumerate(result):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": request.model,
            "choices": [{"delta": {"content": token}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    history = []
    if len(request.messages) > 1:
        history = [{'role': temp.role, 'content': temp.content} for temp in request.messages[:-1]]
    query = request.messages[-1].content
    emb = OllamaEmbed(url=EMB_MODEL_URL, model_name=EMB_MODEL)

    vectordb = ChromaDB(directory=VECTORDB_NAME_DIR)
    vectordb.set_collection_name(name=VECTORDB_NAME, embedding_fn=emb.langchain_default_concept())
    knb = vectordb.query([query], 3, {'app_id': 'morpheus'}, False)

    ## rerank
    rerank_model = BaseRank(model_name=RERANK_MODEL)
    index = rerank_model.text_pair_sort(query, knb)
    knb_rerank = [knb[i] for i in index]

    if request.stream:
        return StreamingResponse(
            _resp_async_generator(request, query, knb_rerank, history), media_type="application/x-ndjson"
        )
    else:
        ## chat
        llm_chat = OllamaChat(server_url=LLM_MODEL_URL, model_name=LLM_MODEL)
        result = llm_chat.chat(prompt=query, context='\n'.join(knb_rerank), history=history)

        return {
            "id": "1337",
            "object": "chat.completion",
            "created": time.time(),
            "model": request.model,
            "choices": [{"message": Message(role="assistant", content=result)}],
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
