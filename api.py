import json
import time

from typing import Optional, List, Dict

from pydantic import BaseModel, Field

from starlette.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request
from modules.embedding_model import OllamaEmbed
from modules.vector_db import ChromaDB
from modules.rerank_model import BaseRank, XinferenceRerank
from modules.QwenChat import QwenChat

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


async def _resp_async_generator(request):
    history = []
    if len(request.messages) > 1:
        history = [{'role': temp.role, 'content': temp.content} for temp in request.messages[:-1]]
    query = request.messages[-1].content

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
    if request.stream:
        return StreamingResponse(
            _resp_async_generator(request), media_type="application/x-ndjson"
        )
    else:
        history = []
        if len(request.messages) > 1:
            history = [{'role': temp.role, 'content': temp.content} for temp in request.messages[:-1]]
        query = request.messages[-1].content
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
