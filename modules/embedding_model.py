import requests
import json
from abc import ABC
from typing import Optional, List, Any
from fastapi import (
    HTTPException,
    status
)


class Base(ABC):
    def __init__(self, url, model_name):
        pass

    def text_embedding(self, texts: list):
        raise NotImplementedError("Please implement encode method!")


class Embed(Base):
    def __init__(
            self,
            url: str = "http://localhost:11434/api/embeddings",
            model_name: str = "all-minilm:l6-v2",
    ):
        self.url = url
        self.model_name = model_name

    def text_embedding(self, texts: List[str]) -> List[Any]:
        payload = json.dumps({"texts": texts})
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.url,
                                 headers=headers,
                                 json=payload)
        if response.status_code == 200:
            ebd = response.json()["embedding"]
            return ebd
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="text embedding server error")
