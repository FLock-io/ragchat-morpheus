import torch
import os
import re
import numpy as np
import requests
import tiktoken
from fastapi import (
    HTTPException,
    status
)
from FlagEmbedding import FlagReranker
from huggingface_hub import snapshot_download
from typing import Any, Dict, List, Optional, Tuple, Union

feature_min_score = 0.52

encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

device_is_cpu = True


def get_home_cache_dir():
    dir = os.path.join(os.path.expanduser('~'), ".FLock")
    try:
        os.mkdir(dir)
    except OSError as error:
        pass
    return dir


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def truncate(string: str, max_len: int) -> int:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoder.encode(string))
    return num_tokens


class BaseRank():
    def __init__(self, model_name='BAAI/bge-reranker-base'):
        try:
            self.model = FlagReranker(os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z]+/", "", model_name)),
                                      use_fp16=False if device_is_cpu else torch.cuda.is_available(),
                                      device='cpu' if device_is_cpu else None
                                      )
        except Exception as e:
            model_dir = snapshot_download(repo_id=model_name,
                                          local_dir=os.path.join(get_home_cache_dir(),
                                                                 re.sub(r"^[a-zA-Z]+/", "", model_name)),
                                          local_dir_use_symlinks=False)
            self.model = FlagReranker(model_dir,
                                      use_fp16=False if device_is_cpu else torch.cuda.is_available(),
                                      device='cpu' if device_is_cpu else None
                                      )

    def text_pair_sort(self, query: str, compare: List) -> List[int]:
        pairs = [(query, truncate(t, 2048)) for t in compare]
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)
        batch_size = 4096
        res = []
        for i in range(0, len(pairs), batch_size):
            scores = self.model.compute_score(pairs[i:i + batch_size])
            scores = sigmoid(np.array(scores)).tolist()
            if isinstance(scores, float):
                res.append(scores)
            else:
                res.extend(scores)
        knb_index = [index for index, score in enumerate(res) if score > feature_min_score]
        return knb_index


class XinferenceRerank():
    def __init__(self, url="http://127.0.0.1:9997/v1/rerank", model_name="bge-reranker-base"):
        self.feature_server_url = url
        self.model_name = model_name

    def text_pair_sort(self, query: str, compare: List) -> List[int]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": compare
        }

        headers = {'Content-Type': 'application/json'}

        response = requests.post(self.feature_server_url,
                                 headers=headers,
                                 json=payload)
        if response.status_code == 200:
            sort_scores = response.json()["results"]
            knb_index = [index for index, score in enumerate(sort_scores) if
                         score['relevance_score'] > feature_min_score]
            return knb_index
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="text rerank server error")
