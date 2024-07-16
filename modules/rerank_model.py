
import re
import  threading
import requests
import torch
from FlagEmbedding import FlagReranker
from huggingface_hub import snapshot_download
import os
from abc import ABC
import numpy as np
from api.utils.file_utils import get_home_cache_dir
from rag.utils import num_tokens_from_string, truncate

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def similarity(self, query: str, texts: list):
        raise NotImplementedError("Please implement encode method!")


class DefaultRerank(Base):
    _model = None
    _model_lock = threading.Lock()

    def __init__(self, key, model_name, **kwargs):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """
        if not DefaultRerank._model:
            with DefaultRerank._model_lock:
                if not DefaultRerank._model:
                    try:
                        DefaultRerank._model = FlagReranker(os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z]+/", "", model_name)), use_fp16=torch.cuda.is_available())
                    except Exception as e:
                        model_dir = snapshot_download(repo_id= model_name,
                                                      local_dir=os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z]+/", "", model_name)),
                                                      local_dir_use_symlinks=False)
                        DefaultRerank._model = FlagReranker(model_dir, use_fp16=torch.cuda.is_available())
        self._model = DefaultRerank._model

    def similarity(self, query: str, texts: list):
        pairs = [(query,truncate(t, 2048)) for t in texts]
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)
        batch_size = 4096
        res = []
        for i in range(0, len(pairs), batch_size):
            scores = self._model.compute_score(pairs[i:i + batch_size], max_length=2048)
            scores = sigmoid(np.array(scores)).tolist()
            if isinstance(scores, float): res.append(scores)
            else:  res.extend(scores)
        return np.array(res), token_count

