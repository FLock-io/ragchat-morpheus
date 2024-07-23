from langchain_community.embeddings import OllamaEmbeddings
from typing import Callable, List
from chromadb.api.types import Embeddable, EmbeddingFunction, Embeddings



class EmbeddingFunc(EmbeddingFunction):
    def __init__(self, embedding_fn: Callable[[List[str]], List[str]]):
        self.embedding_fn = embedding_fn

    def __call__(self, input: Embeddable) -> Embeddings:
        return self.embedding_fn(input)



class OllamaEmbed():
    def __init__(
            self,
            url: str = "http://ollama:11434",
            model_name: str = "all-minilm:latest",
    ):
        self.url = url
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=self.model_name, base_url=self.url)

    def langchain_default_concept(self):
        return EmbeddingFunc(self.embeddings.embed_documents)

