import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from chromadb import Collection, QueryResult
from langchain.docstore.document import Document
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidDimensionException


class ChromaDB():
    """Vector database using ChromaDB."""

    BATCH_SIZE = 100

    def __init__(self, directory="db", allow_reset=True):
        """
        Args:
            directory: where to save
            allow_reset:
        """

        self.settings = Settings(anonymized_telemetry=False)
        self.settings.allow_reset = allow_reset

        self.settings.persist_directory = directory
        self.settings.is_persistent = True

        self.client = chromadb.Client(self.settings)

    def _generate_where_clause(self, where: Dict[str, any]) -> str:
        # If only one filter is supplied, return it as is
        # (no need to wrap in $and based on chroma docs)
        if len(where.keys()) <= 1:
            return where
        where_filters = []
        for k, v in where.items():
            if isinstance(v, str):
                where_filters.append({k: v})
        return {"$and": where_filters}

    def get(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, any]] = None, limit: Optional[int] = None):
        """
        Get existing doc ids present in vector database

        :param ids: list of doc ids to check for existence
        :type ids: List[str]
        :param where: Optional. to filter data
        :type where: Dict[str, Any]
        :param limit: Optional. maximum number of documents
        :type limit: Optional[int]
        :return: Existing documents.
        :rtype: List[str]
        """
        args = {}
        if ids:
            args["ids"] = ids
        if where:
            args["where"] = self._generate_where_clause(where)
        if limit:
            args["limit"] = limit
        return self.collection.get(**args)

    def add(
            self,
            embeddings: List[List[float]],
            documents: List[str],
            metadatas: List[object],
            ids: List[str],
            skip_embedding: bool,
            **kwargs: Optional[Dict[str, Any]],
    ) -> Any:
        """
        Add vectors to chroma database

        :param embeddings: list of embeddings to add
        :type embeddings: List[List[str]]
        :param documents: Documents
        :type documents: List[str]
        :param metadatas: Metadatas
        :type metadatas: List[object]
        :param ids: ids
        :type ids: List[str]
        :param skip_embedding: Optional. If True, then the embeddings are assumed to be already generated.
        :type skip_embedding: bool
        """
        size = len(documents)
        if skip_embedding and (embeddings is None or len(embeddings) != len(documents)):
            raise ValueError("Cannot add documents to chromadb with inconsistent embeddings")

        if len(documents) != size or len(metadatas) != size or len(ids) != size:
            raise ValueError(
                "Cannot add documents to chromadb with inconsistent sizes. Documents size: {}, Metadata size: {},"
                " Ids size: {}".format(len(documents), len(metadatas), len(ids))
            )

        for i in tqdm(range(0, len(documents), self.BATCH_SIZE), desc="Inserting batches in chromadb"):
            if skip_embedding:
                self.collection.add(
                    embeddings=embeddings[i: i + self.BATCH_SIZE],
                    documents=documents[i: i + self.BATCH_SIZE],
                    metadatas=metadatas[i: i + self.BATCH_SIZE],
                    ids=ids[i: i + self.BATCH_SIZE],
                )
            else:
                self.collection.add(
                    documents=documents[i: i + self.BATCH_SIZE],
                    metadatas=metadatas[i: i + self.BATCH_SIZE],
                    ids=ids[i: i + self.BATCH_SIZE],
                )

    def _format_result(self, results: QueryResult) -> list[tuple[Document, float]]:
        """
        Format Chroma results

        :param results: ChromaDB query results to format.
        :type results: QueryResult
        :return: Formatted results
        :rtype: list[tuple[Document, float]]
        """
        return [
            (Document(page_content=result[0], metadata=result[1] or {}), result[2])
            for result in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def query(
            self,
            input_query: List[str],
            n_results: int,
            where: Dict[str, any],
            skip_embedding: bool,
            citations: bool = False,
            **kwargs: Optional[Dict[str, Any]],
    ) -> Union[List[Tuple[str, str, str]], List[str]]:
        """
        Query contents from vector database based on vector similarity

        :param input_query: list of query string
        :type input_query: List[str]
        :param n_results: no of similar documents to fetch from database
        :type n_results: int
        :param where: to filter data
        :type where: Dict[str, Any]
        :param skip_embedding: Optional. If True, then the input_query is assumed to be already embedded.
        :type skip_embedding: bool
        :param citations: we use citations boolean param to return context along with the answer.
        :type citations: bool, default is False.
        :raises InvalidDimensionException: Dimensions do not match.
        :return: The content of the document that matched your query,
        along with url of the source and doc_id (if citations flag is true)
        :rtype: List[str], if citations=False, otherwise List[Tuple[str, str, str]]
        """
        try:
            if skip_embedding:
                result = self.collection.query(
                    query_embeddings=input_query,
                    n_results=n_results,
                    where=self._generate_where_clause(where),
                    **kwargs,
                )
            else:
                result = self.collection.query(
                    query_texts=input_query,
                    n_results=n_results,
                    where=self._generate_where_clause(where),
                    **kwargs,
                )
        except InvalidDimensionException as e:
            raise InvalidDimensionException(
                e.message()
                + ". This is commonly a side-effect when an embedding function, different from the one used to add the"
                  " embeddings, is used to retrieve an embedding from the database."
            ) from None
        results_formatted = self._format_result(result)
        contexts = []
        for result in results_formatted:
            context = result[0].page_content
            if citations:
                metadata = result[0].metadata
                source = metadata["url"]
                doc_id = metadata["doc_id"]
                contexts.append((context, source, doc_id))
            else:
                contexts.append(context)
        return contexts

    def set_collection_name(self, name, embedding_fn):
        if not isinstance(name, str):
            raise TypeError("Collection name must be a string")

        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_fn,
        )


if __name__ == '__main__':
    from langchain_community.embeddings import OllamaEmbeddings
    from collections.abc import Callable
    from chromadb.api.types import Embeddable, EmbeddingFunction, Embeddings


    class EmbeddingFunc(EmbeddingFunction):
        def __init__(self, embedding_fn: Callable[[list[str]], list[str]]):
            self.embedding_fn = embedding_fn

        def __call__(self, input: Embeddable) -> Embeddings:
            return self.embedding_fn(input)


    embeddings = OllamaEmbeddings(model="all-minilm:latest", base_url='http://localhost:11434')
    vectordb = ChromaDB(directory=r"E:\flock\ragchat-morpheus\db")
    vectordb.set_collection_name(name="my-collection", embedding_fn=EmbeddingFunc(embeddings.embed_documents))
    print(vectordb.query(["what is Morpheus?"], 3, {}, False))
