from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


class VectorService:
    """Upserts document chunks into Pinecone using LangChain's PineconeVectorStore.

    Uses ``OpenAIEmbeddings`` (langchain-openai) for dense embeddings and
    ``PineconeVectorStore`` (langchain-pinecone) to manage index interactions.

    The Pinecone index must use the ``cosine`` metric (default for dense-only
    indexes).  If you later want true server-side hybrid (sparse + dense on
    the same vector), the index metric must be ``dotproduct`` - see the
    README for migration notes.

    Environment variables:
        PINECONE_API_KEY    Pinecone API key (required)
        PINECONE_INDEX      Name of the Pinecone index (required)
        OPENAI_API_KEY      OpenAI API key (required)
        EMBEDDING_MODEL     OpenAI embedding model  (default: text-embedding-3-small)
        EMBEDDING_DIM       Embedding dimensions    (default: 1536)
        PINECONE_CLOUD      Serverless cloud        (default: aws)
        PINECONE_REGION     Serverless region       (default: us-east-1)
    """

    def __init__(self) -> None:
        index_name = os.environ["PINECONE_INDEX"]
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")

        # Dense embedding model
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

        # Pinecone client - create index on first use if absent
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self._index = pc.Index(index_name)

        # LangChain vector store wrapper
        self._store = PineconeVectorStore(
            embedding=self._embeddings,
            index=self._index,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_records(
        self,
        records: list[dict[str, Any]],
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Convert ingest records to LangChain Documents and upsert into Pinecone.

        Each record (output of ``DocumentIngestionService.ingest_file()``) must
        contain at minimum ``doc_id`` and ``parent_text``.

        The ``doc_id`` is used as the stable Pinecone vector ID so that
        re-ingesting the same document overwrites existing vectors rather than
        creating duplicates.

        Args:
            records:   List of dicts from the ingest service.
            namespace: Pinecone namespace; falls back to the ``namespace``
                       field on each record, then to the default namespace.

        Returns:
            ``{"upserted_count": int}``
        """
        if not records:
            return {"upserted_count": 0}

        docs: list[Document] = []
        ids: list[str] = []

        for record in records:
            text = record.get("parent_text") or record.get("content") or ""
            # Everything except the text itself becomes LangChain metadata
            metadata = {k: v for k, v in record.items() if k not in ("parent_text", "content")}
            docs.append(Document(page_content=text, metadata=metadata))
            ids.append(record.get("doc_id") or str(uuid4()))

        ns = namespace or (records[0].get("namespace") if records else None)

        kwargs: dict[str, Any] = {"documents": docs, "ids": ids}
        if ns:
            kwargs["namespace"] = ns

        self._store.add_documents(**kwargs)
        return {"upserted_count": len(docs)}

    def delete_records(self, doc_ids: list[str], namespace: str | None = None) -> None:
        """Delete vectors from Pinecone by their stable IDs."""
        kwargs: dict[str, Any] = {"ids": doc_ids}
        if namespace:
            kwargs["namespace"] = namespace
        self._index.delete(**kwargs)

    def as_retriever(self, top_k: int = 5, namespace: str | None = None, filter: dict[str, Any] | None = None):
        """Return a LangChain ``VectorStoreRetriever`` backed by this Pinecone index.

        Useful for composing with ``EnsembleRetriever`` in the retriever service.
        """
        search_kwargs: dict[str, Any] = {"k": top_k}
        if namespace:
            search_kwargs["namespace"] = namespace
        if filter:
            search_kwargs["filter"] = filter
        return self._store.as_retriever(search_kwargs=search_kwargs)
