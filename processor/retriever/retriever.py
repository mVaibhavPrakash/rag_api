from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from pinecone import Pinecone
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict


# ---------------------------------------------------------------------------
# Data models returned by this service
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single document chunk returned from the hybrid retriever."""

    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Full retrieval + generation result from the RAG graph."""

    query: str
    chunks: list[RetrievedChunk]
    answer: str
    model: str
    usage: dict[str, int]


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class RAGState(TypedDict):
    """Typed state passed between LangGraph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    # Populated by the retrieve node; read by grade and generate nodes
    retrieved_docs: list[Document]
    # Rewrite counter — prevents infinite query-rewrite loops
    rewrite_count: int


# ---------------------------------------------------------------------------
# Relevance grader (structured output)
# ---------------------------------------------------------------------------


class RelevanceGrade(BaseModel):
    """Binary relevance score produced by the grader LLM."""

    score: Literal["yes", "no"] = Field(
        description="'yes' if the retrieved context is relevant to the question, else 'no'."
    )


# ---------------------------------------------------------------------------
# RetrieverService
# ---------------------------------------------------------------------------


class RetrieverService:
    """Hybrid retrieval (BM25 + dense Pinecone) + LangGraph RAG pipeline.

    Retrieval strategy
    ------------------
    1. **BM25** (``langchain-community``) — sparse keyword matching over the
       in-memory corpus of documents supplied at query time.  Best for exact
       term overlap and rare tokens.
    2. **Dense vector search** (``langchain-pinecone``) — semantic similarity
       via OpenAI embeddings stored in Pinecone.  Best for paraphrase and
       conceptual matches.
    3. **EnsembleRetriever** merges both result sets with Reciprocal Rank
       Fusion (RRF) and the configurable ``HYBRID_ALPHA`` weight.

    Generation pipeline (LangGraph)
    --------------------------------
    ``retrieve`` → ``grade`` → ``generate``
                       ↓ (irrelevant)
                   ``rewrite`` → ``retrieve`` (loop, max ``MAX_REWRITES``)

    Environment variables:
        PINECONE_API_KEY    Pinecone API key (required)
        PINECONE_INDEX      Pinecone index name (required)
        OPENAI_API_KEY      OpenAI API key (required)
        EMBEDDING_MODEL     Embedding model (default: text-embedding-3-small)
        LLM_MODEL           Chat model     (default: gpt-4o-mini)
        TOP_K               Chunks to retrieve per retriever (default: 5)
        HYBRID_ALPHA        EnsembleRetriever weight for dense retriever,
                            0.0 = pure BM25, 1.0 = pure dense (default: 0.6)
        MAX_REWRITES        Max query-rewrite loops before giving up (default: 2)
        SYSTEM_PROMPT       Optional custom system prompt for the generator LLM
    """

    _DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant for question-answering tasks. "
        "Use ONLY the provided context to answer. "
        "If the context is insufficient, say so clearly. "
        "Do not fabricate information."
    )

    def __init__(self) -> None:
        # -- Config --------------------------------------------------------
        self._top_k = int(os.getenv("TOP_K", "5"))
        self._alpha = float(os.getenv("HYBRID_ALPHA", "0.6"))
        self._max_rewrites = int(os.getenv("MAX_REWRITES", "2"))
        self._system_prompt = os.getenv("SYSTEM_PROMPT", self._DEFAULT_SYSTEM_PROMPT)
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        openai_api_key = os.environ["OPENAI_API_KEY"]

        # -- LangChain components ------------------------------------------
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key,
        )
        self._llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            openai_api_key=openai_api_key,
        )
        self._grader_llm = self._llm.with_structured_output(RelevanceGrade)
        self._rewriter_llm = ChatOpenAI(
            model=llm_model,
            temperature=0.3,  # slight creativity for reformulation
            openai_api_key=openai_api_key,
        )

        # -- Pinecone dense retriever --------------------------------------
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self._pinecone_index = pc.Index(os.environ["PINECONE_INDEX"])
        self._vector_store = PineconeVectorStore(
            embedding=self._embeddings,
            index=self._pinecone_index,
        )

        # -- Compile LangGraph once ----------------------------------------
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_and_answer(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
        corpus: list[str] | None = None,
    ) -> RetrievalResult:
        """Run the full hybrid-retrieval + LangGraph RAG pipeline.

        Args:
            query:     Natural-language question from the user.
            namespace: Pinecone namespace to search within.
            top_k:     Override the ``TOP_K`` env var for this call.
            filter:    Pinecone metadata filter dict (dense retriever only).
            corpus:    Optional list of raw text strings to seed the BM25
                       retriever.  When omitted, BM25 is seeded with the top
                       dense-retrieval results so it always has something to
                       work with even without an explicit corpus.

        Returns:
            :class:`RetrievalResult`
        """
        k = top_k if top_k is not None else self._top_k
        ensemble = self._build_ensemble(k=k, namespace=namespace, filter=filter, corpus=corpus)

        # Initial state — cast to Any so LangGraph's add_messages reducer
        # handles the dict-format HumanMessage at runtime (which it does).
        initial: Any = {
            "messages": [{"role": "user", "content": query}],
            "retrieved_docs": [],
            "rewrite_count": 0,
        }

        # Store ensemble and settings in closures via the graph config
        config = {
            "configurable": {
                "ensemble": ensemble,
                "top_k": k,
            }
        }

        final_state: RAGState = self._graph.invoke(initial, config=config)

        # --- Extract answer ---
        last_ai = next(
            (m for m in reversed(final_state["messages"]) if getattr(m, "type", None) == "ai"),
            None,
        )
        raw_content = last_ai.content if last_ai else None
        answer: str = str(raw_content) if raw_content is not None else "No answer generated."

        # --- Extract usage (summed across all LLM calls in the graph) ---
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for msg in final_state["messages"]:
            ru = getattr(msg, "response_metadata", {}).get("token_usage") or {}
            usage["prompt_tokens"] += ru.get("prompt_tokens", 0)
            usage["completion_tokens"] += ru.get("completion_tokens", 0)
            usage["total_tokens"] += ru.get("total_tokens", 0)

        chunks = _docs_to_chunks(final_state["retrieved_docs"])
        return RetrievalResult(
            query=query,
            chunks=chunks,
            answer=answer,
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            usage=usage,
        )

    def retrieve_only(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
        corpus: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return hybrid-retrieved chunks without calling the LLM."""
        k = top_k if top_k is not None else self._top_k
        ensemble = self._build_ensemble(k=k, namespace=namespace, filter=filter, corpus=corpus)
        docs = ensemble.invoke(query)
        return _docs_to_chunks(docs)

    # ------------------------------------------------------------------
    # LangGraph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Compile the retrieve → grade → generate LangGraph."""

        builder = StateGraph(RAGState)

        # ── Nodes ─────────────────────────────────────────────────────
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("grade", self._node_grade)
        builder.add_node("rewrite", self._node_rewrite)
        builder.add_node("generate", self._node_generate)

        # ── Edges ─────────────────────────────────────────────────────
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade")

        # After grading: relevant → generate, irrelevant → rewrite (or give up)
        builder.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"generate": "generate", "rewrite": "rewrite", "generate_anyway": "generate"},
        )

        builder.add_edge("rewrite", "retrieve")   # loop back
        builder.add_edge("generate", END)

        return builder.compile()

    # ── Node implementations ──────────────────────────────────────────

    def _node_retrieve(self, state: RAGState, config: RunnableConfig) -> dict:
        """Run the EnsembleRetriever and store results in state."""
        configurable = config.get("configurable") or {}
        ensemble: EnsembleRetriever = configurable["ensemble"]
        query = _last_user_message(state["messages"])
        docs = ensemble.invoke(query)
        return {"retrieved_docs": docs}

    def _node_grade(self, state: RAGState) -> dict:
        """Grade relevance of retrieved docs — result stored for routing."""
        # Grading is handled inside _route_after_grade; this node is a pass-through
        # that keeps the graph structure explicit and easy to extend.
        return {}

    def _node_rewrite(self, state: RAGState) -> dict:
        """Rewrite the user query to improve retrieval on the next attempt."""
        original_query = _last_user_message(state["messages"])
        rewritten = self._rewriter_llm.invoke(
            f"The following question did not retrieve useful documents.\n"
            f"Rewrite it to be more specific and retrieval-friendly.\n\n"
            f"Original question: {original_query}\n\n"
            f"Rewritten question (one sentence, no preamble):"
        )
        rewrite_count = state.get("rewrite_count", 0) + 1
        return {
            "messages": [{"role": "user", "content": rewritten.content.strip()}],
            "rewrite_count": rewrite_count,
        }

    def _node_generate(self, state: RAGState) -> dict:
        """Generate a grounded answer from retrieved documents."""
        question = _last_user_message(state["messages"])
        docs = state["retrieved_docs"]

        context_parts: list[str] = []
        for i, doc in enumerate(docs, start=1):
            label = doc.metadata.get("section_title") or doc.metadata.get("doc_id", f"chunk-{i}")
            context_parts.append(f"[{i}] {label}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No context retrieved."

        response = self._llm.invoke(
            [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n\n{context}\n\n"
                        f"---\n\n"
                        f"Question: {question}"
                    ),
                },
            ]
        )
        return {"messages": [response]}

    # ── Routing ──────────────────────────────────────────────────────

    def _route_after_grade(
        self, state: RAGState
    ) -> Literal["generate", "rewrite", "generate_anyway"]:
        """Decide whether to generate, rewrite, or give up after grading."""
        docs = state["retrieved_docs"]
        rewrite_count = state.get("rewrite_count", 0)

        if not docs:
            if rewrite_count >= self._max_rewrites:
                return "generate_anyway"
            return "rewrite"

        question = _last_user_message(state["messages"])
        context_snippet = " ".join(d.page_content[:300] for d in docs[:3])

        grade: RelevanceGrade = self._grader_llm.invoke(
            f"Question: {question}\n\n"
            f"Retrieved context (excerpt):\n{context_snippet}\n\n"
            f"Is this context relevant to answering the question?"
        )

        if grade.score == "yes":
            return "generate"
        if rewrite_count >= self._max_rewrites:
            return "generate_anyway"
        return "rewrite"

    # ------------------------------------------------------------------
    # Ensemble builder
    # ------------------------------------------------------------------

    def _build_ensemble(
        self,
        k: int,
        namespace: str | None,
        filter: dict[str, Any] | None,
        corpus: list[str] | None,
    ) -> EnsembleRetriever:
        """Build an EnsembleRetriever combining BM25 + dense Pinecone search.

        BM25 weight  = 1 - HYBRID_ALPHA  (keyword signal)
        Dense weight = HYBRID_ALPHA       (semantic signal)
        """
        # -- Dense retriever (Pinecone) ------------------------------------
        search_kwargs: dict[str, Any] = {"k": k}
        if namespace:
            search_kwargs["namespace"] = namespace
        if filter:
            search_kwargs["filter"] = filter

        dense_retriever = self._vector_store.as_retriever(search_kwargs=search_kwargs)

        # -- Sparse retriever (BM25) ---------------------------------------
        # Seed BM25 with the provided corpus, or fall back to dense results
        # for the query (so BM25 always has context to rank over).
        if corpus:
            bm25_docs = [Document(page_content=t) for t in corpus]
        else:
            # Bootstrap: use dense retriever to get seed documents
            seed_docs = dense_retriever.invoke("*")  # broad seed query
            bm25_docs = seed_docs if seed_docs else [Document(page_content="placeholder")]

        bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=k)

        # -- Ensemble with RRF fusion --------------------------------------
        bm25_weight = round(1.0 - self._alpha, 4)
        dense_weight = round(self._alpha, 4)

        return EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[bm25_weight, dense_weight],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_user_message(messages: list) -> str:
    """Extract the most recent user message content from the message list."""
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        raw = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
        if role in ("human", "user") and raw is not None:
            return str(raw)
    return ""


def _docs_to_chunks(docs: list[Document]) -> list[RetrievedChunk]:
    """Convert LangChain Documents to RetrievedChunk dataclasses."""
    chunks: list[RetrievedChunk] = []
    for doc in docs:
        meta = dict(doc.metadata)
        chunks.append(
            RetrievedChunk(
                doc_id=meta.pop("doc_id", "unknown"),
                score=float(meta.pop("score", 0.0)),
                text=doc.page_content,
                metadata=meta,
            )
        )
    return chunks
