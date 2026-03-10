"""
Hybrid RAG with Reciprocal Rank Fusion (RRF).

This script implements a hybrid RAG pipeline that combines lexical retrieval
(BM25) and semantic retrieval (ChromaDB), then fuses both ranked lists with
Reciprocal Rank Fusion (RRF).
"""

import os
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate

from src.common.model_provider import get_model_identity
from src.common.usage_metrics import extract_usage_from_ai_message, extract_cost_from_ai_message

# --- Environment and Path Configuration ---

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in the .env file")

# Define paths
chroma_db_dir = PROJECT_ROOT / "data" / "embeddings" / "chroma_db"
collection_name = "guia_embarazo_parto"
chunks_file = PROJECT_ROOT / "data" / "chunks" / "chunks_final.json"

# Retrieval settings
k_bm25_candidates = 15
k_semantic_candidates = 15
k_rrf_pool = 10
k_final = 5
rrf_k = 60
mmr_lambda = 0.7


def load_documents() -> List[Document]:
    """Loads chunks from JSON and converts them to LangChain Documents."""
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    return [Document(page_content=d["content"], metadata=d) for d in chunks_data]


documents = load_documents()

# --- Model and Retriever Configuration ---

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 1. Lexical retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = k_bm25_candidates

# 2. Semantic retriever (Chroma)
vectorstore = Chroma(
    persist_directory=str(chroma_db_dir),
    embedding_function=embeddings,
    collection_name=collection_name,
)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k_semantic_candidates})


# --- Prompt Template ---

qa_template = """
You are a medical expert specializing in pregnancy and childbirth.
Your task is to analyze the provided medical context and answer the user's question accurately and concisely.

STRICT INSTRUCTIONS:
1.  **Base your answer exclusively on the information within the MEDICAL CONTEXT section.** Do not use any external knowledge.
2.  *The context is ordered by relevance.* Give the highest priority to the first few documents (e.g., Documents 1-2) as they are the most relevant. Use subsequent documents to supplement your answer if needed.
3.  *Provide a direct and integrated answer.* Your response should be a single, well-written paragraph. Start with a direct answer to the question, then seamlessly incorporate specific details, data, and recommendations from the context to support it.
4.  *If the context does not contain enough information to answer the question, state that clearly.* Do not try to invent an answer.
5.  *remember always answer in spanish*

MEDICAL CONTEXT (ordered by relevance):
{context}

QUESTION: {question}

DETAILED MEDICAL ANSWER:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)


def _document_unique_id(doc: Document) -> str:
    """Builds a stable identifier for deduplication during rank fusion."""
    metadata = doc.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    page_number = metadata.get("page_number", "na")
    chunk_index = metadata.get("chunk_index", "na")
    return f"{page_number}:{chunk_index}:{hash(doc.page_content)}"


def reciprocal_rank_fusion(rankings: List[List[Document]], k_constant: int = 60, top_k: int = 5) -> List[Document]:
    """Fuses multiple ranked lists using RRF and returns top-k documents."""
    scores: Dict[str, float] = {}
    documents_by_id: Dict[str, Document] = {}

    for ranked_docs in rankings:
        for rank, doc in enumerate(ranked_docs, start=1):
            doc_id = _document_unique_id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank))
            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = doc

    sorted_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
    return [documents_by_id[doc_id] for doc_id in sorted_doc_ids[:top_k]]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two dense vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def mmr_select(query: str, candidate_docs: List[Document], top_k: int, lambda_mult: float = 0.7) -> List[Document]:
    """
    Selects top-k documents using MMR to balance relevance and diversity.
    """
    if not candidate_docs:
        return []
    if len(candidate_docs) <= top_k:
        return candidate_docs

    query_embedding = embeddings.embed_query(query)
    doc_embeddings = embeddings.embed_documents([doc.page_content for doc in candidate_docs])

    relevance_scores = [_cosine_similarity(query_embedding, doc_vec) for doc_vec in doc_embeddings]

    selected_indices: List[int] = []
    remaining_indices = list(range(len(candidate_docs)))

    # First pick: most relevant to query.
    first_idx = max(remaining_indices, key=lambda idx: relevance_scores[idx])
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    # Next picks: maximize MMR objective.
    while remaining_indices and len(selected_indices) < top_k:
        best_idx = None
        best_score = float("-inf")

        for idx in remaining_indices:
            max_similarity_to_selected = max(
                _cosine_similarity(doc_embeddings[idx], doc_embeddings[selected_idx])
                for selected_idx in selected_indices
            )
            mmr_score = (lambda_mult * relevance_scores[idx]) - ((1.0 - lambda_mult) * max_similarity_to_selected)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [candidate_docs[idx] for idx in selected_indices]


def retrieve_hybrid_rrf(query: str) -> List[Document]:
    """Retrieves candidates, fuses with RRF, then applies MMR diversification."""
    bm25_docs = bm25_retriever.invoke(query)
    semantic_docs = semantic_retriever.invoke(query)
    rrf_ranked_docs = reciprocal_rank_fusion(
        [bm25_docs, semantic_docs],
        k_constant=rrf_k,
        top_k=k_rrf_pool,
    )
    return mmr_select(query=query, candidate_docs=rrf_ranked_docs, top_k=k_final, lambda_mult=mmr_lambda)


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents for the answer prompt."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "N/A")
        page = doc.metadata.get("page_number", "N/A")
        formatted_doc = f"""--- Document {i + 1} ---
Source: {source}, Page: {page}
Content: {doc.page_content}"""
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


def process_hybrid_rrf_query(query: str, custom_llm: Optional[BaseChatModel] = None) -> Dict[str, Any]:
    """Processes a query with Hybrid RAG + RRF fusion."""
    retrieved_docs = retrieve_hybrid_rrf(query)
    formatted_context = format_docs(retrieved_docs)

    current_llm = custom_llm if custom_llm else llm
    response = current_llm.invoke(
        qa_prompt.format_messages(
            context=formatted_context,
            question=query,
        )
    )
    usage = extract_usage_from_ai_message(response)
    provider_cost = extract_cost_from_ai_message(response)

    return {
        "answer": response.content,
        "contexts": [doc.page_content for doc in retrieved_docs],
        "retrieved_documents": retrieved_docs,
        "metrics": {
            "input_tokens": int(usage["input_tokens"]),
            "output_tokens": int(usage["output_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
            "usage_source": str(usage["usage_source"]),
            "cost": float(provider_cost["total_cost"]) if provider_cost["total_cost"] is not None else 0.0,
            "cost_source": str(provider_cost["cost_source"]),
        },
    }


def query_for_evaluation(question: str, llm_model: str = None, custom_llm: Optional[BaseChatModel] = None) -> dict:
    """Wrapper function for RAGAS-compatible evaluation."""
    start_time = time.time()

    if custom_llm:
        result = process_hybrid_rrf_query(question, custom_llm)
        model_identity = get_model_identity(llm=custom_llm)
    elif llm_model:
        custom_llm_instance = ChatOpenAI(model_name=llm_model, temperature=0)
        result = process_hybrid_rrf_query(question, custom_llm_instance)
        model_identity = get_model_identity(model_name=llm_model, llm=custom_llm_instance)
    else:
        result = process_hybrid_rrf_query(question)
        model_identity = get_model_identity(model_name="gpt-4o", llm=llm)

    end_time = time.time()
    execution_time = end_time - start_time

    input_tokens = result["metrics"]["input_tokens"]
    output_tokens = result["metrics"]["output_tokens"]

    return {
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "source_documents": result["retrieved_documents"],
        "metadata": {
            "num_contexts": len(result["contexts"]),
            "retrieval_method": "hybrid_bm25_semantic_rrf_mmr",
            "rrf_k": rrf_k,
            "k_bm25_candidates": k_bm25_candidates,
            "k_semantic_candidates": k_semantic_candidates,
            "k_rrf_pool": k_rrf_pool,
            "k_final": k_final,
            "mmr_lambda": mmr_lambda,
            "llm_model": model_identity["model_name"],
            "provider": model_identity["provider"],
            "model_id": model_identity["model_id"],
            "embedding_model": "text-embedding-3-small",
            "execution_time": execution_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": result["metrics"]["cost"],
            "tokens_used": input_tokens + output_tokens,
            "usage_source": result["metrics"]["usage_source"],
            "cost_source": result["metrics"]["cost_source"],
        },
    }


if __name__ == "__main__":
    print("\n=== Hybrid RAG + RRF (BM25 + Semantic) ===")
    print("This system retrieves BM25 and semantic candidates, then fuses with RRF and diversifies with MMR.")
    print(f"Documents loaded for BM25: {len(documents)}")
    try:
        print(f"Vector store documents: {vectorstore._collection.count()}")
    except Exception as e:
        print(f"Could not retrieve vector store document count: {e}")
    print("\nType your question or 'exit' to finish.")

    while True:
        query = input("\nQuestion: ")
        if query.lower() == "exit":
            break

        start_time = time.time()
        result = process_hybrid_rrf_query(query)
        end_time = time.time()

        print("\n" + "=" * 50)
        print("ANSWER:")
        print(result["answer"])
        print("\n" + "=" * 50)

        print("\n DETAILED METRICS:")
        print(f"     Total time: {end_time - start_time:.2f} seconds")
        print(f"   - Input Tokens (prompt): {result['metrics']['input_tokens']}")
        print(f"   - Output Tokens (answer): {result['metrics']['output_tokens']}")
        print(f"   - Total Cost (USD): ${result['metrics']['cost']:.6f}")

    print("\nSystem finished.")