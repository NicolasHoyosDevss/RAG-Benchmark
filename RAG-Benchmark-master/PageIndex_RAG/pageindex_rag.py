"""
PageIndex RAG - Retrieval-Augmented Generation using PageIndex retrieval API.

This module follows the same public contract used by the other RAG modules in the
repository, exposing `query_for_evaluation(...)` so it can be plugged into the
benchmark evaluator later.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pageindex import PageIndexAPIError, PageIndexClient


# --- Environment and Path Configuration ---

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

if not os.getenv("PAGEINDEX_API_KEY"):
    raise ValueError("PAGEINDEX_API_KEY not found in the .env file")

if not os.getenv("PAGEINDEX_DOC_ID"):
    raise ValueError(
        "PAGEINDEX_DOC_ID not found in the .env file. "
        "Set it to an indexed PageIndex document id (e.g. pi-xxxx)."
    )

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in the .env file")


# --- Clients and Models ---

pageindex_client = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)


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


# --- Core Helpers ---

def _wait_for_retrieval_completion(
    retrieval_id: str,
    timeout_seconds: int = 120,
    poll_interval_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Polls PageIndex retrieval endpoint until completion or timeout."""
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        retrieval = pageindex_client.get_retrieval(retrieval_id)
        status = retrieval.get("status", "unknown")

        if status in ("completed", "done", "success"):
            return retrieval
        if status in ("failed", "error"):
            raise RuntimeError(f"PageIndex retrieval failed: {retrieval}")

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timed out waiting for retrieval completion. retrieval_id={retrieval_id}"
    )


def _extract_contexts_from_retrieval(retrieval_result: Dict[str, Any]) -> List[str]:
    """
    Converts PageIndex retrieval payload into the List[str] context format expected by RAGAS.
    """
    contexts: List[str] = []

    for node in retrieval_result.get("retrieved_nodes", []):
        title = node.get("title", "Untitled")
        relevant_contents = node.get("relevant_contents", [])

        snippets: List[str] = []
        for group in relevant_contents:
            if not isinstance(group, list):
                continue
            for item in group:
                content = item.get("relevant_content") if isinstance(item, dict) else None
                if content:
                    snippets.append(str(content).strip())

        # Fallback in case a node has no relevant_contents entries.
        if not snippets:
            metadata = node.get("metadata", [])
            snippets.append(f"No detailed snippet available. metadata={metadata}")

        # Keep each context reasonably compact while preserving high-value evidence.
        top_snippets = snippets[:2]
        contexts.append(
            f"Title: {title}\n" + "\n\n".join(top_snippets)
        )

    return contexts


def _format_contexts(contexts: List[str]) -> str:
    """Formats contexts into a prompt-ready block."""
    formatted_docs = []
    for i, context in enumerate(contexts, start=1):
        formatted_docs.append(f"--- Document {i} ---\n{context}")
    return "\n\n".join(formatted_docs)


# --- Main RAG Functions ---

def process_pageindex_query(
    query: str,
    custom_llm: Optional[ChatOpenAI] = None,
    doc_id: Optional[str] = None,
    thinking: bool = False,
    timeout_seconds: int = 120,
    poll_interval_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Processes a query with PageIndex retrieval and OpenAI answer synthesis.

    Args:
        query: User question.
        custom_llm: Optional custom answer model.
        doc_id: Optional PageIndex document id. If None, uses PAGEINDEX_DOC_ID from .env.
        thinking: Whether to enable PageIndex deeper retrieval mode.
        timeout_seconds: Max wait time for retrieval completion.
        poll_interval_seconds: Poll interval for retrieval status.

    Returns:
        Dictionary with answer, contexts, retrieval payload and metrics.
    """
    effective_doc_id = doc_id or os.getenv("PAGEINDEX_DOC_ID")
    if not effective_doc_id:
        raise ValueError("Missing doc_id and PAGEINDEX_DOC_ID is not set")

    submit_response = pageindex_client.submit_query(
        doc_id=effective_doc_id,
        query=query,
        thinking=thinking,
    )
    retrieval_id = submit_response["retrieval_id"]

    retrieval_result = _wait_for_retrieval_completion(
        retrieval_id=retrieval_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    contexts = _extract_contexts_from_retrieval(retrieval_result)
    formatted_context = _format_contexts(contexts)

    current_llm = custom_llm if custom_llm else llm
    with get_openai_callback() as cb_answer:
        response = current_llm.invoke(
            qa_prompt.format_messages(context=formatted_context, question=query)
        )

    return {
        "answer": response.content,
        "contexts": contexts,
        "retrieved_nodes": retrieval_result.get("retrieved_nodes", []),
        "retrieval_result": retrieval_result,
        "metrics": {
            "input_tokens": cb_answer.prompt_tokens,
            "output_tokens": cb_answer.completion_tokens,
            "cost": cb_answer.total_cost,
            "retrieval_id": retrieval_id,
            "doc_id": effective_doc_id,
        },
    }


def query_for_evaluation(
    question: str,
    llm_model: Optional[str] = None,
    doc_id: Optional[str] = None,
    thinking: bool = False,
) -> Dict[str, Any]:
    """
    Wrapper function compatible with the benchmark evaluator contract.

    Args:
        question: The question to process.
        llm_model: Optional model name for answer generation.
        doc_id: Optional PageIndex document id to override PAGEINDEX_DOC_ID.
        thinking: Optional PageIndex retrieval mode.

    Returns:
        Dictionary with question, answer, contexts and metadata.
    """
    start_time = time.time()

    if llm_model:
        custom_llm = ChatOpenAI(model_name=llm_model, temperature=0)
        result = process_pageindex_query(
            question,
            custom_llm=custom_llm,
            doc_id=doc_id,
            thinking=thinking,
        )
        used_model = llm_model
    else:
        result = process_pageindex_query(question, doc_id=doc_id, thinking=thinking)
        used_model = "gpt-4o"

    execution_time = time.time() - start_time
    input_tokens = result["metrics"]["input_tokens"]
    output_tokens = result["metrics"]["output_tokens"]

    return {
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "source_documents": result["retrieved_nodes"],
        "metadata": {
            "num_contexts": len(result["contexts"]),
            "retrieval_method": "pageindex",
            "llm_model": used_model,
            "execution_time": execution_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": result["metrics"]["cost"],
            "tokens_used": input_tokens + output_tokens,
            "doc_id": result["metrics"]["doc_id"],
            "retrieval_id": result["metrics"]["retrieval_id"],
            "pageindex_thinking": thinking,
        },
    }


if __name__ == "__main__":
    print("\n=== PageIndex RAG ===")
    print("This system uses PageIndex retrieval and OpenAI answer synthesis.")
    print("Set PAGEINDEX_DOC_ID in .env before running interactive mode.")
    print("\nType your question or 'exit' to finish.")

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == "exit":
            break

        start_time = time.time()
        result = process_pageindex_query(query)
        end_time = time.time()

        print("\n" + "=" * 50)
        print("ANSWER:")
        print(result["answer"])
        print("\n" + "=" * 50)

        print("\nDETAILED METRICS:")
        print(f"  Total time: {end_time - start_time:.2f} seconds")
        print(f"  Input Tokens (prompt): {result['metrics']['input_tokens']}")
        print(f"  Output Tokens (answer): {result['metrics']['output_tokens']}")
        print(f"  Total Cost (USD): ${result['metrics']['cost']:.6f}")
        print(f"  Retrieval ID: {result['metrics']['retrieval_id']}")

    print("\nSystem finished.")
