"""
RAG with Multi-Query Rewriting for enhanced retrieval.

This script implements a RAG pipeline that uses a multi-query rewriting
strategy to improve document retrieval. It generates several variations of the
user's question, retrieves documents for each variation, and then synthesizes
an answer based on the combined, re-ranked results.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from src.common.model_provider import get_model_identity
from src.common.usage_metrics import extract_usage_from_ai_message, extract_cost_from_ai_message
from src.common.pricing import resolve_total_cost

# --- Environment and Path Configuration ---

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in the .env file")

# --- ChromaDB Configuration ---
DB_DIRECTORY = PROJECT_ROOT / "data" / "embeddings" / "chroma_db"
COLLECTION_NAME = "guia_embarazo_parto"

# --- Model and Vector Store Configuration ---

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load ChromaDB vector store
try:
    vectorstore = Chroma(
        persist_directory=str(DB_DIRECTORY),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    vectorstore = None

# Configure the base retriever
base_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.05}
)

# --- Query Rewriting Configuration ---

REPHRASE_TEMPLATE_1 = """Rewrite this question to be a standalone, specific query about pregnancy and childbirth.

Original question: {question}

Instructions:
- Maintain the medical/obstetric context if relevant.
- Be specific and clear in medical terms.
- Focus on pregnancy, childbirth, prenatal care, or maternal health.
- Ensure the question is complete and self-contained.

Standalone question:"""

REPHRASE_TEMPLATE_2 = """Rephrase this question about pregnancy and childbirth using synonyms and alternative medical terms.

Original question: {question}

Instructions:
- Use precise medical terminology.
- Include synonyms and alternative terms.
- Maintain the meaning but change the wording.
- Focus on clinical and obstetric aspects.

Rephrased question:"""

REPHRASE_TEMPLATE_3 = """Expand this question to include related aspects and additional context about pregnancy and childbirth.

Base question: {question}

Instructions:
- Expand the question to include related aspects.
- Add context about complications, prevention, or care.
- Include possible variations or special cases.
- Keep the focus on maternal and perinatal health.

Expanded question:"""

REPHRASE_PROMPTS = [
    PromptTemplate.from_template(REPHRASE_TEMPLATE_1),
    PromptTemplate.from_template(REPHRASE_TEMPLATE_2),
    PromptTemplate.from_template(REPHRASE_TEMPLATE_3)
]

llm_rewriter = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
llm_answer = ChatOpenAI(model_name="gpt-4o", temperature=0)


# --- Final Answer Prompt ---

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

DETAILED MEDICAL
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)


# --- Core Functions ---

def format_docs(docs: List[Document]) -> str:
    """Formats documents for the context, indicating relevance."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A')
        relevance = 'High' if i < 2 else 'Medium' if i < 4 else 'Low'
        formatted_doc = f"--- DOCUMENT {i+1} (Relevance: {relevance}) ---\n"
        formatted_doc += f"Source: {source}\n"
        formatted_doc += f"Content: {doc.page_content}"
        formatted_docs.append(formatted_doc)
    return "\n\n".join(formatted_docs)


def _invoke_text_with_usage(current_llm: BaseChatModel, messages: list) -> Tuple[str, Dict[str, Any]]:
    """Invoke a model and return plain text with normalized usage/cost metrics."""
    response = current_llm.invoke(messages)
    usage = extract_usage_from_ai_message(response)
    provider_cost = extract_cost_from_ai_message(response)

    return str(response.content).strip(), {
        "input_tokens": int(usage["input_tokens"]),
        "output_tokens": int(usage["output_tokens"]),
        "total_tokens": int(usage["total_tokens"]),
        "usage_source": str(usage["usage_source"]),
        "cost": float(provider_cost["total_cost"]) if provider_cost["total_cost"] is not None else 0.0,
        "cost_source": str(provider_cost["cost_source"]),
    }


def process_rewriter_query(question: str, custom_rewriter_llm: ChatOpenAI = None, custom_answer_llm: ChatOpenAI = None, max_final_docs: int = 8) -> Dict[str, Any]:
    """
    Processes a query using the multi-query rewriting RAG pipeline.

    Args:
        question (str): The user's question.
        custom_rewriter_llm (ChatOpenAI, optional): Custom model for query rewriting.
        custom_answer_llm (ChatOpenAI, optional): Custom model for answer generation.
        max_final_docs (int): The maximum number of documents to return.

    Returns:
        Dict[str, Any]: A dictionary with the final answer, contexts, and detailed metrics.
    """
    # Use custom models if provided, else use defaults
    current_rewriter_llm = custom_rewriter_llm if custom_rewriter_llm else llm_rewriter
    current_answer_llm = custom_answer_llm if custom_answer_llm else llm_answer
    
    # 1. Generate rewritten queries and track metrics
    rewritten_queries = []
    rewrite_input_tokens, rewrite_output_tokens, rewrite_cost = 0, 0, 0

    for prompt in REPHRASE_PROMPTS:
        rewritten_query, rewrite_metrics = _invoke_text_with_usage(
            current_rewriter_llm,
            prompt.format(question=question)
        )
        rewritten_queries.append(rewritten_query)
        rewrite_input_tokens += rewrite_metrics["input_tokens"]
        rewrite_output_tokens += rewrite_metrics["output_tokens"]
        rewrite_cost += rewrite_metrics["cost"]

    # 2. Retrieve documents for each rewritten query
    all_docs_with_scores = []
    doc_ids_seen = set()

    for i, query in enumerate(rewritten_queries, 1):
        results = vectorstore.similarity_search_with_score(query, k=5)
        for doc, distance in results:
            similarity = max(0.0, 1.0 - distance)
            # Use a slice of content as a unique ID
            doc_id = doc.page_content[:100]
            if doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)
                # Penalize queries from later, more speculative prompts
                query_weight = 1.0 - (i - 1) * 0.05
                all_docs_with_scores.append((doc, similarity * query_weight))

    # 3. Re-rank and select the best documents
    all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    retrieved_docs = [doc for doc, _ in all_docs_with_scores[:max_final_docs]]

    # 4. Format context and generate final answer
    formatted_context = format_docs(retrieved_docs)
    answer, answer_metrics = _invoke_text_with_usage(
        current_answer_llm,
        qa_prompt.format_messages(context=formatted_context, question=question)
    )

    # 5. Consolidate and return all information
    total_input = rewrite_input_tokens + answer_metrics["input_tokens"]
    total_output = rewrite_output_tokens + answer_metrics["output_tokens"]
    total_cost = rewrite_cost + answer_metrics["cost"]

    return {
        'answer': answer,
        'contexts': [doc.page_content for doc in retrieved_docs],
        'retrieved_documents': retrieved_docs,
        'rewritten_queries': rewritten_queries,
        'metrics': {
            'rewrite_input_tokens': rewrite_input_tokens,
            'rewrite_output_tokens': rewrite_output_tokens,
            'rewrite_cost': rewrite_cost,
            'answer_input_tokens': answer_metrics['input_tokens'],
            'answer_output_tokens': answer_metrics['output_tokens'],
            'answer_cost': answer_metrics['cost'],
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_cost': total_cost,
            'usage_source': answer_metrics['usage_source'],
            'cost_source': answer_metrics['cost_source'],
        }
    }


def query_for_evaluation(question: str, rewriter_model: str = None, answer_model: str = None, custom_rewriter_llm: Optional[BaseChatModel] = None, custom_answer_llm: Optional[BaseChatModel] = None) -> dict:
    """
    A wrapper function for RAG evaluation frameworks like Ragas.

    This function processes a question and returns a dictionary structured for
    easy integration with evaluation tools, preserving the original output format.

    Args:
        question (str): The question to process.
        rewriter_model (str, optional): The name of the LLM model to use for query rewriting.
        answer_model (str, optional): The name of the LLM model to use for answer generation.
        custom_rewriter_llm (BaseChatModel, optional): Pre-configured LLM for rewriting. Takes precedence over rewriter_model.
        custom_answer_llm (BaseChatModel, optional): Pre-configured LLM for answer. Takes precedence over answer_model.
    """
    # Create custom LLMs if provided, or fall back to string model names or defaults
    if custom_rewriter_llm:
        final_rewriter_llm = custom_rewriter_llm
        rewriter_model_identity = get_model_identity(llm=custom_rewriter_llm)
    elif rewriter_model:
        final_rewriter_llm = ChatOpenAI(model_name=rewriter_model, temperature=0.3)
        rewriter_model_identity = get_model_identity(model_name=rewriter_model, llm=final_rewriter_llm)
    else:
        final_rewriter_llm = None
        rewriter_model_identity = get_model_identity(model_name="gpt-3.5-turbo", llm=llm_rewriter)
    
    if custom_answer_llm:
        final_answer_llm = custom_answer_llm
        answer_model_identity = get_model_identity(llm=custom_answer_llm)
    elif answer_model:
        final_answer_llm = ChatOpenAI(model_name=answer_model, temperature=0)
        answer_model_identity = get_model_identity(model_name=answer_model, llm=final_answer_llm)
    else:
        final_answer_llm = None
        answer_model_identity = get_model_identity(model_name="gpt-4o", llm=llm_answer)
    
    start_time = time.time()
    result = process_rewriter_query(question, final_rewriter_llm, final_answer_llm)
    end_time = time.time()
    execution_time = end_time - start_time

    total_input = result['metrics']['total_input_tokens']
    total_output = result['metrics']['total_output_tokens']
    resolved_cost = resolve_total_cost(
        provider=answer_model_identity["provider"],
        model_name=answer_model_identity["model_name"],
        model_id=answer_model_identity["model_id"],
        input_tokens=total_input,
        output_tokens=total_output,
        provider_reported_cost=result['metrics']['total_cost'],
        provider_cost_source=result['metrics']['cost_source'],
        execution_time_seconds=execution_time,
    )

    return {
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "source_documents": result["retrieved_documents"],
        "metadata": {
            "num_contexts": len(result["contexts"]),
            "retrieval_method": "multi_query_rewrite",
            "rewrite_count": len(REPHRASE_PROMPTS),
            "llm_model": answer_model_identity["model_name"],
            "rewriter_model": rewriter_model_identity["model_name"],
            "provider": answer_model_identity["provider"],
            "model_id": answer_model_identity["model_id"],
            "rewriter_provider": rewriter_model_identity["provider"],
            "rewriter_model_id": rewriter_model_identity["model_id"],
            "execution_time": execution_time,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_cost": resolved_cost["total_cost"],
            "tokens_used": total_input + total_output,
            "usage_source": result['metrics']['usage_source'],
            "cost_source": resolved_cost["cost_source"],
        }
    }


# --- Main Execution Block ---

if __name__ == "__main__":
    print("\n=== RAG with Multi-Query Rewriter ===")
    print("This system rewrites your question in multiple ways to improve document retrieval.")
    if vectorstore:
        print(
            f"Connected to ChromaDB with {vectorstore._collection.count()} documents.")
    print("\nType your question or 'exit' to finish.")

    while True:
        query = input("\nQuestion: ")
        if query.lower() == "exit":
            break

        start_time = time.time()
        result = process_rewriter_query(query)
        end_time = time.time()

        print("\n" + "="*50)
        print("REWRITTEN QUERIES:")
        for i, rq in enumerate(result['rewritten_queries']):
            print(f"  {i+1}. {rq}")

        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print(result['answer'])
        print("\n" + "="*50)

        # Display detailed metrics
        print("\n DETAILED METRICS:")
        print(f"     Total time: {end_time - start_time:.2f} seconds")

        print("\n QUERY REWRITING:")
        print(
            f"   - Input Tokens: {result['metrics']['rewrite_input_tokens']}")
        print(
            f"   - Output Tokens: {result['metrics']['rewrite_output_tokens']}")
        print(f"   - Cost: ${result['metrics']['rewrite_cost']:.6f}")

        print("\n FINAL ANSWER GENERATION:")
        print(f"   - Input Tokens: {result['metrics']['answer_input_tokens']}")
        print(
            f"   - Output Tokens: {result['metrics']['answer_output_tokens']}")
        print(f"   - Cost: ${result['metrics']['answer_cost']:.6f}")

        print("\n TOTALS:")
        print(
            f"   - Total Input Tokens: {result['metrics']['total_input_tokens']}")
        print(
            f"   - Total Output Tokens: {result['metrics']['total_output_tokens']}")
        print(f"   - Total Cost (USD): ${result['metrics']['total_cost']:.6f}")

    print("\nSystem finished.")
