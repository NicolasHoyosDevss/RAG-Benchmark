"""
Simple Semantic RAG - A basic RAG pipeline using semantic search with ChromaDB.

This script implements a simple RAG (Retrieval-Augmented Generation) pipeline.
It uses a semantic retriever to find relevant documents in a ChromaDB vector
store and then uses a language model to generate an answer based on the
retrieved context.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Environment and Path Configuration ---

# Load environment variables from .env file
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in the .env file")

# Define paths
script_dir = Path(__file__).resolve().parent
chroma_db_dir = script_dir.parent / "Data" / "embeddings" / "chroma_db"
collection_name = "guia_embarazo_parto"

# --- Model and Vector Store Configuration ---

# Configure OpenAI models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Load ChromaDB vector store
vectorstore = Chroma(
    persist_directory=str(chroma_db_dir),
    embedding_function=embeddings,
    collection_name=collection_name,
)

# Configure the semantic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# --- Prompt Templates ---

# Prompt for generating the final answer
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


# --- Core Functions ---

def format_docs(docs: List[Document]) -> str:
    """
    Formats the retrieved documents to be included in the final prompt.

    Args:
        docs (list): A list of retrieved LangChain Document objects.

    Returns:
        str: A formatted string containing the content of the documents.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page_number', 'N/A')

        formatted_doc = f"""--- Document {i+1} ---
Source: {source}, Page: {page}
Content: {doc.page_content}"""
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs)


def process_semantic_query(query: str, custom_llm: ChatOpenAI = None) -> Dict[str, Any]:
    """
    Processes a query using the simple semantic RAG pipeline.

    Args:
        query (str): The user's question.
        custom_llm (ChatOpenAI, optional): Custom LLM to use. If None, uses default llm.

    Returns:
        Dict[str, Any]: A dictionary with the final answer, contexts, and detailed metrics.
    """
    # 1. Retrieve similar documents
    retrieved_docs = retriever.invoke(query)

    # 2. Format context
    formatted_context = format_docs(retrieved_docs)

    # 3. Generate final answer using custom LLM if provided, otherwise use default
    current_llm = custom_llm if custom_llm else llm
    
    with get_openai_callback() as cb_answer:
        response = current_llm.invoke(qa_prompt.format_messages(
            context=formatted_context,
            question=query
        ))

    # 4. Return response and all metrics
    return {
        'answer': response.content,
        'contexts': [doc.page_content for doc in retrieved_docs],
        'retrieved_documents': retrieved_docs,
        'metrics': {
            'input_tokens': cb_answer.prompt_tokens,
            'output_tokens': cb_answer.completion_tokens,
            'cost': cb_answer.total_cost
        }
    }


def query_for_evaluation(question: str, llm_model: str = None) -> dict:
    """
    A wrapper function for RAG evaluation frameworks like Ragas.

    This function processes a question and returns a dictionary structured for
    easy integration with evaluation tools. The output structure is preserved
    to match the original implementation for consistency.

    Args:
        question (str): The question to process.
        llm_model (str, optional): Model to use. If None, uses default "gpt-4o".

    Returns:
        dict: A dictionary containing the question, answer, contexts, and metadata.
    """
    start_time = time.time()
    
    # Create custom LLM if model is specified, otherwise use default behavior
    if llm_model:
        custom_llm = ChatOpenAI(model_name=llm_model, temperature=0)
        result = process_semantic_query(question, custom_llm)
        used_model = llm_model
    else:
        result = process_semantic_query(question)
        used_model = "gpt-4o"  # Default model
    
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
            "retrieval_method": "semantic_only",
            "llm_model": used_model,
            "embedding_model": "text-embedding-3-small",
            "execution_time": execution_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": result["metrics"]["cost"],
            "tokens_used": input_tokens + output_tokens,
        }
    }


# --- Main Execution Block ---

if __name__ == "__main__":
    print("\n=== Simple Semantic RAG ===")
    print("This system uses semantic search to retrieve relevant documents and generate an answer.")
    try:
        print(f"Documents in the database: {vectorstore._collection.count()}")
    except Exception as e:
        print(f"Could not retrieve document count: {e}")
    print("\nType your question or 'exit' to finish.")

    while True:
        query = input("\nQuestion: ")
        if query.lower() == "exit":
            break

        start_time = time.time()
        result = process_semantic_query(query)
        end_time = time.time()

        print("\n" + "="*50)
        print("ANSWER:")
        print(result['answer'])
        print("\n" + "="*50)

        # Display detailed metrics
        print("\n DETAILED METRICS:")
        print(f"     Total time: {end_time - start_time:.2f} seconds")
        print(
            f"   - Input Tokens (prompt): {result['metrics']['input_tokens']}")
        print(
            f"   - Output Tokens (answer): {result['metrics']['output_tokens']}")
        print(f"   - Total Cost (USD): ${result['metrics']['cost']:.6f}")

    print("\nSystem finished.")
