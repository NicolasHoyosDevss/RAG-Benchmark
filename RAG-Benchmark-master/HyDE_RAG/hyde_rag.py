"""
HyDE RAG - Hypothetical Document Embeddings for medical Q&A.

This script implements a RAG (Retrieval-Augmented Generation) pipeline using
the HyDE (Hypothetical Document Embeddings) strategy. It generates a hypothetical
document based on a user's query and then uses that document to perform
semantic search for relevant context. This approach can improve retrieval
accuracy by searching for a more detailed document rather than a short query.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
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
# Use a more creative model for HyDE document generation
llm_hyde = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
# Use a powerful model for final answer generation
llm_answer = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Load ChromaDB vector store
vectorstore = Chroma(
    persist_directory=str(chroma_db_dir),
    embedding_function=embeddings,
    collection_name=collection_name,
)

# Configure the semantic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# --- Prompt Templates ---

# Prompt for generating the hypothetical document
hyde_prompt_template = """
You are a medical expert writing a detailed section for a medical guide on pregnancy and childbirth.

Based on this question: {question}

Write a detailed and comprehensive medical document that would perfectly answer this question.
The document should include:
- Accurate medical information on the topic
- Relevant clinical details
- Appropriate medical recommendations
- Important considerations for maternal health
- Practical information and advice

Write the document as if it were part of an official medical guide on pregnancy and childbirth.
Be specific, detailed, and use appropriate medical terminology.

HYPOTHETICAL DOCUMENT:
"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)

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

DETAILED MEDICAL
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)


# --- Core Functions ---

def generate_hypothetical_document(query: str) -> Dict[str, Any]:
    """
    Generates a hypothetical document based on the user's query.

    Args:
        query (str): The user's question.

    Returns:
        Dict[str, Any]: A dictionary containing the generated document and token/cost metrics.
    """
    with get_openai_callback() as cb:
        response = (hyde_prompt | llm_hyde | StrOutputParser()
                    ).invoke({"question": query})

    hypothetical_doc = response.strip()

    return {
        'document': hypothetical_doc,
        'input_tokens': cb.prompt_tokens,
        'output_tokens': cb.completion_tokens,
        'cost': cb.total_cost
    }


def format_docs(docs: List[Any]) -> str:
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


def process_hyde_query(query: str, custom_hyde_llm: ChatOpenAI = None, custom_answer_llm: ChatOpenAI = None) -> Dict[str, Any]:
    """
    Processes a query using the full HyDE RAG pipeline.

    Args:
        query (str): The user's question.
        custom_hyde_llm (ChatOpenAI, optional): Custom model for hypothetical document generation.
        custom_answer_llm (ChatOpenAI, optional): Custom model for answer generation.

    Returns:
        Dict[str, Any]: A dictionary with the final answer, contexts, and detailed metrics.
    """
    # Use custom models if provided, else use defaults
    current_hyde_llm = custom_hyde_llm if custom_hyde_llm else llm_hyde
    current_answer_llm = custom_answer_llm if custom_answer_llm else llm_answer

    # 1. Generate hypothetical document using the appropriate model
    with get_openai_callback() as cb:
        response = (hyde_prompt | current_hyde_llm | StrOutputParser()
                    ).invoke({"question": query})
    hypothetical_doc = response.strip()
    hyde_result = {
        'document': hypothetical_doc,
        'input_tokens': cb.prompt_tokens,
        'output_tokens': cb.completion_tokens,
        'cost': cb.total_cost
    }

    # 2. Retrieve similar documents
    retrieved_docs = retriever.invoke(hypothetical_doc)

    # 3. Format context
    formatted_context = format_docs(retrieved_docs)

    # 4. Generate final answer using the appropriate model
    with get_openai_callback() as cb_answer:
        response = current_answer_llm.invoke(qa_prompt.format_messages(
            context=formatted_context,
            question=query
        ))

    # 5. Return response and all metrics
    return {
        'answer': response.content,
        'contexts': [doc.page_content for doc in retrieved_docs],
        'hypothetical_document': hypothetical_doc,
        'hyde_metrics': hyde_result,
        'answer_metrics': {
            'input_tokens': cb_answer.prompt_tokens,
            'output_tokens': cb_answer.completion_tokens,
            'cost': cb_answer.total_cost
        },
        'total_cost': hyde_result['cost'] + cb_answer.total_cost,
        'total_input_tokens': hyde_result['input_tokens'] + cb_answer.prompt_tokens,
        'total_output_tokens': hyde_result['output_tokens'] + cb_answer.completion_tokens
    }


def query_for_evaluation(question: str, hyde_model: str = None, answer_model: str = None) -> dict:
    """
    A wrapper function for RAG evaluation frameworks like Ragas.

    This function processes a question and returns a dictionary structured for
    easy integration with evaluation tools.

    Args:
        question (str): The question to process.
        hyde_model (str, optional): The name of the LLM model to use for HyDE generation.
        answer_model (str, optional): The name of the LLM model to use for answer generation.

    Returns:
        dict: A dictionary containing the question, answer, contexts, and metadata.
    """
    # Create custom LLMs if models are specified
    custom_hyde_llm = ChatOpenAI(model_name=hyde_model, temperature=0.7) if hyde_model else None
    custom_answer_llm = ChatOpenAI(model_name=answer_model, temperature=0) if answer_model else None
    
    # Track which models are being used
    used_hyde_model = hyde_model if hyde_model else "gpt-3.5-turbo"
    used_answer_model = answer_model if answer_model else "gpt-4o"
    
    start_time = time.time()
    result = process_hyde_query(question, custom_hyde_llm, custom_answer_llm)
    end_time = time.time()
    execution_time = end_time - start_time

    return {
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "metadata": {
            "execution_time": execution_time,
            "input_tokens": result["total_input_tokens"],
            "output_tokens": result["total_output_tokens"],
            "total_cost": result["total_cost"],
            "retrieval_method": "hyde",
            "llm_hyde_model": used_hyde_model,
            "llm_answer_model": used_answer_model,
            "hyde_cost": result["hyde_metrics"]["cost"],
            "answer_cost": result["answer_metrics"]["cost"]
        }
    }


# --- Main Execution Block ---

if __name__ == "__main__":
    print("\n=== RAG with HyDE (Hypothetical Document Embeddings) ===")
    print("This system generates a hypothetical document based on your question")
    print("and then searches for documents similar to that hypothetical document.")
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
        result = process_hyde_query(query)
        end_time = time.time()

        print("\n" + "="*50)
        print("HYPOTHETICAL DOCUMENT (Preview):")
        print(result['hypothetical_document'][:300] + "...")
        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print(result['answer'])
        print("\n" + "="*50)

        # Display detailed metrics
        print("\n DETAILED METRICS:")
        print(f"     Total time: {end_time - start_time:.2f} seconds")

        print("\n HYPOTHETICAL DOCUMENT GENERATION:")
        print(
            f"   - Input Tokens (prompt): {result['hyde_metrics']['input_tokens']}")
        print(
            f"   - Output Tokens (document): {result['hyde_metrics']['output_tokens']}")
        print(f"   - Cost: ${result['hyde_metrics']['cost']:.6f}")

        print("\n FINAL ANSWER GENERATION:")
        print(
            f"   - Input Tokens (prompt): {result['answer_metrics']['input_tokens']}")
        print(
            f"   - Output Tokens (answer): {result['answer_metrics']['output_tokens']}")
        print(f"   - Cost: ${result['answer_metrics']['cost']:.6f}")

        print("\n TOTALS:")
        print(f"   - Total Input Tokens: {result['total_input_tokens']}")
        print(f"   - Total Output Tokens: {result['total_output_tokens']}")
        print(f"   - Total Cost (USD): ${result['total_cost']:.6f}")

    print("\nSystem finished.")
