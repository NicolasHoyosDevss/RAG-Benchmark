"""
Embedding Creation Module

This module handles the creation and storage of text embeddings using OpenAI's embedding models
and ChromaDB for vector storage. It processes pre-chunked text data and generates embeddings
for efficient similarity search and retrieval.

Dependencies:
- langchain-openai: For OpenAI embeddings
- langchain-community: For Chroma vector store
- python-dotenv: For environment variable management
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Configuration
# Build absolute paths based on script location to avoid errors
SCRIPT_DIR = Path(__file__).resolve().parent
CHUNKS_FILE_PATH = SCRIPT_DIR.parent / "chunks" / "chunks_final.json"
DB_DIRECTORY = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME = "guia_embarazo_parto"


def load_chunks(file_path):
    """
    Load chunks from a JSON file.

    Args:
        file_path (Path): Path to the JSON file containing chunks.

    Returns:
        list or None: List of chunk dictionaries if successful, None if failed.
    """
    print(f"Loading chunks from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        print(f"Successfully loaded {len(chunks_data)} chunks.")
        return chunks_data
    except FileNotFoundError:
        print(f"Error: Chunks file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None


def create_and_store_embeddings(chunks_data):
    """
    Create embeddings using OpenAI and store them in ChromaDB.

    Args:
        chunks_data (list): List of chunk dictionaries with content and metadata.
    """
    if not chunks_data:
        print("Warning: No chunks to process.")
        return

    # Extract content and metadata from chunks
    contents = [chunk['content'] for chunk in chunks_data]
    metadatas = [
        {
            "page_number": chunk['page_number'],
            "chunk_index": chunk['chunk_index'],
            "section_number": chunk['section_number'],
            "section_title": chunk['section_title'],
            "source": chunk['source']
        } for chunk in chunks_data
    ]

    print("Initializing OpenAI embeddings model...")
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception as e:
        print(f"Error initializing OpenAI embeddings: {e}")
        print("Ensure your OPENAI_API_KEY is configured in the .env file")
        return

    print(f"Creating and storing vector database at: {DB_DIRECTORY}")
    print(f"Collection: {COLLECTION_NAME}")

    try:
        # Create Chroma database from documents
        # This will generate embeddings and save them to the specified directory
        db = Chroma.from_texts(
            texts=contents,
            embedding=embeddings_model,
            metadatas=metadatas,
            persist_directory=str(DB_DIRECTORY),
            collection_name=COLLECTION_NAME
        )

        print("\nEmbeddings created and stored successfully!")
        print(f"Total vectors in database: {db._collection.count()}")
        print(f"Database saved at: {DB_DIRECTORY.absolute()}")

    except Exception as e:
        print(f"Error creating or storing embeddings in ChromaDB: {e}")


def main():
    """
    Main function to execute the embedding creation process.
    """
    print("=== STARTING EMBEDDING CREATION PROCESS ===")

    # Load the chunks
    chunks = load_chunks(CHUNKS_FILE_PATH)

    # Create and store embeddings if chunks were loaded
    if chunks:
        create_and_store_embeddings(chunks)

    print("\n=== PROCESS COMPLETED ===")


if __name__ == "__main__":
    main()
