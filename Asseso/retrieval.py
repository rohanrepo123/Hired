from langchain_chroma import Chroma

# OpenAI embedding model
from langchain_openai import OpenAIEmbeddings

# Load environment variables
from dotenv import load_dotenv

import os
import json
from pathlib import Path

# Load .env variables
load_dotenv()


# -----------------------------
# Embedding Model
# -----------------------------

# OpenAI embedding model used for semantic retrieval
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# -----------------------------
# Chroma Configuration
# -----------------------------

# Local Chroma vector database path
CHROMA_PATH = "./chroma_db"


# -----------------------------
# Auto DB Initialization
# -----------------------------

def _ensure_db_populated():

    """
    Ensure Chroma vector database exists.

    If database folder is missing or empty,
    automatically rebuild vector database
    from corrected SHL catalog.
    """

    # Check whether vector DB exists
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):

        print("Vector DB not found. Populating from catalog...")

        # Import DB creation utilities
        from DB import load_json, create_Vdatabase

        # Load corrected SHL catalog
        data = load_json()

        # Create vector database
        create_Vdatabase(data)

    else:
        print("Vector DB already exists.")


# Automatically initialize DB at startup
_ensure_db_populated()


# -----------------------------
# Load Chroma Vector DB
# -----------------------------

# Connect to persistent Chroma vector store
vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)


# -----------------------------
# Retrieval Function
# -----------------------------

def retrieve_data(query: str, k: int = 6):

    """
    Semantic retrieval from SHL catalog
    using cosine similarity search.

    Parameters:
    ----------
    query : str
        User hiring query / search text

    k : int
        Number of retrieved documents

    Returns:
    -------
    list[dict]
        Retrieved SHL catalog entries
    """

    # Prevent empty retrieval queries
    if not query.strip():
        return []

    try:

        # Perform semantic similarity search
        results = vector_db.similarity_search(
            query,
            k=k
        )

        output = []

        # Convert LangChain documents into serializable dictionaries
        for doc in results:

            output.append({

                # Retrieved assessment text
                "page_content": doc.page_content,

                # SHL metadata
                "metadata": doc.metadata,
            })

        # Debug logging
        print(output)

        return output

    except Exception as e:

        # Graceful retrieval failure handling
        print(f"Retrieval error: {e}")

        return []


# -----------------------------
# Local Retrieval Test
# -----------------------------

# Test semantic retrieval locally
print(
    retrieve_data(
        "Global Skills Development Report"
    )
)
