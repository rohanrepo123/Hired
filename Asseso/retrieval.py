from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import json
from pathlib import Path

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

CHROMA_PATH = "./chroma_db"

def _ensure_db_populated():
    """If Chroma DB directory is missing or empty, create it from catalog."""
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        print("Vector DB not found. Populating from catalog...")
        from DB import load_json, create_Vdatabase
        data = load_json()
        create_Vdatabase(data)
    else:
        print("Vector DB already exists.")

_ensure_db_populated()

vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)

def retrieve_data(query: str, k: int = 6):
    """
    Semantic retrieval from SHL catalog using cosine similarity.
    """
    if not query.strip():
        return []
    try:
        results = vector_db.similarity_search(query, k=k)
        output = []
        for doc in results:
            output.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })
        print(output)
        return output
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []
    
print(retrieve_data("Global Skills Development Report"))
