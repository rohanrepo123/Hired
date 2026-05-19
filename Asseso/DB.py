import requests
from langchain_community.document_loaders import JSONLoader
import json

from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import re
from langchain_chroma import Chroma

def load_json():
    import requests
    import json

    with open('shl_product_catalog.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def create_Vdatabase(data):
    documents = []

    for item in data:
        text = f"""
        Name: {item.get('name', '')}

        Description:
        {item.get('description', '')}

        Job Levels:
        {', '.join(item.get('job_levels', []))}

        Languages:
        {', '.join(item.get('languages', []))}

        Categories:
        {', '.join(item.get('keys', []))}
        """

        documents.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "name": str(item.get("name", "")),
                    "entity_id": str(item.get("entity_id", "")),
                    "link": str(item.get("link", "")),
                    "job_levels": ", ".join(item.get("job_levels", [])),
                    "languages": ", ".join(item.get("languages", [])),
                    "keys": ", ".join(item.get("keys", [])),
                    "remote": str(item.get("remote", "")),
                    "adaptive": str(item.get("adaptive", "")),
                    "duration": str(item.get("duration", "")),
                    "status": str(item.get("status", "")),
                    "scraped_at": str(item.get("scraped_at", ""))
                }
            )
        )

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    print("Vector DB created successfully!")

if __name__ == "__main__":
    data = load_json()
    create_Vdatabase(data)