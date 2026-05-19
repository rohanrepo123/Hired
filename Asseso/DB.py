import requests
from langchain_community.document_loaders import JSONLoader
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# LangChain document structure
from langchain_core.documents import Document

# Embedding models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

import re

# Chroma vector database
from langchain_chroma import Chroma


# Function to load the corrected SHL product catalog JSON
def load_json():

    import requests
    import json

    # Open and load local corrected JSON file
    with open('shl_product_catalog.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


# Function to create vector database from SHL catalog
def create_Vdatabase(data):

    # List to store LangChain documents
    documents = []

    # Iterate through each assessment item
    for item in data:

        # Convert assessment data into structured text
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

        # Create LangChain document object
        documents.append(
            Document(
                page_content=text.strip(),

                # Metadata stored separately for retrieval/reference
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

    # Initialize OpenAI embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Create and persist Chroma vector database
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    print("Vector DB created successfully!")


# Main execution block
if __name__ == "__main__":

    # Load SHL catalog JSON
    data = load_json()

    # Create vector database
    create_Vdatabase(data)
