from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Load persistent Chroma DB
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


def retrieve_data(query: str, k: int = 4):
    """
    Semantic retrieval from SHL catalog.
    """

    results = vector_db.similarity_search(query, k=k)

    output = []

    for doc in results:
        output.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return output