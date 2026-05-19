# SHL Conversational Assessment Recommender

A production-style conversational AI agent built for the SHL AI Intern Assignment.

The system helps recruiters and hiring managers discover relevant SHL assessments through natural language conversations instead of manual keyword-based catalog search.

The agent supports:

* Conversational hiring requirement gathering
* Clarification-based interaction
* Semantic retrieval using vector embeddings
* Grounded SHL assessment recommendations
* Assessment comparison
* Recommendation refinement
* Stateless conversation handling
* Prompt-injection and off-topic refusal

---

# Features

* FastAPI backend
* Stateless conversational API
* Retrieval-Augmented Generation (RAG)
* ChromaDB vector database
* OpenAI embeddings
* Groq LLM integration
* SHL catalog grounding
* Structured JSON responses
* Context compression and summarization
* Recommendation validation against catalog
* Semantic retrieval pipeline
* Production-style agent workflow

---

# Architecture

```text
User Query
    ↓
FastAPI API Layer
    ↓
Conversation State Extraction
    ↓
Conversation Summarization
    ↓
Semantic Retrieval (ChromaDB)
    ↓
Retrieved SHL Catalog Context
    ↓
LLM Grounded Reasoning
    ↓
Recommendation Validation
    ↓
Structured JSON Response
```

---

# Tech Stack

| Component       | Technology                    |
| --------------- | ----------------------------- |
| Backend         | FastAPI                       |
| LLM             | Groq                          |
| Embeddings      | OpenAI text-embedding-3-small |
| Vector Database | ChromaDB                      |
| Framework       | LangChain                     |
| Deployment      | Render                        |
| Language        | Python 3.11                   |

---

# Project Structure

```text
.
├── main.py
├── recommender.py
├── retrieval.py
├── llm.py
├── DB.py
├── shl_product_catalog.json
├── chroma_db/
├── requirements.txt
├── README.md
└── APPROACH.md
```

---

# Original Dataset Source

The SHL product catalog was originally fetched from:

```text
https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json
```

During development, the original JSON file was found to contain a formatting/parsing issue around line `4795`, which caused failures during vector database generation and JSON loading.

To ensure reproducibility and stable retrieval:

* the corrupted JSON structure was manually fixed,
* the corrected version has been uploaded and committed into this repository as:

```text
shl_product_catalog.json
```

The recommendation pipeline and vector database generation use this corrected local catalog file as the source of truth.

---

# Installation & Setup

## Step 1 — Clone Repository

```bash
git clone https://github.com/rohanrepo123/Hired.git
cd Hired
```

---

## Step 2 — Create Virtual Environment

Windows:

```powershell
python -m venv venv311
```

Linux / Mac:

```bash
python3 -m venv venv311
```

---

## Step 3 — Activate Virtual Environment

Windows:

```powershell
venv311\Scripts\activate
```

Linux / Mac:

```bash
source venv311/bin/activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies used in this project include: 

* FastAPI
* LangChain
* ChromaDB
* OpenAI embeddings
* Groq
* Uvicorn
* LangSmith

---

## Step 5 — Create `.env` File

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=llama-3.1-8b-instant
```

---

## Step 6 — Build Vector Database

Run:

```bash
python DB.py
```

This creates the Chroma vector database using:

```text
shl_product_catalog.json
```

---

## Step 7 — Start FastAPI Server

```bash
uvicorn main:app --reload
```

Server starts at:

```text
http://127.0.0.1:8000
```

---

# API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Chat Endpoint

```http
POST /chat
```

Request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a senior Java developer with Spring Boot and SQL."
    }
  ]
}
```

Response:

```json
{
  "reply": "Here are suitable SHL assessments for the role.",
  "recommendations": [
    {
      "name": "Automata (New)",
      "url": "https://www.shl.com/products/product-catalog/view/automata-new/",
      "test_type": "S"
    }
  ],
  "end_of_conversation": false
}
```

---

# Stateless Architecture

The API is fully stateless.

Every `/chat` request contains the complete conversation history.

The backend stores:

* no user sessions,
* no Redis memory,
* no server-side chat state.

Instead, the system reconstructs conversational state dynamically from:

* previous messages,
* summarized hiring intent,
* retrieved catalog context.

This design aligns with scalable production conversational AI architecture.

---

# Retrieval Pipeline

The system uses:

* OpenAI embeddings
* Chroma cosine similarity search
* semantic retrieval
* catalog grounding

Recommendations are generated ONLY from retrieved SHL catalog entries.

The retrieval pipeline dynamically:

* analyzes hiring intent,
* retrieves semantically relevant assessments,
* compresses retrieved context,
* and passes grounded information to the LLM.

---

# Context Engineering

To avoid token overflow and large-context failures:

* conversation summarization was implemented,
* retrieved documents were truncated,
* recent-message prioritization was added,
* compact conversational memory was introduced.

This improved:

* latency,
* retrieval quality,
* token efficiency,
* conversational coherence.

---

# Safety Features

The agent:

* refuses prompt injection attempts
* rejects unrelated/off-topic queries
* prevents hallucinated recommendations
* validates URLs against SHL catalog
* limits recommendations to 1–10 items
* sanitizes structured outputs

---

# Deployment

The application is deployed on Render.

Environment variables are configured through:

* Render Environment Settings
* automatic redeployment workflow

---

# Assignment Alignment

This implementation follows the SHL assignment requirements:

* conversational recommendation flow
* clarification handling
* recommendation refinement
* grounded catalog retrieval
* stateless API design
* JSON schema compliance
* refusal handling
* context engineering
* agentic conversational workflow

The project was designed according to the assignment goals mentioned in the official SHL document:

* problem solving
* programming skills
* context engineering
* agent design 

---

# Important Notes

* The API is fully stateless.
* Every `/chat` request must include the complete conversation history.
* Recommendations are generated only from retrieved SHL catalog entries.
* Conversation summarization is used to maintain scalable conversational memory.
* Retrieved context is compressed before being passed to the LLM.
* Final outputs are validated before returning structured JSON responses.
* The uploaded corrected catalog file is used instead of the broken original dataset source.

---

# Author

Rohan
