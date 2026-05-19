# SHL Assessment Recommender

FastAPI service for the SHL AI Intern assignment. The service is fully stateless and exposes the required endpoints:

* `GET /health`
* `POST /chat`

The project implements a conversational RAG-based SHL assessment recommendation agent using:

* FastAPI
* ChromaDB
* OpenAI embeddings
* Groq LLM
* LangChain

`DB.py` is intentionally left untouched. It remains the manual/vector database creation script for the SHL catalog.

---

# Live Deployment

Public API Endpoint:

```text id="cnjlwm"
https://get-assesement-for-hrs.onrender.com/
```

Health endpoint:

```text id="ny6kof"
https://get-assesement-for-hrs.onrender.com/health
```

Chat endpoint:

```text id="3ui0c0"
https://get-assesement-for-hrs.onrender.com/chat
```

---

# GitHub Repository

Repository:

```text id="8is5h7"
https://github.com/rohanrepo123/Hired.git
```

---

# Dataset Source

The SHL product catalog was originally fetched from:

```text id="y0b0dq"
https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json
```

The originally provided JSON file contained a formatting/parsing issue around line `4795`, which caused failures during JSON loading and vector database creation.

To ensure reproducibility and stable retrieval:

* the corrupted JSON structure was manually fixed,
* the corrected catalog file has been uploaded into this repository as:

```text id="5u5v6z"
shl_product_catalog.json
```

The recommendation pipeline and Chroma vector database generation use this corrected local catalog file as the source of truth.

---

# How To Use

## Step 1 — Clone Repository

```powershell id="2yl7qw"
git clone https://github.com/rohanrepo123/Hired.git
cd Hired
```

---

## Step 2 — Create Python Virtual Environment

Windows:

```powershell id="q2if49"
python -m venv venv311
```

Linux / Mac:

```bash id="djlwm2"
python3 -m venv venv311
```

---

## Step 3 — Activate Virtual Environment

Windows:

```powershell id="b2u56h"
venv311\Scripts\activate
```

Linux / Mac:

```bash id="5u3n4r"
source venv311/bin/activate
```

---

## Step 4 — Install Dependencies

```powershell id="9r0uxj"
pip install -r requirements.txt
```

---

## Step 5 — Create `.env` File

Create a local `.env` file in the project root:

```env id="g5lgd4"
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=llama-3.1-8b-instant
```

---

## Step 6 — Build Chroma Vector Database

After dependencies are installed, build or refresh the local Chroma vector database:

```powershell id="2wkq0z"
python DB.py
```

This creates the vector database using:

```text id="v4s7wx"
shl_product_catalog.json
```

---

## Step 7 — Start FastAPI Server

```powershell id="34nlf7"
uvicorn main:app --reload
```

Server starts at:

```text id="ws9c5w"
http://127.0.0.1:8000
```

---

# API

## Health Endpoint

```powershell id="jlwm7n"
curl https://get-assesement-for-hrs.onrender.com/health
```

Expected response:

```json id="awp10q"
{
  "status": "ok"
}
```

---

## Chat Endpoint

```powershell id="9xg8t5"
curl -X POST https://get-assesement-for-hrs.onrender.com/chat `
  -H "Content-Type: application/json" `
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hiring a senior Java developer with Spring and SQL\"}]}"
```

---

# Response Shape

```json id="q7y4u8"
{
  "reply": "Based on the retrieved SHL catalog context, here are suitable assessments for the role.",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

---

# Architecture

```text id="ry0v1j"
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
LLM Grounded Reasoning (Groq)
    ↓
Recommendation Validation
    ↓
Structured JSON Response
```

---

# Notes

* `/chat` expects the complete conversation history on every request.
* The service stores no per-conversation state.
* The API is fully stateless.
* Recommendations are generated only from retrieved SHL catalog entries.
* Retrieved context is compressed and summarized before being passed to the LLM.
* Conversation summarization is used to reduce token overflow and improve context handling.
* Refusals and clarification turns return an empty `recommendations` list.
* Recommendations are validated against the corrected `shl_product_catalog.json` included in this repository.
* The system uses semantic retrieval with OpenAI embeddings and ChromaDB.
* The LLM/RAG pipeline is wrapped with timeout handling to avoid evaluator timeout failures.
* Prompt injection and off-topic requests are explicitly refused.

---

# Assignment Alignment

This implementation follows the SHL assignment requirements:

* conversational recommendation flow
* clarification handling
* recommendation refinement
* grounded catalog retrieval
* stateless API design
* schema compliance
* hallucination prevention
* refusal handling
* context engineering
* agentic conversational workflow

---

# Author

Rohan
