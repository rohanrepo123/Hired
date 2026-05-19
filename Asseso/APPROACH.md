# Approach

## Problem Understanding

The SHL assignment required building a conversational AI agent capable of transforming vague hiring requirements into grounded SHL assessment recommendations.

The core challenge was not simple chatbot generation, but designing a production-style conversational retrieval system capable of:

* clarification,
* semantic retrieval,
* refinement,
* comparison,
* grounding,
* refusal,
* and stateless conversation handling.

---

# System Design

The architecture follows a Retrieval-Augmented Generation (RAG) pipeline with conversational state reconstruction.

```text
Conversation History
        ↓
Conversation State Extraction
        ↓
Conversation Summarization
        ↓
Semantic Retrieval
        ↓
Retrieved SHL Assessments
        ↓
LLM Grounded Reasoning
        ↓
Structured JSON Response
```

---

# Stateless Conversation Design

The API is fully stateless.

Every `/chat` request contains the entire conversation history.

The backend stores:

* no sessions,
* no Redis memory,
* no per-user context.

Instead, the system reconstructs conversational state dynamically from message history during every request.

This approach aligns with scalable production API architecture.

---

# Context Engineering

A major focus was controlling prompt size while preserving conversational context.

Initially, sending:

* full conversation history,
* full retrieved documents,
* large prompts

caused token overflow issues.

To solve this:

* conversation summarization was introduced,
* retrieved documents were truncated,
* retrieval count was reduced,
* recent-message prioritization was added.

The final architecture combines:

* summarized memory,
* latest messages,
* retrieved catalog chunks.

This significantly improved:

* latency,
* retrieval quality,
* token efficiency,
* response reliability.

---

# Retrieval Pipeline

The retrieval system uses:

* OpenAI embeddings (`text-embedding-3-small`)
* ChromaDB vector database
* cosine similarity search

The vector database is built from:

```text
shl_product_catalog.json
```

Each catalog entry stores:

* assessment name
* URL
* description
* job levels
* categories
* duration
* metadata

Retrieval is semantic rather than keyword-only, allowing the system to handle vague recruiter queries naturally.

---

# Recommendation Generation

Initially, the system used large deterministic rule-based recommendation logic.

This created:

* repetitive outputs,
* weak semantic reasoning,
* poor adaptability.

The architecture was redesigned so that:

* retrieval drives recommendations,
* the LLM reasons over retrieved catalog context,
* recommendations remain grounded in SHL catalog data.

The LLM now:

* asks clarification questions,
* compares assessments,
* refines recommendations,
* refuses off-topic queries,
* generates grounded structured responses.

---

# Hallucination Prevention

Hallucination prevention was a major design goal.

The system prevents hallucinations by:

* restricting recommendations to retrieved catalog entries,
* validating URLs against catalog data,
* rejecting non-SHL recommendations,
* sanitizing structured outputs,
* limiting recommendation counts.

---

# Conversation Intelligence

The agent dynamically decides whether to:

* clarify,
* retrieve,
* recommend,
* compare,
* or refuse.

Examples:

* vague role → clarification
* complete requirements → recommendation
* updated constraints → refinement
* unrelated request → refusal

This behavior creates a more realistic conversational recruiter workflow.

---

# Performance Optimizations

Several optimizations were added to satisfy evaluator constraints:

* Reduced retrieval chunk sizes
* Retrieval count limiting
* Context summarization
* Timeout handling
* Response validation
* Recommendation deduplication

These changes improved:

* response speed,
* API stability,
* evaluator reliability.

---

# Evaluation Strategy

The system was tested across:

* technical hiring,
* leadership hiring,
* graduate recruitment,
* finance,
* healthcare,
* customer support,
* software engineering,
* AI/ML hiring.

Testing focused on:

* clarification quality,
* retrieval grounding,
* recommendation relevance,
* hallucination prevention,
* stateless behavior,
* schema correctness.

---

# Challenges Faced

## 1. Deterministic Recommendations

Early versions relied heavily on hardcoded rules, causing repetitive outputs.

This was fixed by shifting recommendation control to the retrieval + LLM pipeline.

---

## 2. Token Overflow

Large prompts and retrieved documents exceeded model context limits.

This was solved through:

* summarization,
* chunk truncation,
* reduced retrieval size,
* compressed memory representation.

---

## 3. Stateless Context Reconstruction

Maintaining coherent conversations without server-side memory required reconstructing hiring intent dynamically from conversation history.

Conversation summarization and structured state extraction solved this challenge.

---

# Final Outcome

The final system behaves as:

* a conversational AI recruiter assistant,
* a grounded SHL recommendation engine,
* and a production-style stateless RAG application.

The implementation aligns closely with the assignment goals of:

* problem solving,
* context engineering,
* agent design,
* and production AI system thinking.
