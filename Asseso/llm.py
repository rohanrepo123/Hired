import json
import os
import re

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Message type used for tool responses in LangChain
from langchain_core.messages import ToolMessage

# Tool decorator for defining LangChain tools
from langchain_core.tools import tool

# Groq LLM support (currently commented out)
from langchain_groq import ChatGroq

# OpenAI chat model
from langchain_openai import ChatOpenAI


# -----------------------------
# LLM Configuration
# -----------------------------

# MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

# Default OpenAI model
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Timeout configuration for LLM requests
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

# Initialize OpenAI chat model
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.35,
    max_retries=1,
    request_timeout=LLM_TIMEOUT_SECONDS,
)

# Previous Groq setup kept for reference
# llm = ChatGroq(
#     model=MODEL_NAME,
#     temperature=0.35,
#     max_retries=1,
#     request_timeout=LLM_TIMEOUT_SECONDS,
# )


# -----------------------------
# Prompt Templates
# -----------------------------

# Main system prompt for tool-calling conversational agent
TOOL_AGENT_PROMPT = """
You are an SHL assessment recommendation agent with tools.

Return only valid JSON with this exact shape:
{
  "reply": "assistant message",
  "recommendations": [
    {"name": "exact catalog product name", "url": "exact catalog URL", "test_type": "K"}
  ],
  "end_of_conversation": false
}

Tool policy:
- First call `summarize_hiring_intent` with the full conversation.
- If `ready` is true (role + level + abilities known), **immediately call `retrieve_assessments`**.
- If `ready` is false, ask the single `missing_question` – never retrieve.
- Never recommend assessments without knowing both job level and at least one key ability.
- After retrieval, present a markdown table with recommendations.
- Never say "couldn't find exact matches" – if results are few, present what exists and offer to refine abilities.
""".strip()

# Prompt used to summarize hiring intent and detect missing information
SUMMARY_PROMPT = """
You decide whether a hiring conversation has enough information to retrieve SHL assessments.

Return only valid JSON:
{
  "ready": true,
  "search_query": "short retrieval query, max 18 words",
  "role_summary": "one-sentence hiring brief",
  "missing_question": "one targeted question if not ready"
}

Rules:
- Mark ready=true ONLY when ALL THREE are known from the conversation:
  1. role name (e.g., developer, manager, CXO, director)
  2. job level (e.g., entry, junior, mid, senior, executive)
  3. key abilities or skills (e.g., Java, leadership, strategic thinking, Python)
- As soon as role, level, and abilities are present, prefer retrieval.
- search_query must combine role, level, and top ability (e.g., "senior executive leadership strategic thinking").
- missing_question must be empty when ready=true.
- If not ready, ask only for the single missing field among role, level, or abilities.
- Never retrieve if abilities are missing.
""".strip()

# Prompt used to detect whether the user is satisfied with recommendations
SATISFACTION_PROMPT = """
You check whether the client is satisfied with the current SHL assessment shortlist.

Return only valid JSON:
{
  "satisfied": false,
  "reply": "brief closing reply when satisfied, else empty"
}

Rules:
- Mark satisfied=true only when the latest user message clearly accepts, confirms, thanks, or closes the discussion.
- Do not mark satisfied=true when the user is asking to see results, refine the shortlist, compare options, or add constraints.
- Prefer false when the message is ambiguous.
- Keep reply short.
""".strip()


# -----------------------------
# Helper Constants
# -----------------------------

# Mapping SHL category names to short test type codes
KEY_CODES = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


# -----------------------------
# Helper Functions
# -----------------------------

# Convert conversation messages into plain text
def _messages_to_text(messages: list[dict]) -> str:

    lines = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


# Get latest user message from conversation history
def _latest_user_text(messages: list[dict]) -> str:

    for msg in reversed(messages):

        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"].strip()

    return ""


# Extract valid JSON from model response text
def _json_from_text(text: str) -> dict | None:

    text = text.strip()

    # Remove markdown JSON fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try direct JSON parsing
    try:
        return json.loads(text)

    except:

        # Fallback regex extraction
        match = re.search(r"\{.*\}", text, flags=re.S)

        if match:
            try:
                return json.loads(match.group(0))
            except:
                return None

    return None


# Convert SHL category string into short type code
def _type_code(keys_str: str) -> str:

    codes = []

    for key in keys_str.split(","):

        code = KEY_CODES.get(key.strip())

        if code and code not in codes:
            codes.append(code)

    return ",".join(codes) or "K"


# Build markdown table for recommendation display
def _build_markdown_table(recs: list, role_summary: str) -> str:

    if not recs:
        return ""

    lines = [
        f"For {role_summary or 'your role'}:",
        "",
        "| # | Name | Test Type | Keys | Duration | Languages | URL |",
        "|---|------|-----------|------|----------|-----------|-----|"
    ]

    for idx, r in enumerate(recs, 1):

        lines.append(
            f"| {idx} | {r['name']} | {r['test_type']} | - | - | - | <{r['url']}> |"
        )

    return "\n".join(lines)


# -----------------------------
# Tool Definitions
# -----------------------------

@tool
def summarize_hiring_intent(conversation: str) -> str:
    """
    Extract role name, job level, key abilities, and vector search query.
    """

    # Wrap conversation into message format
    messages = [{"role": "user", "content": conversation}]

    # Call LLM with summarization prompt
    response = llm.invoke([
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": _messages_to_text(messages)}
    ])

    # Parse JSON response
    summary = _json_from_text(response.content)

    # Fallback if parsing fails
    if not isinstance(summary, dict):

        return json.dumps({
            "ready": False,
            "search_query": "",
            "role_summary": "",
            "missing_question": "Tell me the role name, job level, and key abilities."
        })

    # Clean retrieval query
    raw = summary.get("search_query", "")

    stop_words = {
        "assess",
        "assessment",
        "test",
        "candidate",
        "need",
        "want",
        "for",
        "role",
        "skills",
        "ability"
    }

    words = raw.lower().split()

    filtered = [
        w for w in words
        if w not in stop_words and len(w) > 2
    ]

    # Keep concise semantic query
    summary["search_query"] = " ".join(filtered[:6]) if filtered else raw

    # Ensure readiness validity
    summary["ready"] = bool(summary.get("ready", False)) and bool(summary["search_query"])

    return json.dumps(summary, ensure_ascii=False)


@tool
def retrieve_assessments(search_query: str) -> str:
    """
    Retrieve SHL assessments from Chroma vector database.
    """

    # Import retrieval pipeline
    from retrieval import retrieve_data

    # Semantic vector retrieval
    results = retrieve_data(search_query, k=6)

    # No results fallback
    if not results:
        return json.dumps([])

    compact = []

    # Compress metadata before sending to LLM
    for item in results:

        meta = item.get("metadata", {})

        compact.append({
            "name": meta.get("name", ""),
            "url": meta.get("link", ""),
            "keys": meta.get("keys", ""),
            "duration": meta.get("duration", ""),
            "languages": meta.get("languages", ""),
            "job_levels": meta.get("job_levels", ""),
        })

    return json.dumps(compact, ensure_ascii=False)


@tool
def check_client_satisfaction_tool(conversation: str) -> str:
    """
    Detect whether recruiter is satisfied with recommendations.
    """

    messages = [{"role": "user", "content": conversation}]

    # Call LLM with satisfaction prompt
    response = llm.invoke([
        {"role": "system", "content": SATISFACTION_PROMPT},
        {"role": "user", "content": _messages_to_text(messages)}
    ])

    payload = _json_from_text(response.content)

    # Fallback if parsing fails
    if not payload:
        payload = {
            "satisfied": False,
            "reply": ""
        }

    return json.dumps(payload, ensure_ascii=False)


# -----------------------------
# Tool Registry
# -----------------------------

# List of all tools available to the agent
TOOLS = [
    summarize_hiring_intent,
    retrieve_assessments,
    check_client_satisfaction_tool
]

# Mapping tool names to implementations
TOOL_MAP = {t.name: t for t in TOOLS}

# Bind tools to LLM
tool_calling_llm = llm.bind_tools(TOOLS)
