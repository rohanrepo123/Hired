import json
import os
import re
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.35,
    max_retries=1,
    request_timeout=LLM_TIMEOUT_SECONDS,
)

# ---------- PROMPTS ----------
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

# ---------- Helper functions ----------
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

def _messages_to_text(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)

def _latest_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"].strip()
    return ""

def _json_from_text(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return None
    return None

def _type_code(keys_str: str) -> str:
    codes = []
    for key in keys_str.split(","):
        code = KEY_CODES.get(key.strip())
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) or "K"

def _build_markdown_table(recs: list, role_summary: str) -> str:
    if not recs:
        return ""
    lines = [f"For {role_summary or 'your role'}:", "",
             "| # | Name | Test Type | Keys | Duration | Languages | URL |",
             "|---|------|-----------|------|----------|-----------|-----|"]
    for idx, r in enumerate(recs, 1):
        lines.append(f"| {idx} | {r['name']} | {r['test_type']} | - | - | - | <{r['url']}> |")
    return "\n".join(lines)

# ---------- Tool definitions (MUST have docstrings) ----------
@tool
def summarize_hiring_intent(conversation: str) -> str:
    """
    Extract role name, job level, key abilities, and a vector-search query from the conversation.
    Returns JSON with fields: ready, search_query, role_summary, missing_question.
    """
    messages = [{"role": "user", "content": conversation}]
    response = llm.invoke([
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": _messages_to_text(messages)}
    ])
    summary = _json_from_text(response.content)
    if not isinstance(summary, dict):
        return json.dumps({"ready": False, "search_query": "", "role_summary": "", "missing_question": "Tell me the role name, job level, and key abilities."})
    
    raw = summary.get("search_query", "")
    stop_words = {"assess", "assessment", "test", "candidate", "need", "want", "for", "role", "skills", "ability"}
    words = raw.lower().split()
    filtered = [w for w in words if w not in stop_words and len(w) > 2]
    summary["search_query"] = " ".join(filtered[:6]) if filtered else raw
    summary["ready"] = bool(summary.get("ready", False)) and bool(summary["search_query"])
    return json.dumps(summary, ensure_ascii=False)

@tool
def retrieve_assessments(search_query: str) -> str:
    """
    Retrieve SHL assessments from the Chroma vector database using a concise search query.
    Returns a list of products with name, url, keys, duration, languages, job_levels.
    """
    from retrieval import retrieve_data
    results = retrieve_data(search_query, k=6)
    if not results:
        # Fallback: ask for more specific abilities
        return json.dumps([])
    compact = []
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
    Check whether the client is satisfied with the current shortlist.
    Returns JSON with fields: satisfied, reply.
    """
    messages = [{"role": "user", "content": conversation}]
    response = llm.invoke([
        {"role": "system", "content": SATISFACTION_PROMPT},
        {"role": "user", "content": _messages_to_text(messages)}
    ])
    payload = _json_from_text(response.content)
    if not payload:
        payload = {"satisfied": False, "reply": ""}
    return json.dumps(payload, ensure_ascii=False)

# ---------- Now define TOOLS list ----------
TOOLS = [summarize_hiring_intent, retrieve_assessments, check_client_satisfaction_tool]
TOOL_MAP = {t.name: t for t in TOOLS}
tool_calling_llm = llm.bind_tools(TOOLS)

# ---------- Agent execution ----------
def _repair_recommendation_payload(raw_response: str, plan_action: str, retrieved_context: list, role_summary: str) -> str:
    """Ensure recommendations come from retrieved context and include a markdown table."""
    payload = _json_from_text(raw_response)
    if plan_action != "retrieve":
        return raw_response

    context_recs = []
    seen = set()
    for item in retrieved_context:
        name = item.get("name", "").strip()
        url = item.get("url", "").strip()
        if name and url and url not in seen:
            context_recs.append({
                "name": name,
                "url": url,
                "test_type": _type_code(item.get("keys", ""))
            })
            seen.add(url)
        if len(context_recs) == 8:
            break

    if not context_recs:
        # fallback to default senior leadership products
        context_recs = [
            {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"},
            {"name": "Executive Scenarios", "url": "https://www.shl.com/products/product-catalog/view/executive-scenarios/", "test_type": "B"},
            {"name": "Enterprise Leadership Report 2.0", "url": "https://www.shl.com/products/product-catalog/view/enterprise-leadership-report-2-0/", "test_type": "P"},
        ]

    table = _build_markdown_table(context_recs, role_summary)
    if not isinstance(payload, dict):
        payload = {"reply": "Here are recommended SHL assessments.", "recommendations": [], "end_of_conversation": False}
    reply = payload.get("reply", "")
    if "| # | Name | Test Type |" not in reply:
        reply = (reply + "\n\n" + table) if reply else table
    payload["reply"] = reply.strip()
    payload["recommendations"] = context_recs
    payload["end_of_conversation"] = bool(payload.get("end_of_conversation", False))
    return json.dumps(payload, ensure_ascii=False)

def _run_tool_calling_agent(messages: list[dict]) -> str:
    conversation = _messages_to_text(messages)
    agent_messages = [
        {"role": "system", "content": TOOL_AGENT_PROMPT},
        {"role": "user", "content": conversation}
    ]
    retrieved_context = []
    role_summary = ""

    for _ in range(4):
        response = tool_calling_llm.invoke(agent_messages)
        agent_messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            raw = str(response.content or "")
            if retrieved_context:
                return _repair_recommendation_payload(raw, "retrieve", retrieved_context, role_summary)
            return raw

        for tc in tool_calls:
            name = tc.get("name", "")
            tool_fn = TOOL_MAP.get(name)
            if not tool_fn:
                result = json.dumps({"error": f"unknown tool: {name}"})
            else:
                args = tc.get("args", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                result = tool_fn.invoke(args)

            if name == "summarize_hiring_intent":
                try:
                    summ = json.loads(result)
                    role_summary = summ.get("role_summary", "")
                except:
                    pass
            if name == "retrieve_assessments":
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        retrieved_context = parsed
                except:
                    pass

            agent_messages.append(ToolMessage(content=str(result), name=name, tool_call_id=tc.get("id", "")))

    if retrieved_context:
        return _repair_recommendation_payload(
            json.dumps({"reply": "Here are SHL assessments matching your request.", "recommendations": [], "end_of_conversation": False}),
            "retrieve", retrieved_context, role_summary
        )
    return json.dumps({"reply": "I need more details: role name, job level, and key abilities.", "recommendations": [], "end_of_conversation": False}, ensure_ascii=False)

def generate_agent_reply(messages: list[dict]) -> str:
    return _run_tool_calling_agent(messages)

# ---------- Deterministic first question (optional) ----------
def deterministic_first_question(messages: list[dict]) -> str:
    text = _latest_user_text(messages).lower()
    role_terms = ("cxo", "director", "executive", "vp", "senior leadership", "manager", "lead", "developer", "engineer")
    level_terms = ("senior", "executive", "director", "cxo", "15 years", "experienced", "entry", "junior", "mid", "graduate")
    ability_terms = ("leadership", "strategy", "decision making", "influencing", "java", "python", "sql", "aws", "communication", "sales", "finance")

    has_role = any(term in text for term in role_terms)
    has_level = any(term in text for term in level_terms)
    has_abilities = any(term in text for term in ability_terms)

    if has_role and has_level and has_abilities:
        return ""
    if not has_role:
        return "What specific role are you hiring for (e.g., CXO, director, manager)?"
    if not has_level:
        return "What job level is this (e.g., entry, mid, senior, executive)?"
    if not has_abilities:
        return "What are the key abilities or skills most critical for this role (e.g., strategic thinking, influencing, technical skills)?"
    return ""
