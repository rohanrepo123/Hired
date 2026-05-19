from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from pathlib import Path

from llm import generate_agent_reply          # ✅ correct import

CATALOG_PATH = Path(__file__).with_name("shl_product_catalog.json")
MAX_TURNS = 16
PIPELINE_TIMEOUT_SECONDS = 25

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

@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def catalog_by_url() -> dict[str, dict]:
    return {item.get("link", ""): item for item in load_catalog() if item.get("link")}

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def type_code(item: dict) -> str:
    codes = []
    for key in item.get("keys", []):
        code = KEY_CODES.get(key)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) or "K"

def format_keys(item: dict) -> str:
    keys = item.get("keys", [])
    if isinstance(keys, list):
        return ", ".join(str(k).strip() for k in keys if str(k).strip()) or "-"
    return str(keys).strip() or "-"

def format_languages(item: dict, visible: int = 3) -> str:
    langs = item.get("languages", [])
    if not isinstance(langs, list):
        return str(langs).strip() or "-"
    cleaned = [str(l).strip() for l in langs if str(l).strip()]
    if len(cleaned) <= visible:
        return ", ".join(cleaned) or "-"
    return f"{', '.join(cleaned[:visible])} (+{len(cleaned)-visible} more)"

def format_duration(item: dict) -> str:
    return str(item.get("duration", "")).strip() or "-"

def catalog_item_for(candidate: dict) -> dict | None:
    url = candidate.get("url", "").strip()
    if url and url in catalog_by_url():
        return catalog_by_url()[url]
    return None

def clean_recommendations(raw_recommendations) -> list[dict]:
    cleaned = []
    seen_urls = set()
    if not isinstance(raw_recommendations, list):
        return cleaned
    for raw in raw_recommendations:
        if not isinstance(raw, dict):
            continue
        item = catalog_item_for(raw)
        if not item:
            continue
        url = item.get("link", "")
        if not url or url in seen_urls:
            continue
        cleaned.append({
            "name": item.get("name", ""),
            "url": url,
            "test_type": type_code(item),
            "keys": format_keys(item),
            "duration": format_duration(item),
            "languages": format_languages(item),
        })
        seen_urls.add(url)
        if len(cleaned) == 10:
            break
    return cleaned

def extract_json_object(text: str) -> dict | None:
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

def sanitize_payload(payload: dict | None, raw_text: str) -> dict:
    if not isinstance(payload, dict):
        return {
            "reply": raw_text.strip() or "I need a bit more detail before I can recommend SHL assessments.",
            "recommendations": [],
            "end_of_conversation": False,
        }
    reply = str(payload.get("reply") or "").strip()
    recommendations = clean_recommendations(payload.get("recommendations", []))
    if not reply:
        reply = "I need a bit more detail before I can recommend SHL assessments."
    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": bool(payload.get("end_of_conversation", False)),
    }

def build_reply(messages: list[dict]) -> dict:
    if len(messages) > MAX_TURNS:
        return {
            "reply": "This conversation has reached the evaluator turn limit. Please start a new request with the key role details and constraints.",
            "recommendations": [],
            "end_of_conversation": True,
        }

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(generate_agent_reply, messages)
        raw_response = future.result(timeout=PIPELINE_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        return {
            "reply": "I could not complete the recommendation within the evaluator timeout. Please retry with a shorter role description.",
            "recommendations": [],
            "end_of_conversation": False,
        }
    except Exception as exc:
        return {
            "reply": f"The LLM pipeline could not complete this turn: {exc}",
            "recommendations": [],
            "end_of_conversation": False,
        }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    payload = extract_json_object(raw_response)
    return sanitize_payload(payload, raw_response)
