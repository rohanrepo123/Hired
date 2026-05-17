from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from pathlib import Path

from llm import generate_agent_reply


CATALOG_PATH = Path(__file__).with_name("shl_product_catalog.json")
MAX_TURNS = 15
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
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def catalog_by_url() -> dict[str, dict]:
    return {item.get("link", ""): item for item in load_catalog() if item.get("link")}


@lru_cache(maxsize=1)
def catalog_by_name() -> dict[str, dict]:
    return {normalize(item.get("name", "")): item for item in load_catalog()}


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
        return ", ".join(str(key).strip() for key in keys if str(key).strip()) or "-"
    return str(keys).strip() or "-"


def format_languages(item: dict, visible: int = 3) -> str:
    languages = item.get("languages", [])
    if not isinstance(languages, list):
        text = str(languages).strip()
        return text or "-"

    cleaned = [str(language).strip() for language in languages if str(language).strip()]
    if len(cleaned) <= visible:
        return ", ".join(cleaned) or "-"

    remaining = len(cleaned) - visible
    return f"{', '.join(cleaned[:visible])} (+{remaining} more)"


def format_duration(item: dict) -> str:
    return str(item.get("duration", "")).strip() or "-"


def extract_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def catalog_item_for(candidate: dict) -> dict | None:
    url = str(candidate.get("url", "")).strip()
    name = normalize(str(candidate.get("name", "")))

    if url and url in catalog_by_url():
        return catalog_by_url()[url]

    if name and name in catalog_by_name():
        return catalog_by_name()[name]

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

        cleaned.append(
            {
                "name": item.get("name", ""),
                "url": url,
                "test_type": type_code(item),
                "keys": format_keys(item),
                "duration": format_duration(item),
                "languages": format_languages(item),
            }
        )
        seen_urls.add(url)

        if len(cleaned) == 10:
            break

    return cleaned


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
            "reply": "I could not complete the recommendation within the evaluator timeout. Please retry with a shorter role description and the most important skills.",
            "recommendations": [],
            "end_of_conversation": False,
        }
    except Exception as exc:
        return {
            "reply": f"The LLM/RAG pipeline could not complete this turn: {exc}",
            "recommendations": [],
            "end_of_conversation": False,
        }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    payload = extract_json_object(raw_response)
    return sanitize_payload(payload, raw_response)
