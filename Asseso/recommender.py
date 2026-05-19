from __future__ import annotations

import json
import re

# Used to run LLM pipeline with timeout protection
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Cache expensive operations in memory
from functools import lru_cache

# File path handling
from pathlib import Path

# Main conversational agent pipeline
from llm import generate_agent_reply


# -----------------------------
# Configuration
# -----------------------------

# Path to corrected SHL catalog JSON
CATALOG_PATH = Path(__file__).with_name("shl_product_catalog.json")

# Maximum conversation history size
MAX_TURNS = 16

# Maximum allowed pipeline execution time
PIPELINE_TIMEOUT_SECONDS = 30


# -----------------------------
# SHL Category Mapping
# -----------------------------

# Mapping SHL assessment categories to compact type codes
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
# Catalog Loading
# -----------------------------

@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:

    """
    Load corrected SHL catalog JSON.
    Cached to avoid repeated disk reads.
    """

    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def catalog_by_url() -> dict[str, dict]:

    """
    Create fast lookup dictionary:
    URL -> catalog item
    """

    return {
        item.get("link", ""): item
        for item in load_catalog()
        if item.get("link")
    }


# -----------------------------
# Text Utilities
# -----------------------------

def normalize(text: str) -> str:

    """
    Normalize whitespace and lowercase text.
    """

    return re.sub(r"\s+", " ", text.lower()).strip()


# -----------------------------
# Formatting Helpers
# -----------------------------

def type_code(item: dict) -> str:

    """
    Convert SHL categories into short type codes.
    """

    codes = []

    for key in item.get("keys", []):

        code = KEY_CODES.get(key)

        if code and code not in codes:
            codes.append(code)

    return ",".join(codes) or "K"


def format_keys(item: dict) -> str:

    """
    Format assessment categories into readable string.
    """

    keys = item.get("keys", [])

    if isinstance(keys, list):

        return ", ".join(
            str(k).strip()
            for k in keys
            if str(k).strip()
        ) or "-"

    return str(keys).strip() or "-"


def format_languages(item: dict, visible: int = 3) -> str:

    """
    Format supported languages.
    Truncate long language lists for cleaner UI.
    """

    langs = item.get("languages", [])

    if not isinstance(langs, list):
        return str(langs).strip() or "-"

    cleaned = [
        str(l).strip()
        for l in langs
        if str(l).strip()
    ]

    # Return all if short
    if len(cleaned) <= visible:
        return ", ".join(cleaned) or "-"

    # Compress long lists
    return f"{', '.join(cleaned[:visible])} (+{len(cleaned)-visible} more)"


def format_duration(item: dict) -> str:

    """
    Format assessment duration.
    """

    return str(item.get("duration", "")).strip() or "-"


# -----------------------------
# Catalog Validation
# -----------------------------

def catalog_item_for(candidate: dict) -> dict | None:

    """
    Validate whether recommended URL exists in SHL catalog.
    """

    url = candidate.get("url", "").strip()

    if url and url in catalog_by_url():
        return catalog_by_url()[url]

    return None


# -----------------------------
# Recommendation Sanitization
# -----------------------------

def clean_recommendations(raw_recommendations) -> list[dict]:

    """
    Clean and validate LLM recommendations.

    Removes:
    - hallucinated URLs
    - duplicates
    - invalid formats
    """

    cleaned = []

    seen_urls = set()

    # Ensure recommendations are list
    if not isinstance(raw_recommendations, list):
        return cleaned

    for raw in raw_recommendations:

        # Skip invalid objects
        if not isinstance(raw, dict):
            continue

        # Validate against SHL catalog
        item = catalog_item_for(raw)

        if not item:
            continue

        url = item.get("link", "")

        # Remove duplicates
        if not url or url in seen_urls:
            continue

        # Final cleaned recommendation
        cleaned.append({
            "name": item.get("name", ""),
            "url": url,
            "test_type": type_code(item),
            "keys": format_keys(item),
            "duration": format_duration(item),
            "languages": format_languages(item),
        })

        seen_urls.add(url)

        # Hard recommendation limit
        if len(cleaned) == 10:
            break

    return cleaned


# -----------------------------
# JSON Extraction
# -----------------------------

def extract_json_object(text: str) -> dict | None:

    """
    Extract JSON object from LLM response.
    Supports markdown JSON blocks.
    """

    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):

        text = re.sub(r"^```(?:json)?\s*", "", text)

        text = re.sub(r"\s*```$", "", text)

    # Direct JSON parsing
    try:
        return json.loads(text)

    except:

        # Regex fallback extraction
        match = re.search(r"\{.*\}", text, flags=re.S)

        if match:
            try:
                return json.loads(match.group(0))
            except:
                return None

    return None


# -----------------------------
# Payload Sanitization
# -----------------------------

def sanitize_payload(payload: dict | None, raw_text: str) -> dict:

    """
    Ensure final API response always matches required schema.
    """

    # Fallback when JSON parsing fails
    if not isinstance(payload, dict):

        return {
            "reply":
                raw_text.strip()
                or "I need a bit more detail before I can recommend SHL assessments.",

            "recommendations": [],

            "end_of_conversation": False,
        }

    # Extract assistant reply
    reply = str(payload.get("reply") or "").strip()

    # Clean recommendations
    recommendations = clean_recommendations(
        payload.get("recommendations", [])
    )

    # Fallback reply if empty
    if not reply:

        reply = (
            "I need a bit more detail before I can recommend SHL assessments."
        )

    return {
        "reply": reply,

        "recommendations": recommendations,

        "end_of_conversation":
            bool(payload.get("end_of_conversation", False)),
    }


# -----------------------------
# Main Conversation Pipeline
# -----------------------------

def build_reply(messages: list[dict]) -> dict:

    """
    Main orchestration pipeline.

    Responsibilities:
    - validate turn limits
    - enforce timeout protection
    - invoke conversational agent
    - sanitize outputs
    - guarantee schema compliance
    """

    # Conversation length protection
    if len(messages) > MAX_TURNS:

        return {
            "reply":
                "This conversation has reached the evaluator turn limit. "
                "Please start a new request with the key role details and constraints.",

            "recommendations": [],

            "end_of_conversation": True,
        }

    # Create isolated execution thread
    executor = ThreadPoolExecutor(max_workers=1)

    try:

        # Run LLM pipeline asynchronously
        future = executor.submit(
            generate_agent_reply,
            messages
        )

        # Timeout protection
        raw_response = future.result(
            timeout=PIPELINE_TIMEOUT_SECONDS
        )

    except TimeoutError:

        # Cancel stuck execution
        future.cancel()

        return {
            "reply":
                "I could not complete the recommendation within the evaluator timeout. "
                "Please retry with a shorter role description.",

            "recommendations": [],

            "end_of_conversation": False,
        }

    except Exception as exc:

        # Graceful error handling
        return {
            "reply":
                f"The LLM pipeline could not complete this turn: {exc}",

            "recommendations": [],

            "end_of_conversation": False,
        }

    finally:

        # Cleanup execution thread
        executor.shutdown(
            wait=False,
            cancel_futures=True
        )

    # Extract structured JSON response
    payload = extract_json_object(raw_response)

    # Final schema-safe output
    return sanitize_payload(
        payload,
        raw_response
    )
