import os
import sys
import json
import time
import random
import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "questions.json"
TOPICS_PATH = SCRIPT_DIR / "topics.json"

# Get a free key (no credit card) from https://aistudio.google.com/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.6-flash"  # generous free tier: ~250 requests/day
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

DIFFICULTIES = ["easy", "medium", "hard"]
QUESTIONS_PER_BATCH = 10
MAX_COMBOS_PER_RUN = 10  # topic + difficulty pairs per run (well under free daily quota)
DELAY_BETWEEN_CALLS_SEC = 5
RETENTION_DAYS = 60

# Retry/backoff settings for transient API failures (503 overload, 429 rate limit,
# and truncated/invalid JSON responses).
MAX_ATTEMPTS = 4
BASE_BACKOFF_SEC = 5

# Generous headroom so 10 questions + explanations don't get cut off mid-JSON.
MAX_OUTPUT_TOKENS = 8192

# Simple in-memory cache so we don't scrape Wikipedia twice for the
# same topic within a single run.
_wiki_cache = {}


def scrape_topic_context(topic_name: str) -> str:
    """
    Pulls a short factual summary from Wikipedia's public REST API to
    ground question generation in real facts. Falls back to an empty
    string if the page isn't found or the request fails - the prompt
    still works fine without it, just less grounded.
    """
    if topic_name in _wiki_cache:
        return _wiki_cache[topic_name]

    context = ""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic_name.replace(' ', '_')}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "quiz-app-question-generator"})
        if resp.status_code == 200:
            data = resp.json()
            context = data.get("extract", "")
    except requests.RequestException as e:
        print(f"  (scrape failed for '{topic_name}': {e})")

    _wiki_cache[topic_name] = context
    return context


def build_prompt(topic_name: str, difficulty: str, context: str, num_questions: int) -> str:
    context_block = (
        f'\nHere is some factual background you can draw on for accuracy:\n"""{context}"""\n'
        if context
        else ""
    )
    return f"""Generate {num_questions} unique interview-style multiple-choice questions on the topic "{topic_name}".
{context_block}
Requirements:
- Difficulty level: {difficulty}
- Write the question text and options in English
- Each question must have exactly 4 options, only one correct
- Questions should be realistic interview questions someone could be asked for a job in this area, not trivia
- Avoid duplicating extremely common textbook questions where possible; vary phrasing
- Include a short 1-2 sentence explanation for the correct answer
- Keep each explanation concise (max ~30 words) so the full response fits comfortably

Respond with ONLY a raw JSON array (no markdown fences, no preamble), where each item has this exact shape:
{{
  "question": "string",
  "options": ["string", "string", "string", "string"],
  "correct_index": 0,
  "explanation": "string"
}}"""


class TransientAPIError(Exception):
    """Raised for errors worth retrying: 429/503 responses, empty/truncated output."""


def _extract_text(data: dict) -> str:
    try:
        candidate = data["candidates"][0]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response shape: {data}")

    finish_reason = candidate.get("finishReason")
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()

    if not text:
        raise TransientAPIError(f"Empty response text (finishReason={finish_reason}): {data}")

    if finish_reason == "MAX_TOKENS":
        # The response was cut off mid-generation - almost certainly invalid JSON.
        raise TransientAPIError("Response was truncated (hit MAX_TOKENS)")

    return text


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip a leading ```json / ``` fence and a trailing ``` fence, if present.
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def call_model(prompt: str) -> list:
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                ENDPOINT,
                headers={"Content-Type": "application/json"},
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.8,
                        "maxOutputTokens": MAX_OUTPUT_TOKENS,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=90,
            )

            if resp.status_code in (429, 503):
                raise TransientAPIError(f"Gemini API error {resp.status_code}: {resp.text}")
            if not resp.ok:
                # Non-transient errors (400, 401, 404, ...) - fail fast, no point retrying.
                raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

            data = resp.json()
            text = _extract_text(data)
            cleaned = _clean_json_text(text)

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise TransientAPIError(f"Invalid JSON from model: {e}")

        except TransientAPIError as e:
            last_error = e
            if attempt == MAX_ATTEMPTS:
                break
            backoff = BASE_BACKOFF_SEC * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"  Attempt {attempt}/{MAX_ATTEMPTS} failed ({e}); retrying in {backoff:.1f}s")
            time.sleep(backoff)
        except requests.RequestException as e:
            last_error = e
            if attempt == MAX_ATTEMPTS:
                break
            backoff = BASE_BACKOFF_SEC * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"  Network error on attempt {attempt}/{MAX_ATTEMPTS} ({e}); retrying in {backoff:.1f}s")
            time.sleep(backoff)

    raise RuntimeError(f"Gave up after {MAX_ATTEMPTS} attempts: {last_error}")


def make_id(topic_id: str, difficulty: str, question_text: str) -> str:
    h = hashlib.sha1(f"{topic_id}-{difficulty}-{question_text}".encode()).hexdigest()[:12]
    return f"{topic_id}-{difficulty}-{h}"


def main():
    if not GEMINI_API_KEY:
        print("Missing GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    with open(TOPICS_PATH) as f:
        topics_config = json.load(f)

    topics = topics_config["topics"]
    combos = [(topic, difficulty) for topic in topics for difficulty in DIFFICULTIES]

    day_of_year = datetime.now().timetuple().tm_yday
    start_idx = (day_of_year * MAX_COMBOS_PER_RUN) % len(combos)
    todays_combos = [combos[(start_idx + i) % len(combos)] for i in range(MAX_COMBOS_PER_RUN)]

    today_str = date.today().isoformat()
    all_questions = []
    failures = []

    for topic, difficulty in todays_combos:
        print(f"Generating: {topic['name']} | {difficulty}")

        context = scrape_topic_context(topic["name"])
        prompt = build_prompt(topic["name"], difficulty, context, QUESTIONS_PER_BATCH)

        try:
            questions = call_model(prompt)
            for q in questions:
                all_questions.append(
                    {
                        "id": make_id(topic["id"], difficulty, q["question"]),
                        "topic": topic["id"],
                        "topic_name": topic["name"],
                        "difficulty": difficulty,
                        "question": q["question"],
                        "options": q["options"],
                        "correct_index": q["correct_index"],
                        "explanation": q.get("explanation", ""),
                        "date_added": today_str,
                    }
                )
        except Exception as e:
            msg = f"Failed for {topic['id']}/{difficulty}: {e}"
            print(f"  {msg}", file=sys.stderr)
            failures.append(msg)

        time.sleep(DELAY_BETWEEN_CALLS_SEC)

    # Merge with existing questions, keep last RETENTION_DAYS days only
    existing = {"generated_at": None, "questions": []}
    if DATA_PATH.exists():
        with open(DATA_PATH) as f:
            existing = json.load(f)

    cutoff_str = (date.today() - timedelta(days=RETENTION_DAYS)).isoformat()
    merged = [q for q in existing["questions"] if q["date_added"] >= cutoff_str]
    existing_ids = {q["id"] for q in merged}

    for q in all_questions:
        if q["id"] not in existing_ids:
            merged.append(q)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_questions": len(merged),
        "questions": merged,
    }

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(merged)} total questions to {DATA_PATH}")

    if failures:
        print(f"\n{len(failures)}/{len(todays_combos)} combo(s) failed after retries:", file=sys.stderr)
        for f_msg in failures:
            print(f"  - {f_msg}", file=sys.stderr)
        # Don't fail the whole CI run over partial generation failures - the questions
        # that did succeed are still written. Remove this exit(1) suppression if you'd
        # rather CI go red whenever any combo fails.
        # sys.exit(1)


if __name__ == "__main__":
    main()
    
