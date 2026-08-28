import os
import sys
import json
import time
import random
import hashlib
from collections import defaultdict
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
- Prioritize the most commonly asked, highest-value interview questions for this topic and
  difficulty level - the questions a real candidate is most likely to actually be asked, not
  obscure trivia
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
    """Raised for errors worth retrying: 503 responses, empty/truncated output,
    and short-lived 429 rate limits."""


class QuotaExhaustedError(Exception):
    """Raised when a 429 is a daily/hard quota cap rather than a short burst
    limit. Retrying this within the same run just wastes remaining quota on
    other combos, so we fail fast instead."""


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
        raise TransientAPIError("Response was truncated (hit MAX_TOKENS)")

    return text


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
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

            if resp.status_code == 429:
                is_daily_quota = "PerDay" in resp.text
                if is_daily_quota:
                    raise QuotaExhaustedError(f"Gemini API error 429 (daily quota): {resp.text}")
                raise TransientAPIError(f"Gemini API error 429: {resp.text}")
            if resp.status_code == 503:
                raise TransientAPIError(f"Gemini API error 503: {resp.text}")
            if not resp.ok:
                raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

            data = resp.json()
            text = _extract_text(data)
            cleaned = _clean_json_text(text)

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise TransientAPIError(f"Invalid JSON from model: {e}")

        except QuotaExhaustedError as e:
            print(f"  Daily quota exhausted, skipping retries for this combo: {e}")
            raise RuntimeError(str(e))
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


def select_todays_combos(topics: list, existing_questions: list) -> list:
    """
    Picks which (topic, difficulty) combos to generate today, prioritizing
    combos with the fewest existing questions so every topic fills in as
    fast as possible instead of waiting on a fixed calendar rotation.

    Within topics that are equally under-covered, selection is randomized
    (not alphabetical) so runs don't always favor the same topics in the
    same order.
    """
    counts = defaultdict(int)
    for q in existing_questions:
        counts[(q["topic"], q["difficulty"])] += 1

    topic_totals = {
        topic["id"]: sum(counts[(topic["id"], d)] for d in DIFFICULTIES)
        for topic in topics
    }

    # Order topics least-covered first; shuffle ties so it's not always
    # alphabetical / the same topic order every run.
    topics_shuffled = topics[:]
    random.shuffle(topics_shuffled)
    topics_sorted = sorted(topics_shuffled, key=lambda t: topic_totals[t["id"]])

    # For each topic, queue its difficulties least-covered first (ties shuffled).
    per_topic_queue = {}
    for topic in topics_sorted:
        diffs = DIFFICULTIES[:]
        random.shuffle(diffs)
        diffs.sort(key=lambda d: counts[(topic["id"], d)])
        per_topic_queue[topic["id"]] = diffs

    # Round-robin across topics (in least-covered-first order) so a single
    # run spreads across many topics rather than exhausting one topic's
    # three difficulties before moving to the next.
    selected = []
    topic_cursor = {topic["id"]: 0 for topic in topics_sorted}
    while len(selected) < MAX_COMBOS_PER_RUN:
        made_progress = False
        for topic in topics_sorted:
            if len(selected) >= MAX_COMBOS_PER_RUN:
                break
            cursor = topic_cursor[topic["id"]]
            queue = per_topic_queue[topic["id"]]
            if cursor < len(queue):
                selected.append((topic, queue[cursor]))
                topic_cursor[topic["id"]] += 1
                made_progress = True
        if not made_progress:
            break  # every topic's difficulties are exhausted (shouldn't happen with 3 diffs)

    return selected


def main():
    if not GEMINI_API_KEY:
        print("Missing GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    with open(TOPICS_PATH) as f:
        topics_config = json.load(f)
    topics = topics_config["topics"]

    # Load existing questions up front so selection can see current coverage.
    existing = {"generated_at": None, "questions": []}
    if DATA_PATH.exists():
        with open(DATA_PATH) as f:
            existing = json.load(f)

    cutoff_str = (date.today() - timedelta(days=RETENTION_DAYS)).isoformat()
    kept_existing = [q for q in existing["questions"] if q["date_added"] >= cutoff_str]

    todays_combos = select_todays_combos(topics, kept_existing)
    print("Today's combos (least-covered first):")
    for topic, difficulty in todays_combos:
        print(f"  - {topic['id']}/{difficulty}")

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

    # Merge with existing questions (dedupe by id), keep last RETENTION_DAYS days only
    existing_ids = {q["id"] for q in kept_existing}
    merged = kept_existing[:]
    for q in all_questions:
        if q["id"] not in existing_ids:
            merged.append(q)
            existing_ids.add(q["id"])

    # Also drop any topic ids no longer present in topics.json (e.g. HR was removed) -
    # keeps the question bank from carrying dead weight for topics you took out.
    valid_topic_ids = {t["id"] for t in topics}
    removed_count = len([q for q in merged if q["topic"] not in valid_topic_ids])
    merged = [q for q in merged if q["topic"] in valid_topic_ids]
    if removed_count:
        print(f"Removed {removed_count} question(s) belonging to topics no longer in topics.json")

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


if __name__ == "__main__":
    main()
    
