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
DATA_DIR = SCRIPT_DIR.parent / "data"
QUESTIONS_DIR = DATA_DIR / "questions"
MANIFEST_PATH = DATA_DIR / "manifest.json"
TOPICS_PATH = SCRIPT_DIR / "topics.json"

# GitHub Models - OpenAI-compatible free inference API.
# In GitHub Actions, GITHUB_TOKEN already has access if the workflow requests
# the `models: read` permission - no separate API key/secret needed.
# Locally, use a fine-grained PAT with the `models:read` scope instead.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
MODEL = "openai/gpt-4o-mini"
ENDPOINT = "https://models.github.ai/inference/chat/completions"

DIFFICULTIES = ["easy", "medium", "hard"]
QUESTIONS_PER_BATCH = 10
MAX_COMBOS_PER_RUN = 10
DELAY_BETWEEN_CALLS_SEC = 5
RETENTION_DAYS = 60

# Once a topic/difficulty has this many questions, stop generating more for it.
MAX_QUESTIONS_PER_COMBO = 60

MAX_ATTEMPTS = 4
BASE_BACKOFF_SEC = 5

# GitHub Models free tier caps responses around 4K output tokens per request -
# keep comfortably under that.
MAX_OUTPUT_TOKENS = 3800

_wiki_cache = {}


def scrape_topic_context(topic_name: str) -> str:
    if topic_name in _wiki_cache:
        return _wiki_cache[topic_name]
    context = ""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic_name.replace(' ', '_')}"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "quiz-app-question-generator"})
        if resp.status_code == 200:
            context = resp.json().get("extract", "")
    except requests.RequestException as e:
        print(f"  (scrape failed for '{topic_name}': {e})")
    _wiki_cache[topic_name] = context
    return context


def build_prompt(topic_name: str, difficulty: str, context: str, num_questions: int) -> str:
    context_block = (
        f'\nHere is some factual background you can draw on for accuracy:\n"""{context}"""\n'
        if context else ""
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
    """Retryable: 429 short-lived rate limit, 503, empty/truncated output."""


class QuotaExhaustedError(Exception):
    """A daily/hard quota cap - retrying within this run won't help."""


def _extract_text(data: dict) -> str:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected GitHub Models response shape: {data}")

    finish_reason = choice.get("finish_reason")
    text = (choice.get("message", {}) or {}).get("content", "") or ""
    text = text.strip()

    if not text:
        raise TransientAPIError(f"Empty response text (finish_reason={finish_reason}): {data}")
    if finish_reason == "length":
        raise TransientAPIError("Response was truncated (hit max_tokens)")

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
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                },
                timeout=90,
            )

            if resp.status_code == 429:
                # A daily/per-user-per-model quota exhaustion won't recover within
                # this run; a short-lived RPM rate limit will. Check which one.
                if "UserByModelByDay" in resp.text or "per day" in resp.text.lower():
                    raise QuotaExhaustedError(f"GitHub Models 429 (daily quota): {resp.text}")
                raise TransientAPIError(f"GitHub Models 429: {resp.text}")
            if resp.status_code == 503:
                raise TransientAPIError(f"GitHub Models 503: {resp.text}")
            if not resp.ok:
                raise RuntimeError(f"GitHub Models error {resp.status_code}: {resp.text}")

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


def load_topic_file(topic_id: str) -> list:
    path = QUESTIONS_DIR / f"{topic_id}.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("questions", [])


def select_todays_combos(topics: list, counts: dict) -> list:
    topic_totals = {
        topic["id"]: sum(counts[(topic["id"], d)] for d in DIFFICULTIES)
        for topic in topics
    }

    topics_shuffled = topics[:]
    random.shuffle(topics_shuffled)
    topics_sorted = sorted(topics_shuffled, key=lambda t: topic_totals[t["id"]])

    per_topic_queue = {}
    for topic in topics_sorted:
        diffs = [d for d in DIFFICULTIES if counts[(topic["id"], d)] < MAX_QUESTIONS_PER_COMBO]
        random.shuffle(diffs)
        diffs.sort(key=lambda d: counts[(topic["id"], d)])
        per_topic_queue[topic["id"]] = diffs

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
            break

    return selected


def main():
    if not GITHUB_TOKEN:
        print("Missing GITHUB_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    with open(TOPICS_PATH) as f:
        topics = json.load(f)["topics"]

    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    cutoff_str = (date.today() - timedelta(days=RETENTION_DAYS)).isoformat()

    topic_questions = {}
    for topic in topics:
        raw = load_topic_file(topic["id"])
        topic_questions[topic["id"]] = [q for q in raw if q["date_added"] >= cutoff_str]

    counts = defaultdict(int)
    for topic_id, qs in topic_questions.items():
        for q in qs:
            counts[(topic_id, q["difficulty"])] += 1

    todays_combos = select_todays_combos(topics, counts)
    print(f"Today's combos (least-covered first, capped at {MAX_QUESTIONS_PER_COMBO}/combo):")
    for topic, difficulty in todays_combos:
        print(f"  - {topic['id']}/{difficulty}")

    today_str = date.today().isoformat()
    failures = []

    for topic, difficulty in todays_combos:
        print(f"Generating: {topic['name']} | {difficulty}")
        context = scrape_topic_context(topic["name"])
        prompt = build_prompt(topic["name"], difficulty, context, QUESTIONS_PER_BATCH)

        try:
            questions = call_model(prompt)
            existing_ids = {q["id"] for q in topic_questions[topic["id"]]}
            for q in questions:
                new_q = {
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
                if new_q["id"] not in existing_ids:
                    topic_questions[topic["id"]].append(new_q)
                    existing_ids.add(new_q["id"])
        except Exception as e:
            msg = f"Failed for {topic['id']}/{difficulty}: {e}"
            print(f"  {msg}", file=sys.stderr)
            failures.append(msg)

        time.sleep(DELAY_BETWEEN_CALLS_SEC)

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_topics = []

    valid_topic_ids = {t["id"] for t in topics}
    for existing_file in QUESTIONS_DIR.glob("*.json"):
        if existing_file.stem not in valid_topic_ids:
            existing_file.unlink()
            print(f"Removed stale topic file: {existing_file.name}")

    for topic in topics:
        qs = topic_questions[topic["id"]]
        with open(QUESTIONS_DIR / f"{topic['id']}.json", "w") as f:
            json.dump(
                {"topic": topic["id"], "topic_name": topic["name"], "generated_at": now_iso, "questions": qs},
                f, indent=2, ensure_ascii=False,
            )
        diff_counts = {d: len([q for q in qs if q["difficulty"] == d]) for d in DIFFICULTIES}
        manifest_topics.append({
            "id": topic["id"], "name": topic["name"], "counts": diff_counts, "total": len(qs),
        })

    with open(MANIFEST_PATH, "w") as f:
        json.dump({"generated_at": now_iso, "topics": manifest_topics}, f, indent=2, ensure_ascii=False)

    total = sum(t["total"] for t in manifest_topics)
    print(f"Wrote {len(topics)} topic files ({total} total questions) and manifest.json")

    if failures:
        print(f"\n{len(failures)}/{len(todays_combos)} combo(s) failed after retries:", file=sys.stderr)
        for f_msg in failures:
            print(f"  - {f_msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
