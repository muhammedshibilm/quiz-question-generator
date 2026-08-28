"""
generate_questions.py

Runs daily via GitHub Actions.
1. Scrapes a short factual summary about each topic from Wikipedia (for grounding).
2. Feeds that context + instructions to the Google Gemini API (free tier,
   no credit card required) to generate interview-style multiple-choice
   questions in English.
3. Writes results to data/questions.json (committed back to the repo by the workflow).

NOTE: This used to call GitHub Models, but that service was fully retired
by GitHub on July 30, 2026. Gemini's free tier is the replacement - see
README.md for how to get a free API key and add it as a repo secret.

The app fetches this JSON from:
  https://raw.githubusercontent.com/<you>/<repo>/main/data/questions.json

Each question looks like:
{
  "id": "...",
  "topic": "ai-ml",
  "topic_name": "Artificial Intelligence & Machine Learning",
  "difficulty": "medium",
  "question": "...",
  "options": ["...", "...", "...", "..."],
  "correct_index": 2,
  "explanation": "...",
  "date_added": "2026-08-28"
}
"""

import os
import sys
import json
import time
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "questions.json"
TOPICS_PATH = SCRIPT_DIR / "topics.json"

# Get a free key (no credit card) from https://aistudio.google.com/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"  # generous free tier: ~250 requests/day
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

DIFFICULTIES = ["easy", "medium", "hard"]
QUESTIONS_PER_BATCH = 10
MAX_COMBOS_PER_RUN = 10  # topic + difficulty pairs per run (well under free daily quota)
DELAY_BETWEEN_CALLS_SEC = 5
RETENTION_DAYS = 60

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


def build_prompt(topic_name: str, difficulty: str, context: str) -> str:
    context_block = (
        f'\nHere is some factual background you can draw on for accuracy:\n"""{context}"""\n'
        if context
        else ""
    )
    return f"""Generate {QUESTIONS_PER_BATCH} unique interview-style multiple-choice questions on the topic "{topic_name}".
{context_block}
Requirements:
- Difficulty level: {difficulty}
- Write the question text and options in English
- Each question must have exactly 4 options, only one correct
- Questions should be realistic interview questions someone could be asked for a job in this area, not trivia
- Avoid duplicating extremely common textbook questions where possible; vary phrasing
- Include a short 1-2 sentence explanation for the correct answer

Respond with ONLY a raw JSON array (no markdown fences, no preamble), where each item has this exact shape:
{{
  "question": "string",
  "options": ["string", "string", "string", "string"],
  "correct_index": 0,
  "explanation": "string"
}}"""


def call_model(prompt: str) -> list:
    resp = requests.post(
        ENDPOINT,
        headers={"Content-Type": "application/json"},
        params={"key": GEMINI_API_KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 4000,
                "responseMimeType": "application/json",
            },
        },
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response shape: {data}")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


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

    for topic, difficulty in todays_combos:
        print(f"Generating: {topic['name']} | {difficulty}")

        context = scrape_topic_context(topic["name"])
        prompt = build_prompt(topic["name"], difficulty, context)

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
            print(f"  Failed for {topic['id']}/{difficulty}: {e}", file=sys.stderr)

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
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_questions": len(merged),
        "questions": merged,
    }

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(merged)} total questions to {DATA_PATH}")


if __name__ == "__main__":
    main()
