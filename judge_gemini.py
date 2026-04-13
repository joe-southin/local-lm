#!/usr/bin/env python3
"""Run Gemini 2.5 Pro as a secondary judge on existing raw_answers.json.

Reads raw_answers.json from a results directory, runs Gemini judge on every
answer, writes scores_gemini.json alongside the existing scores.json. Opus
scores are left untouched.

Usage:
    python judge_gemini.py [results_dir] [--model MODEL]

If no directory given, runs on ./results/full_sweep.
Default model is gemini-2.5-flash (higher rate limits than pro).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from eval_harness import AnswerRecord, EvalQuestion, ScoreRecord, log

load_dotenv()

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator. Score the following answer on three dimensions, "
    "each from 1 (worst) to 5 (best):\n"
    "- accuracy: Is the answer factually correct given the context?\n"
    "- completeness: Does it address all parts of the question?\n"
    "- coherence: Is it well-structured and easy to understand?\n\n"
    'Respond in JSON only: {"accuracy": N, "completeness": N, "coherence": N, "reasoning": "..."}'
)


def judge_with_gemini(
    client: genai.Client,
    question: EvalQuestion,
    answer: AnswerRecord,
    model: str = "gemini-2.5-flash",
) -> ScoreRecord:
    """Ask Gemini to score an answer blind (no model name)."""
    user_msg = (
        f"Context:\n{question.context}\n\n"
        f"Question:\n{question.question}\n\n"
        f"Reference answer:\n{question.reference_answer}\n\n"
        f"Answer to evaluate:\n{answer.answer}"
    )

    config = genai_types.GenerateContentConfig(
        system_instruction=JUDGE_SYSTEM_PROMPT,
        temperature=0.0,
        response_mime_type="application/json",
    )

    for attempt in range(2):
        response = client.models.generate_content(
            model=model,
            contents=user_msg,
            config=config,
        )
        text = response.text or ""
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned)
            return ScoreRecord(
                question_id=answer.question_id,
                model_name=answer.model_name,
                temperature=answer.temperature,
                top_p=answer.top_p,
                accuracy=int(data["accuracy"]),
                completeness=int(data["completeness"]),
                coherence=int(data["coherence"]),
                judge_reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt == 0:
                continue
            return ScoreRecord(
                question_id=answer.question_id,
                model_name=answer.model_name,
                temperature=answer.temperature,
                top_p=answer.top_p,
                accuracy=-1,
                completeness=-1,
                coherence=-1,
                judge_reasoning=f"Parse failure: {text[:200]}",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="./results/full_sweep")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model (default: gemini-2.5-flash)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model = args.model
    log(f"Using judge model: {model}")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Load questions
    with open(Path(__file__).parent / "eval_questions.json") as f:
        questions = {q["id"]: EvalQuestion(**q) for q in json.load(f)}

    # Load raw answers
    raw_path = results_dir / "raw_answers.json"
    with open(raw_path) as f:
        all_answers = [AnswerRecord(**a) for a in json.load(f)]
    log(f"Loaded {len(all_answers)} answers from {raw_path}")

    # Check if scores_gemini.json exists — allow resume
    scores_path = results_dir / "scores_gemini.json"
    already_scored: dict[tuple, ScoreRecord] = {}
    if scores_path.exists():
        with open(scores_path) as f:
            existing = [ScoreRecord(**s) for s in json.load(f)]
        for s in existing:
            if s.accuracy >= 0:  # only count valid scores
                key = (s.question_id, s.model_name, s.temperature, s.top_p)
                already_scored[key] = s
        log(f"Resuming — {len(already_scored)} answers already judged by Gemini")

    all_scores: list[ScoreRecord] = list(already_scored.values())

    # Judge remaining answers
    log("")
    log(f"=== Judging with {model} ===")
    for i, answer in enumerate(all_answers):
        key = (answer.question_id, answer.model_name, answer.temperature, answer.top_p)
        if key in already_scored:
            continue

        q = questions[answer.question_id]
        for attempt in range(5):
            try:
                score = judge_with_gemini(client, q, answer, model=model)
                all_scores.append(score)
                log(f"  [{i+1}/{len(all_answers)}] {answer.model_name} / {answer.question_id} "
                    f"t={answer.temperature} p={answer.top_p} "
                    f"-> acc={score.accuracy} comp={score.completeness} coh={score.coherence}")
                break
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "503" in msg or "UNAVAILABLE" in msg:
                    wait = min(60, 2 ** (attempt + 1))
                    log(f"  Rate/server issue ({e.__class__.__name__}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    log(f"  [{i+1}/{len(all_answers)}] Judge error: {e}")
                    all_scores.append(ScoreRecord(
                        question_id=answer.question_id,
                        model_name=answer.model_name,
                        temperature=answer.temperature,
                        top_p=answer.top_p,
                        accuracy=-1,
                        completeness=-1,
                        coherence=-1,
                        judge_reasoning=f"Error: {e}",
                    ))
                    break

        # Periodic checkpoint save (every 25 answers)
        if (i + 1) % 25 == 0:
            with open(scores_path, "w") as f:
                json.dump([asdict(s) for s in all_scores], f, indent=2)

    # Final save
    with open(scores_path, "w") as f:
        json.dump([asdict(s) for s in all_scores], f, indent=2)
    log(f"Saved {len(all_scores)} Gemini scores to {scores_path}")


if __name__ == "__main__":
    main()
