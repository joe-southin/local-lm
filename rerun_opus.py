#!/usr/bin/env python3
"""Rerun just Opus inference + rejudge all answers."""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import anthropic

from eval_harness import (
    AnswerRecord, EvalQuestion, ScoreRecord,
    SYSTEM_PROMPT, judge_answer, generate_summary, log,
)
from tabulate import tabulate

def main():
    os.environ.setdefault("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic()

    # Load questions
    questions = [EvalQuestion(**q) for q in json.load(open("eval_questions.json"))]

    # Load existing answers
    raw_path = Path("results/raw_answers.json")
    all_answers = [AnswerRecord(**a) for a in json.load(open(raw_path))]

    # Remove old Opus records
    all_answers = [a for a in all_answers if a.model_name != "Claude Opus 4.6"]
    log(f"Kept {len(all_answers)} local model answers")

    # Run Opus inference (temperature only, no top_p)
    log("")
    log("--- Claude Opus 4.6 ---")
    for q in questions:
        for attempt in range(3):
            try:
                user_msg = f"Context:\n{q.context}\n\nQuestion:\n{q.question}"
                t0 = time.time()
                response = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=1024,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                latency = time.time() - t0
                content = response.content[0].text if response.content else ""
                tokens = response.usage.output_tokens

                record = AnswerRecord(
                    question_id=q.id,
                    model_name="Claude Opus 4.6",
                    temperature=0.0,
                    top_p=1.0,
                    answer=content.strip(),
                    latency_seconds=round(latency, 2),
                    tokens_generated=tokens,
                )
                all_answers.append(record)
                log(f"  [{q.id}] -> {tokens} tok, {latency:.1f}s")
                break
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                log(f"  [{q.id}] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                log(f"  [{q.id}] ERROR: {e}")
                all_answers.append(AnswerRecord(
                    question_id=q.id, model_name="Claude Opus 4.6",
                    temperature=0.0, top_p=1.0,
                    answer=f"[ERROR: {e}]", latency_seconds=0, tokens_generated=0,
                ))
                break

    # Save updated raw answers
    with open(raw_path, "w") as f:
        json.dump([asdict(a) for a in all_answers], f, indent=2)
    log(f"\nSaved {len(all_answers)} answers to {raw_path}")

    # Rejudge all answers
    log("\n=== Judging all answers with Opus ===")
    all_scores = []
    for i, answer in enumerate(all_answers):
        q = next(q for q in questions if q.id == answer.question_id)
        for attempt in range(3):
            try:
                score = judge_answer(q, answer)
                all_scores.append(score)
                log(f"  [{i+1}/{len(all_answers)}] {answer.model_name} / {answer.question_id} "
                    f"-> acc={score.accuracy} comp={score.completeness} coh={score.coherence}")
                break
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                log(f"  Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                log(f"  [{i+1}/{len(all_answers)}] Judge error: {e}")
                all_scores.append(ScoreRecord(
                    question_id=answer.question_id, model_name=answer.model_name,
                    temperature=answer.temperature, top_p=answer.top_p,
                    accuracy=-1, completeness=-1, coherence=-1,
                    judge_reasoning=f"Error: {e}",
                ))
                break

    # Save scores
    scores_path = Path("results/scores.json")
    with open(scores_path, "w") as f:
        json.dump([asdict(s) for s in all_scores], f, indent=2)
    log(f"Saved {len(all_scores)} scores to {scores_path}")

    # Generate summary
    summary_rows = generate_summary(all_scores, all_answers)
    if summary_rows:
        import csv
        csv_path = Path("results/summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        log(f"Saved summary to {csv_path}")

    log("\n=== Summary ===")
    if summary_rows:
        log(tabulate(summary_rows, headers="keys", tablefmt="simple"))
    else:
        log("No valid scores to summarise.")


if __name__ == "__main__":
    main()
