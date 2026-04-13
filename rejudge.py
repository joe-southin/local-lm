#!/usr/bin/env python3
"""Re-run Opus-as-judge on existing raw_answers.json."""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Reuse data models and judge function from the harness
from eval_harness import (
    AnswerRecord,
    EvalQuestion,
    ScoreRecord,
    generate_summary,
    judge_answer,
    log,
)

import anthropic


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./results/quant_comparison")

    # Load questions
    with open(Path(__file__).parent / "eval_questions.json") as f:
        questions = {q["id"]: EvalQuestion(**q) for q in json.load(f)}

    # Load raw answers
    raw_path = results_dir / "raw_answers.json"
    with open(raw_path) as f:
        all_answers = [AnswerRecord(**a) for a in json.load(f)]
    log(f"Loaded {len(all_answers)} answers from {raw_path}")

    # Judge all answers
    log("")
    log("=== Judging answers with Opus ===")
    all_scores: list[ScoreRecord] = []
    for i, answer in enumerate(all_answers):
        q = questions[answer.question_id]
        for attempt in range(3):
            try:
                score = judge_answer(q, answer)
                all_scores.append(score)
                log(f"  [{i+1}/{len(all_answers)}] {answer.model_name} / {answer.question_id} "
                    f"t={answer.temperature} p={answer.top_p} "
                    f"-> acc={score.accuracy} comp={score.completeness} coh={score.coherence}")
                break
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                log(f"  Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
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

    # Save scores
    scores_path = results_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump([asdict(s) for s in all_scores], f, indent=2)
    log(f"Saved {len(all_scores)} scores to {scores_path}")

    # Generate and save summary
    summary_rows = generate_summary(all_scores, all_answers)
    if summary_rows:
        import csv
        csv_path = results_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        log(f"Saved summary to {csv_path}")

    # Print summary
    log("")
    log("=== Summary ===")
    if summary_rows:
        from tabulate import tabulate
        log(tabulate(summary_rows, headers="keys", tablefmt="simple"))
    else:
        log("No valid scores to summarise.")


if __name__ == "__main__":
    main()
