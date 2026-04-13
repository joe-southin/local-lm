#!/usr/bin/env python3
"""Local LLM Eval Harness — compares open-weights models against Claude Opus 4.6."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
import httpx
import openai
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EvalQuestion:
    id: str
    category: str
    context: str
    question: str
    reference_answer: str


@dataclass
class ModelConfig:
    name: str
    gguf_path: str | None
    context_size: int | None
    is_local: bool
    max_tokens: int = 1024
    timeout_seconds: float = 120.0


@dataclass
class AnswerRecord:
    question_id: str
    model_name: str
    temperature: float
    top_p: float
    answer: str
    latency_seconds: float
    tokens_generated: int


@dataclass
class ScoreRecord:
    question_id: str
    model_name: str
    temperature: float
    top_p: float
    accuracy: int
    completeness: int
    coherence: int
    judge_reasoning: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARAM_SWEEP = {
    "temperature": [0.0, 0.3, 0.7, 1.0],
    "top_p": [0.5, 0.9, 1.0],
}

QUICK_PARAMS = [{"temperature": 0.3, "top_p": 1.0}]

SYSTEM_PROMPT = (
    "Answer the question based only on the provided context. "
    "Be concise and accurate."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator. Score the following answer on three dimensions, "
    "each from 1 (worst) to 5 (best):\n"
    "- accuracy: Is the answer factually correct given the context?\n"
    "- completeness: Does it address all parts of the question?\n"
    "- coherence: Is it well-structured and easy to understand?\n\n"
    'Respond in JSON only: {"accuracy": N, "completeness": N, "coherence": N, "reasoning": "..."}'
)


# ---------------------------------------------------------------------------
# llama-server lifecycle
# ---------------------------------------------------------------------------

def start_server(gguf_path: str, ctx_size: int = 4096, port: int = 8080) -> subprocess.Popen:
    """Start llama-server and block until /health returns 200."""
    expanded = os.path.expanduser(gguf_path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"GGUF not found: {expanded}")

    proc = subprocess.Popen(
        ["llama-server", "-m", expanded, "-c", str(ctx_size), "--port", str(port), "-ngl", "99"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    import urllib.request
    import urllib.error

    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            if resp.status == 200:
                return proc
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)

    proc.kill()
    raise TimeoutError(f"llama-server health check timed out after 120s for {gguf_path}")


def stop_server(proc: subprocess.Popen) -> None:
    """Send SIGTERM, wait up to 10s, then SIGKILL if needed."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def query_local(
    question: EvalQuestion,
    temperature: float,
    top_p: float,
    port: int = 8080,
    max_tokens: int = 1024,
    timeout_seconds: float = 120.0,
) -> AnswerRecord:
    """Send a chat completion request to the local llama-server."""
    client = openai.OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="not-needed",
        timeout=httpx.Timeout(timeout_seconds, connect=10.0),
    )

    user_msg = f"Context:\n{question.context}\n\nQuestion:\n{question.question}"

    t0 = time.time()
    response = client.chat.completions.create(
        model="local",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - t0

    content = response.choices[0].message.content or ""
    tokens = response.usage.completion_tokens if response.usage else 0

    return AnswerRecord(
        question_id=question.id,
        model_name="",  # filled by caller
        temperature=temperature,
        top_p=top_p,
        answer=content.strip(),
        latency_seconds=round(latency, 2),
        tokens_generated=tokens,
    )


def query_opus(
    question: EvalQuestion,
    temperature: float,
    top_p: float,
) -> AnswerRecord:
    """Send the same prompt to Claude Opus 4.6 via the Anthropic SDK."""
    client = anthropic.Anthropic()

    user_msg = f"Context:\n{question.context}\n\nQuestion:\n{question.question}"

    # Anthropic API doesn't allow both temperature and top_p simultaneously.
    # Pass temperature only (primary parameter); top_p recorded for comparison but not sent.
    t0 = time.time()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    latency = time.time() - t0

    content = response.content[0].text if response.content else ""
    tokens = response.usage.output_tokens

    return AnswerRecord(
        question_id=question.id,
        model_name="Claude Opus 4.6",
        temperature=temperature,
        top_p=top_p,
        answer=content.strip(),
        latency_seconds=round(latency, 2),
        tokens_generated=tokens,
    )


# ---------------------------------------------------------------------------
# Opus-as-judge
# ---------------------------------------------------------------------------

def judge_answer(
    question: EvalQuestion,
    answer: AnswerRecord,
) -> ScoreRecord:
    """Ask Opus to score an answer blind (no model name)."""
    client = anthropic.Anthropic()

    user_msg = (
        f"Context:\n{question.context}\n\n"
        f"Question:\n{question.question}\n\n"
        f"Reference answer:\n{question.reference_answer}\n\n"
        f"Answer to evaluate:\n{answer.answer}"
    )

    for attempt in range(2):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            temperature=0.0,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text if response.content else ""
        try:
            # Handle possible markdown code fences
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


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(scores: list[ScoreRecord], answers: list[AnswerRecord]) -> list[dict]:
    """Aggregate scores by model and category. Returns list of dicts for CSV."""
    from collections import defaultdict

    # Build latency lookup
    latency_map: dict[tuple, float] = {}
    for a in answers:
        latency_map[(a.question_id, a.model_name, a.temperature, a.top_p)] = a.latency_seconds

    # Group scores
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for s in scores:
        if s.accuracy < 0:
            continue  # skip parse failures
        # Find the question category
        q_cat = s.question_id.split("_")[0]
        groups[(s.model_name, q_cat)].append(s)

    rows = []
    for (model, category) in sorted(groups.keys()):
        sc = groups[(model, category)]
        latencies = [
            latency_map.get((s.question_id, s.model_name, s.temperature, s.top_p), 0)
            for s in sc
        ]
        rows.append({
            "model": model,
            "category": category,
            "avg_accuracy": round(sum(s.accuracy for s in sc) / len(sc), 2),
            "avg_completeness": round(sum(s.completeness for s in sc) / len(sc), 2),
            "avg_coherence": round(sum(s.coherence for s in sc) / len(sc), 2),
            "avg_latency": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "n": len(sc),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Local LLM Eval Harness")
    parser.add_argument("--quick", action="store_true", help="Only run default params (temp=0.0, top_p=1.0)")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to run")
    parser.add_argument("--output-dir", type=str, default="./results", help="Where to write results")
    parser.add_argument("--category", type=str, default=None, help="Comma-separated question categories to run (e.g. cot,reasoning)")
    parser.add_argument("--questions", type=str, default=None, help="Comma-separated question IDs to run (e.g. cot_02,cot_03)")
    parser.add_argument("--skip-questions", type=str, default=None, help="Comma-separated question IDs to skip")
    args = parser.parse_args()

    # Load questions
    questions_path = Path(__file__).parent / "eval_questions.json"
    with open(questions_path) as f:
        questions = [EvalQuestion(**q) for q in json.load(f)]

    # Filter categories if requested
    if args.category:
        requested_cats = {c.strip().lower() for c in args.category.split(",")}
        questions = [q for q in questions if q.category in requested_cats]
        if not questions:
            log(f"No questions matching categories: {requested_cats}")
            sys.exit(1)

    # Filter by question IDs (include) if requested
    if args.questions:
        requested_ids = {q.strip() for q in args.questions.split(",")}
        questions = [q for q in questions if q.id in requested_ids]
        if not questions:
            log(f"No questions matching IDs: {requested_ids}")
            sys.exit(1)

    # Skip question IDs if requested
    if args.skip_questions:
        skip_ids = {q.strip() for q in args.skip_questions.split(",")}
        questions = [q for q in questions if q.id not in skip_ids]

    log(f"Loaded {len(questions)} eval questions")

    # Load model configs
    models_path = Path(__file__).parent / "models.json"
    with open(models_path) as f:
        all_models = [ModelConfig(**m) for m in json.load(f)]

    # Filter models if requested
    if args.models:
        requested = {n.strip().lower() for n in args.models.split(",")}
        all_models = [m for m in all_models if m.name.lower() in requested]
        if not all_models:
            log(f"No matching models found. Available: {[m.name for m in all_models]}")
            sys.exit(1)

    # Build param combos
    if args.quick:
        param_combos = QUICK_PARAMS
    else:
        param_combos = [
            {"temperature": t, "top_p": p}
            for t in PARAM_SWEEP["temperature"]
            for p in PARAM_SWEEP["top_p"]
        ]

    log(f"Models: {[m.name for m in all_models]}")
    log(f"Param combos: {len(param_combos)}")
    log(f"Total inference calls: {len(all_models) * len(questions) * len(param_combos)}")
    log("")

    # Collect answers
    all_answers: list[AnswerRecord] = []

    # Run local models
    local_models = [m for m in all_models if m.is_local]
    for model in local_models:
        log(f"--- {model.name} ---")
        try:
            log(f"  Starting llama-server for {model.name}...")
            proc = start_server(model.gguf_path, model.context_size)
            log(f"  Server ready")
        except (FileNotFoundError, TimeoutError) as e:
            log(f"  SKIP: {e}")
            continue

        try:
            for params in param_combos:
                for q in questions:
                    try:
                        record = query_local(
                            q, params["temperature"], params["top_p"],
                            max_tokens=model.max_tokens,
                            timeout_seconds=model.timeout_seconds,
                        )
                        record.model_name = model.name
                        all_answers.append(record)
                        log(f"  [{q.id}] temp={params['temperature']} top_p={params['top_p']} "
                            f"-> {record.tokens_generated} tok, {record.latency_seconds}s")
                    except Exception as e:
                        log(f"  [{q.id}] TIMEOUT/ERROR: {e}")
                        all_answers.append(AnswerRecord(
                            question_id=q.id,
                            model_name=model.name,
                            temperature=params["temperature"],
                            top_p=params["top_p"],
                            answer="[TIMEOUT]",
                            latency_seconds=model.timeout_seconds,
                            tokens_generated=0,
                        ))
        finally:
            log(f"  Stopping server...")
            stop_server(proc)
            # Give memory a moment to free up
            time.sleep(3)
        log("")

    # Run Opus
    opus_models = [m for m in all_models if not m.is_local]
    for model in opus_models:
        log(f"--- {model.name} ---")
        for params in param_combos:
            for q in questions:
                for attempt in range(3):
                    try:
                        record = query_opus(q, params["temperature"], params["top_p"])
                        all_answers.append(record)
                        log(f"  [{q.id}] temp={params['temperature']} top_p={params['top_p']} "
                            f"-> {record.tokens_generated} tok, {record.latency_seconds}s")
                        break
                    except anthropic.RateLimitError:
                        wait = 2 ** (attempt + 1)
                        log(f"  [{q.id}] Rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                    except Exception as e:
                        log(f"  [{q.id}] ERROR: {e}")
                        all_answers.append(AnswerRecord(
                            question_id=q.id,
                            model_name=model.name,
                            temperature=params["temperature"],
                            top_p=params["top_p"],
                            answer=f"[ERROR: {e}]",
                            latency_seconds=0,
                            tokens_generated=0,
                        ))
                        break
        log("")

    # Save raw answers
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_answers.json"
    with open(raw_path, "w") as f:
        json.dump([asdict(a) for a in all_answers], f, indent=2)
    log(f"Saved {len(all_answers)} answers to {raw_path}")

    # Judge all answers
    log("")
    log("=== Judging answers with Opus ===")
    all_scores: list[ScoreRecord] = []
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
    scores_path = output_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump([asdict(s) for s in all_scores], f, indent=2)
    log(f"Saved {len(all_scores)} scores to {scores_path}")

    # Generate summary
    summary_rows = generate_summary(all_scores, all_answers)

    csv_path = output_dir / "summary.csv"
    if summary_rows:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        log(f"Saved summary to {csv_path}")

    # Print summary table
    log("")
    log("=== Summary ===")
    if summary_rows:
        log(tabulate(summary_rows, headers="keys", tablefmt="simple"))
    else:
        log("No valid scores to summarise.")


if __name__ == "__main__":
    main()
