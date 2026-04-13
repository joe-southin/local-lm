"""Smoke tests for pure-function logic in eval_harness.py.

These tests never make network calls and never start llama-server — they only
exercise dataclasses and the offline aggregation logic.
"""
from dataclasses import asdict

import pytest

from eval_harness import (
    AnswerRecord,
    EvalQuestion,
    ModelConfig,
    ScoreRecord,
    generate_summary,
)


def _q(qid, cat):
    return EvalQuestion(
        id=qid, category=cat, context="ctx", question="q?", reference_answer="a",
    )


def _a(qid, model, temp=0.0, top_p=1.0, latency=1.5, tokens=42):
    return AnswerRecord(
        question_id=qid, model_name=model, temperature=temp, top_p=top_p,
        answer="answer", latency_seconds=latency, tokens_generated=tokens,
    )


def _s(qid, model, acc=4, comp=4, coh=4, temp=0.0, top_p=1.0):
    return ScoreRecord(
        question_id=qid, model_name=model, temperature=temp, top_p=top_p,
        accuracy=acc, completeness=comp, coherence=coh, judge_reasoning="ok",
    )


# ---------------------------------------------------------------------------
# Dataclass round-trip
# ---------------------------------------------------------------------------

def test_dataclasses_round_trip_through_asdict():
    q = _q("factual_01", "factual")
    a = _a("factual_01", "Qwen 2.5 7B")
    s = _s("factual_01", "Qwen 2.5 7B")
    for obj, cls in [(q, EvalQuestion), (a, AnswerRecord), (s, ScoreRecord)]:
        assert cls(**asdict(obj)) == obj


def test_modelconfig_defaults():
    m = ModelConfig(name="x", gguf_path=None, context_size=None, is_local=False)
    assert m.max_tokens == 1024
    assert m.timeout_seconds == 120.0


# ---------------------------------------------------------------------------
# generate_summary()
# ---------------------------------------------------------------------------

def test_generate_summary_groups_by_model_and_category():
    answers = [
        _a("factual_01", "Qwen"),
        _a("factual_02", "Qwen"),
        _a("reasoning_01", "Qwen"),
        _a("factual_01", "Llama"),
    ]
    scores = [
        _s("factual_01", "Qwen", acc=5, comp=5, coh=5),
        _s("factual_02", "Qwen", acc=3, comp=3, coh=3),
        _s("reasoning_01", "Qwen", acc=4, comp=4, coh=4),
        _s("factual_01", "Llama", acc=2, comp=2, coh=2),
    ]
    rows = generate_summary(scores, answers)

    by_key = {(r["model"], r["category"]): r for r in rows}
    assert set(by_key) == {
        ("Qwen", "factual"),
        ("Qwen", "reasoning"),
        ("Llama", "factual"),
    }

    # Qwen factual averages (5+3)/2 = 4.0 on every dimension
    qf = by_key[("Qwen", "factual")]
    assert qf["avg_accuracy"] == 4.0
    assert qf["avg_completeness"] == 4.0
    assert qf["avg_coherence"] == 4.0
    assert qf["n"] == 2

    # Qwen reasoning has one data point at 4/4/4
    qr = by_key[("Qwen", "reasoning")]
    assert qr["avg_accuracy"] == 4.0
    assert qr["n"] == 1


def test_generate_summary_skips_parse_failures():
    answers = [_a("factual_01", "Qwen"), _a("factual_02", "Qwen")]
    scores = [
        _s("factual_01", "Qwen", acc=5, comp=5, coh=5),
        _s("factual_02", "Qwen", acc=-1, comp=-1, coh=-1),  # parse failure sentinel
    ]
    rows = generate_summary(scores, answers)
    assert len(rows) == 1
    assert rows[0]["n"] == 1
    assert rows[0]["avg_accuracy"] == 5.0


def test_generate_summary_handles_latency_lookup():
    answers = [_a("factual_01", "Qwen", latency=2.0), _a("factual_02", "Qwen", latency=4.0)]
    scores = [
        _s("factual_01", "Qwen"),
        _s("factual_02", "Qwen"),
    ]
    rows = generate_summary(scores, answers)
    assert len(rows) == 1
    assert rows[0]["avg_latency"] == 3.0


def test_generate_summary_empty_inputs():
    assert generate_summary([], []) == []
