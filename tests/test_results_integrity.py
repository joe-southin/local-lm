"""Check that every shipped results/*/ directory parses and is self-consistent."""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


def _result_dirs():
    if not RESULTS_ROOT.exists():
        return []
    return [d for d in RESULTS_ROOT.iterdir() if d.is_dir() and (d / "scores.json").exists()]


@pytest.mark.parametrize("results_dir", _result_dirs(), ids=lambda d: d.name)
def test_results_dir_is_valid(results_dir):
    scores = json.load(open(results_dir / "scores.json"))
    answers = json.load(open(results_dir / "raw_answers.json"))

    assert isinstance(scores, list) and scores
    assert isinstance(answers, list) and answers

    # Score schema
    score_fields = {
        "question_id", "model_name", "temperature", "top_p",
        "accuracy", "completeness", "coherence", "judge_reasoning",
    }
    for s in scores:
        assert score_fields <= set(s.keys()), f"score missing fields in {results_dir.name}"

    # Answer schema
    answer_fields = {
        "question_id", "model_name", "temperature", "top_p",
        "answer", "latency_seconds", "tokens_generated",
    }
    for a in answers:
        assert answer_fields <= set(a.keys()), f"answer missing fields in {results_dir.name}"

    # Every score points at an answer we actually recorded
    answer_keys = {
        (a["question_id"], a["model_name"], a["temperature"], a["top_p"])
        for a in answers
    }
    orphan = [
        s for s in scores
        if (s["question_id"], s["model_name"], s["temperature"], s["top_p"]) not in answer_keys
    ]
    assert not orphan, f"{len(orphan)} score(s) in {results_dir.name} have no matching answer"


def test_at_least_one_results_dir_shipped():
    assert _result_dirs(), "no results/*/ directories with scores.json were found"
