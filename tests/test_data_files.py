"""Schema tests for the JSON data files shipped with the repo."""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# eval_questions.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def questions():
    with open(REPO_ROOT / "eval_questions.json") as f:
        return json.load(f)


def test_questions_non_empty(questions):
    assert len(questions) >= 12


def test_question_ids_unique(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"


def test_question_schema(questions):
    required = {"id", "category", "context", "question", "reference_answer"}
    for q in questions:
        missing = required - set(q.keys())
        assert not missing, f"{q.get('id', '?')} missing fields: {missing}"
        for field in required:
            assert isinstance(q[field], str) and q[field].strip(), (
                f"{q['id']}: field {field!r} is empty or not a string"
            )


def test_question_categories_recognised(questions):
    recognised = {"factual", "reasoning", "synthesis", "cot"}
    for q in questions:
        assert q["category"] in recognised, (
            f"{q['id']}: unknown category {q['category']!r}"
        )


def test_question_id_prefix_matches_category(questions):
    for q in questions:
        prefix = q["id"].split("_", 1)[0]
        assert prefix == q["category"], (
            f"{q['id']}: id prefix {prefix!r} does not match category {q['category']!r}"
        )


# ---------------------------------------------------------------------------
# models.json
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def models():
    with open(REPO_ROOT / "models.json") as f:
        return json.load(f)


def test_models_non_empty(models):
    assert len(models) >= 2


def test_model_names_unique(models):
    names = [m["name"] for m in models]
    assert len(names) == len(set(names))


def test_model_schema(models):
    for m in models:
        assert isinstance(m["name"], str) and m["name"]
        assert isinstance(m["is_local"], bool)
        if m["is_local"]:
            assert isinstance(m["gguf_path"], str) and m["gguf_path"].endswith(".gguf")
            assert isinstance(m["context_size"], int) and m["context_size"] > 0
        else:
            assert m["gguf_path"] is None
            assert m["context_size"] is None


def test_exactly_one_frontier_baseline(models):
    """Paper design: exactly one non-local frontier model acts as the baseline + judge."""
    non_local = [m for m in models if not m["is_local"]]
    assert len(non_local) == 1, f"expected 1 frontier model, found {len(non_local)}"
