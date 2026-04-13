#!/usr/bin/env python3
"""Compare Opus and Gemini judges on the same answers.

Loads scores.json (Opus) and scores_gemini.json from a results directory and
produces:
- Per-model averages by each judge
- Inter-judge correlation (Pearson per dimension)
- Self-preference check (does Opus rate Opus higher than Gemini does?)
- Disagreement cases where judges differ by 2+ points on any dimension

Usage:
    python multi_judge_analysis.py [results_dir]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_both_judges(results_dir: Path):
    with open(results_dir / "scores.json") as f:
        opus = json.load(f)
    gemini_path = results_dir / "scores_gemini.json"
    if not gemini_path.exists():
        print(f"No Gemini scores at {gemini_path}")
        sys.exit(1)
    with open(gemini_path) as f:
        gemini = json.load(f)

    # Build key -> score lookups
    def key(s):
        return (s["question_id"], s["model_name"], s["temperature"], s["top_p"])

    opus_map = {key(s): s for s in opus if s["accuracy"] >= 0}
    gemini_map = {key(s): s for s in gemini if s["accuracy"] >= 0}

    # Intersection — only answers judged by both
    shared = set(opus_map) & set(gemini_map)
    print(f"Opus valid: {len(opus_map)}, Gemini valid: {len(gemini_map)}, shared: {len(shared)}")
    return opus_map, gemini_map, shared


def per_model_comparison(opus_map, gemini_map, shared):
    """Per-model averages under each judge."""
    opus_by_model = defaultdict(lambda: defaultdict(list))
    gemini_by_model = defaultdict(lambda: defaultdict(list))

    for k in shared:
        o = opus_map[k]
        g = gemini_map[k]
        model = o["model_name"]
        for dim in ("accuracy", "completeness", "coherence"):
            opus_by_model[model][dim].append(o[dim])
            gemini_by_model[model][dim].append(g[dim])

    print("\n=== Per-Model Averages (Opus vs Gemini) ===\n")
    print(f"{'Model':<28} {'Dim':<14} {'Opus':>6} {'Gemini':>7} {'Delta':>7}")
    print("-" * 70)
    for model in sorted(opus_by_model):
        for dim in ("accuracy", "completeness", "coherence"):
            o_avg = np.mean(opus_by_model[model][dim])
            g_avg = np.mean(gemini_by_model[model][dim])
            delta = o_avg - g_avg
            print(f"{model:<28} {dim:<14} {o_avg:>6.2f} {g_avg:>7.2f} {delta:>+7.2f}")
        print()


def inter_judge_correlation(opus_map, gemini_map, shared):
    """Pearson correlation per dimension across all shared answers."""
    print("=== Inter-Judge Agreement (Pearson r) ===\n")
    for dim in ("accuracy", "completeness", "coherence"):
        o_vec = np.array([opus_map[k][dim] for k in shared])
        g_vec = np.array([gemini_map[k][dim] for k in shared])
        if o_vec.std() == 0 or g_vec.std() == 0:
            print(f"  {dim}: insufficient variance")
            continue
        r = np.corrcoef(o_vec, g_vec)[0, 1]
        # Mean absolute difference
        mad = np.mean(np.abs(o_vec - g_vec))
        # Exact agreement rate
        exact = np.mean(o_vec == g_vec)
        # Agreement within 1 point
        within1 = np.mean(np.abs(o_vec - g_vec) <= 1)
        print(f"  {dim:<14}: r={r:.3f}  MAD={mad:.2f}  exact={exact:.0%}  within1={within1:.0%}")
    print()


def self_preference_check(opus_map, gemini_map, shared):
    """Does Opus rate Opus's answers higher than Gemini does?"""
    print("=== Self-Preference Check (Opus judging Opus vs Gemini judging Opus) ===\n")
    opus_on_opus = {dim: [] for dim in ("accuracy", "completeness", "coherence")}
    gemini_on_opus = {dim: [] for dim in ("accuracy", "completeness", "coherence")}

    for k in shared:
        if opus_map[k]["model_name"] != "Claude Opus 4.6":
            continue
        for dim in ("accuracy", "completeness", "coherence"):
            opus_on_opus[dim].append(opus_map[k][dim])
            gemini_on_opus[dim].append(gemini_map[k][dim])

    n = len(opus_on_opus["accuracy"])
    print(f"  n={n} Opus answers")
    for dim in ("accuracy", "completeness", "coherence"):
        o = np.mean(opus_on_opus[dim])
        g = np.mean(gemini_on_opus[dim])
        print(f"  {dim:<14}: Opus->{o:.2f}  Gemini->{g:.2f}  bias={o-g:+.2f}")
    print()


def disagreement_cases(opus_map, gemini_map, shared, threshold=2):
    """Find answers where judges disagree by threshold+ points on any dimension."""
    print(f"=== Disagreement Cases (|delta| >= {threshold} on any dimension) ===\n")
    cases = []
    for k in shared:
        o = opus_map[k]
        g = gemini_map[k]
        max_delta = max(
            abs(o[dim] - g[dim]) for dim in ("accuracy", "completeness", "coherence")
        )
        if max_delta >= threshold:
            cases.append((k, o, g, max_delta))

    cases.sort(key=lambda x: -x[3])
    print(f"  {len(cases)} cases out of {len(shared)} ({len(cases)/len(shared):.1%})")

    for (k, o, g, delta) in cases[:10]:
        qid, model, temp, top_p = k
        print(f"\n  {model} / {qid} (t={temp}, p={top_p})  max_delta={delta}")
        print(f"    Opus:   acc={o['accuracy']} comp={o['completeness']} coh={o['coherence']}")
        print(f"    Gemini: acc={g['accuracy']} comp={g['completeness']} coh={g['coherence']}")
        print(f"    Opus reason:   {o['judge_reasoning'][:160]}")
        print(f"    Gemini reason: {g['judge_reasoning'][:160]}")
    print()


def model_ranking_comparison(opus_map, gemini_map, shared):
    """Does the ranking of models change by judge?"""
    print("=== Model Ranking by Judge (avg composite) ===\n")
    opus_avg = defaultdict(list)
    gemini_avg = defaultdict(list)
    for k in shared:
        o = opus_map[k]
        g = gemini_map[k]
        opus_avg[o["model_name"]].append(
            (o["accuracy"] + o["completeness"] + o["coherence"]) / 3
        )
        gemini_avg[g["model_name"]].append(
            (g["accuracy"] + g["completeness"] + g["coherence"]) / 3
        )

    opus_rank = sorted(opus_avg.items(), key=lambda kv: -np.mean(kv[1]))
    gemini_rank = sorted(gemini_avg.items(), key=lambda kv: -np.mean(kv[1]))

    print(f"  {'Opus ranking':<40} {'Gemini ranking':<40}")
    print("  " + "-" * 80)
    for i in range(max(len(opus_rank), len(gemini_rank))):
        o_str = f"{i+1}. {opus_rank[i][0]} ({np.mean(opus_rank[i][1]):.2f})" if i < len(opus_rank) else ""
        g_str = f"{i+1}. {gemini_rank[i][0]} ({np.mean(gemini_rank[i][1]):.2f})" if i < len(gemini_rank) else ""
        print(f"  {o_str:<40} {g_str:<40}")
    print()


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./results/full_sweep")
    print(f"Analysing {results_dir}\n")

    opus_map, gemini_map, shared = load_both_judges(results_dir)
    per_model_comparison(opus_map, gemini_map, shared)
    inter_judge_correlation(opus_map, gemini_map, shared)
    self_preference_check(opus_map, gemini_map, shared)
    model_ranking_comparison(opus_map, gemini_map, shared)
    disagreement_cases(opus_map, gemini_map, shared, threshold=2)


if __name__ == "__main__":
    main()
