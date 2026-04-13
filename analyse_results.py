#!/usr/bin/env python3
"""Analyse eval results and produce charts + tables."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_data(results_dir):
    scores = json.load(open(Path(results_dir) / "scores.json"))
    answers = json.load(open(Path(results_dir) / "raw_answers.json"))
    return scores, answers


def plot_scores_by_model(scores, out_dir):
    """Bar chart: average score by model, grouped by category."""
    categories = ["factual", "reasoning", "synthesis"]
    data = defaultdict(lambda: defaultdict(list))

    for s in scores:
        cat = s["question_id"].split("_")[0]
        composite = (s["accuracy"] + s["completeness"] + s["coherence"]) / 3
        data[s["model_name"]][cat].append(composite)

    models = sorted(data.keys())
    x = np.arange(len(categories))
    width = 0.18
    offsets = np.arange(len(models)) - (len(models) - 1) / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#4A90D9", "#E8913A", "#50B88C", "#D94A6B"]

    for i, model in enumerate(models):
        means = [np.mean(data[model].get(cat, [0])) for cat in categories]
        bars = ax.bar(x + offsets[i] * width, means, width, label=model, color=colours[i])
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Average Composite Score (1-5)")
    ax.set_title("Model Quality by Category (Full Sweep)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in categories])
    ax.set_ylim(0, 5.8)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "scores_by_model.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir}/scores_by_model.png")


def plot_quality_vs_latency(scores, answers, out_dir):
    """Scatter plot: quality vs latency per model."""
    model_data = defaultdict(lambda: {"scores": [], "latencies": []})

    for s in scores:
        composite = (s["accuracy"] + s["completeness"] + s["coherence"]) / 3
        model_data[s["model_name"]]["scores"].append(composite)

    for a in answers:
        model_data[a["model_name"]]["latencies"].append(a["latency_seconds"])

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["#4A90D9", "#E8913A", "#50B88C", "#D94A6B"]
    markers = ["o", "s", "D", "^"]

    for i, (model, d) in enumerate(sorted(model_data.items())):
        avg_score = np.mean(d["scores"])
        avg_latency = np.mean(d["latencies"])
        ax.scatter(avg_latency, avg_score, s=200, color=colours[i],
                   marker=markers[i], label=model, zorder=5)
        ax.annotate(model, (avg_latency, avg_score),
                    textcoords="offset points", xytext=(10, -5), fontsize=8)

    ax.set_xlabel("Average Latency (seconds)")
    ax.set_ylabel("Average Composite Score (1-5)")
    ax.set_title("Quality vs Latency Trade-off (Full Sweep)")
    ax.set_ylim(3.5, 5.3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "quality_vs_latency.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir}/quality_vs_latency.png")


def plot_dimension_breakdown(scores, out_dir):
    """Heatmap: model x dimension scores."""
    models = sorted(set(s["model_name"] for s in scores))
    dimensions = ["accuracy", "completeness", "coherence"]

    data = defaultdict(lambda: defaultdict(list))
    for s in scores:
        for dim in dimensions:
            data[s["model_name"]][dim].append(s[dim])

    matrix = []
    for model in models:
        row = [np.mean(data[model][dim]) for dim in dimensions]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=3, vmax=5, aspect="auto")

    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels([d.title() for d in dimensions])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(dimensions)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if matrix[i][j] < 4 else "black")

    plt.colorbar(im, ax=ax, label="Average Score")
    ax.set_title("Score Breakdown by Model and Dimension (Full Sweep)")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "dimension_breakdown.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir}/dimension_breakdown.png")


def plot_token_counts(answers, out_dir):
    """Bar chart: average tokens generated per model per category."""
    data = defaultdict(lambda: defaultdict(list))
    for a in answers:
        cat = a["question_id"].split("_")[0]
        data[a["model_name"]][cat].append(a["tokens_generated"])

    models = sorted(data.keys())
    categories = ["factual", "reasoning", "synthesis"]
    x = np.arange(len(categories))
    width = 0.18
    offsets = np.arange(len(models)) - (len(models) - 1) / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#4A90D9", "#E8913A", "#50B88C", "#D94A6B"]

    for i, model in enumerate(models):
        means = [np.mean(data[model].get(cat, [0])) for cat in categories]
        bars = ax.bar(x + offsets[i] * width, means, width, label=model, color=colours[i])
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Average Tokens Generated")
    ax.set_title("Response Length by Model and Category (Full Sweep)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in categories])
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "token_counts.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir}/token_counts.png")


def plot_score_vs_temperature(scores, out_dir):
    """Line chart: composite score vs temperature for each model."""
    data = defaultdict(lambda: defaultdict(list))

    for s in scores:
        composite = (s["accuracy"] + s["completeness"] + s["coherence"]) / 3
        data[s["model_name"]][s["temperature"]].append(composite)

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#4A90D9", "#E8913A", "#50B88C", "#D94A6B"]
    markers = ["o", "s", "D", "^"]

    for i, (model, temps) in enumerate(sorted(data.items())):
        temp_vals = sorted(temps.keys())
        means = [np.mean(temps[t]) for t in temp_vals]
        sems = [np.std(temps[t]) / np.sqrt(len(temps[t])) for t in temp_vals]
        ax.errorbar(temp_vals, means, yerr=sems, label=model,
                    color=colours[i], marker=markers[i], linewidth=2,
                    capsize=4, markersize=8)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Average Composite Score (1-5)")
    ax.set_title("Score vs Temperature (error bars = SEM)")
    ax.set_xticks([0.0, 0.3, 0.7, 1.0])
    ax.set_ylim(3.0, 5.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "score_vs_temperature.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir}/score_vs_temperature.png")


def plot_score_vs_top_p(scores, out_dir):
    """Line chart: composite score vs top_p for each model (at temp > 0 only)."""
    data = defaultdict(lambda: defaultdict(list))

    for s in scores:
        if s["temperature"] == 0.0:
            continue  # top_p has no effect at temp=0
        composite = (s["accuracy"] + s["completeness"] + s["coherence"]) / 3
        data[s["model_name"]][s["top_p"]].append(composite)

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#4A90D9", "#E8913A", "#50B88C", "#D94A6B"]
    markers = ["o", "s", "D", "^"]

    for i, (model, top_ps) in enumerate(sorted(data.items())):
        p_vals = sorted(top_ps.keys())
        means = [np.mean(top_ps[p]) for p in p_vals]
        sems = [np.std(top_ps[p]) / np.sqrt(len(top_ps[p])) for p in p_vals]
        ax.errorbar(p_vals, means, yerr=sems, label=model,
                    color=colours[i], marker=markers[i], linewidth=2,
                    capsize=4, markersize=8)

    ax.set_xlabel("Top-p")
    ax.set_ylabel("Average Composite Score (1-5)")
    ax.set_title("Score vs Top-p (temperature > 0 only, error bars = SEM)")
    ax.set_xticks([0.5, 0.9, 1.0])
    ax.set_ylim(3.0, 5.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "score_vs_top_p.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir}/score_vs_top_p.png")


def plot_best_params_heatmap(scores, out_dir):
    """Heatmap: best temperature x top_p combo per model."""
    models = sorted(set(s["model_name"] for s in scores))
    temps = sorted(set(s["temperature"] for s in scores))
    top_ps = sorted(set(s["top_p"] for s in scores))

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        matrix = np.zeros((len(temps), len(top_ps)))
        for s in scores:
            if s["model_name"] != model:
                continue
            ti = temps.index(s["temperature"])
            pi = top_ps.index(s["top_p"])
            composite = (s["accuracy"] + s["completeness"] + s["coherence"]) / 3
            matrix[ti][pi] += composite

        # Average over questions (12 per cell)
        count_matrix = np.zeros_like(matrix)
        for s in scores:
            if s["model_name"] != model:
                continue
            ti = temps.index(s["temperature"])
            pi = top_ps.index(s["top_p"])
            count_matrix[ti][pi] += 1
        matrix = np.divide(matrix, count_matrix, where=count_matrix > 0)

        im = axes[idx].imshow(matrix, cmap="RdYlGn", vmin=3.5, vmax=5.0, aspect="auto")
        axes[idx].set_xticks(range(len(top_ps)))
        axes[idx].set_xticklabels([f"{p}" for p in top_ps])
        axes[idx].set_yticks(range(len(temps)))
        axes[idx].set_yticklabels([f"{t}" for t in temps])
        axes[idx].set_xlabel("Top-p")
        if idx == 0:
            axes[idx].set_ylabel("Temperature")
        axes[idx].set_title(model, fontsize=10)

        for i in range(len(temps)):
            for j in range(len(top_ps)):
                axes[idx].text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                               fontsize=9, fontweight="bold",
                               color="white" if matrix[i][j] < 4.2 else "black")

    fig.suptitle("Average Score by Parameter Combination", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "param_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir}/param_heatmap.png")


def print_best_worst(scores, answers):
    """Print best and worst examples from local models."""
    local_scores = [s for s in scores if s["model_name"] != "Claude Opus 4.6"]

    local_scores.sort(key=lambda s: s["accuracy"] + s["completeness"] + s["coherence"])
    print("\n=== Worst Local Model Answers ===\n")
    seen = set()
    count = 0
    for s in local_scores:
        key = (s["model_name"], s["question_id"])
        if key in seen:
            continue
        seen.add(key)
        a = next((a for a in answers
                  if a["question_id"] == s["question_id"]
                  and a["model_name"] == s["model_name"]
                  and a["temperature"] == s["temperature"]
                  and a["top_p"] == s["top_p"]), None)
        total = s["accuracy"] + s["completeness"] + s["coherence"]
        print(f"  {s['model_name']} / {s['question_id']} (t={s['temperature']}, p={s['top_p']}) — {total}/15")
        print(f"    Scores: acc={s['accuracy']} comp={s['completeness']} coh={s['coherence']}")
        print(f"    Judge: {s['judge_reasoning'][:200]}")
        print()
        count += 1
        if count >= 5:
            break

    hard = [s for s in local_scores if "reasoning" in s["question_id"] or "synthesis" in s["question_id"]]
    hard.sort(key=lambda s: s["accuracy"] + s["completeness"] + s["coherence"], reverse=True)
    print("=== Best Local Model Answers (Reasoning/Synthesis) ===\n")
    seen = set()
    count = 0
    for s in hard:
        key = (s["model_name"], s["question_id"])
        if key in seen:
            continue
        seen.add(key)
        a = next((a for a in answers
                  if a["question_id"] == s["question_id"]
                  and a["model_name"] == s["model_name"]
                  and a["temperature"] == s["temperature"]
                  and a["top_p"] == s["top_p"]), None)
        total = s["accuracy"] + s["completeness"] + s["coherence"]
        tok = a["tokens_generated"] if a else "?"
        lat = a["latency_seconds"] if a else "?"
        print(f"  {s['model_name']} / {s['question_id']} (t={s['temperature']}, p={s['top_p']}) — {total}/15")
        print(f"    Scores: acc={s['accuracy']} comp={s['completeness']} coh={s['coherence']}")
        print(f"    Tokens: {tok}, Latency: {lat}s")
        print()
        count += 1
        if count >= 5:
            break

    # Best params per model
    print("=== Best Parameter Settings per Model ===\n")
    model_params = defaultdict(lambda: defaultdict(list))
    for s in scores:
        key = (s["temperature"], s["top_p"])
        composite = (s["accuracy"] + s["completeness"] + s["coherence"]) / 3
        model_params[s["model_name"]][key].append(composite)

    for model in sorted(model_params.keys()):
        best_key = max(model_params[model].keys(),
                       key=lambda k: np.mean(model_params[model][k]))
        best_score = np.mean(model_params[model][best_key])
        worst_key = min(model_params[model].keys(),
                        key=lambda k: np.mean(model_params[model][k]))
        worst_score = np.mean(model_params[model][worst_key])
        print(f"  {model}:")
        print(f"    Best:  temp={best_key[0]}, top_p={best_key[1]} -> {best_score:.2f}")
        print(f"    Worst: temp={worst_key[0]}, top_p={worst_key[1]} -> {worst_score:.2f}")
        print(f"    Delta: {best_score - worst_score:.2f}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/full_sweep",
                        help="Directory containing scores.json and raw_answers.json")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save charts (defaults to results-dir)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_dir
    scores, answers = load_data(args.results_dir)

    plot_scores_by_model(scores, out_dir)
    plot_quality_vs_latency(scores, answers, out_dir)
    plot_dimension_breakdown(scores, out_dir)
    plot_token_counts(answers, out_dir)
    plot_score_vs_temperature(scores, out_dir)
    plot_score_vs_top_p(scores, out_dir)
    plot_best_params_heatmap(scores, out_dir)
    print_best_worst(scores, answers)
    print(f"\nAll charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
