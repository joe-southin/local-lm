#!/usr/bin/env python3
"""Generate publication-quality figures combining all experimental data.

Sources:
    results/full_sweep/       original 4 models x 12 questions x 12 params
    results/quant_comparison/ Llama Q5 + Gemma Q4 variants
    results/new_models_baseline/ Qwen + DeepSeek on original 12 questions
    results/cot_sweep/        all 8 models on 4 CoT questions

Outputs vector PDFs to paper/figures/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paul Tol bright palette
COLOURS = {
    "blue":    "#4477AA",
    "cyan":    "#66CCEE",
    "green":   "#228833",
    "yellow":  "#CCBB44",
    "red":     "#EE6677",
    "purple":  "#AA3377",
    "orange":  "#EE7733",
    "teal":    "#009988",
    "grey":    "#BBBBBB",
}

MODEL_STYLE = {
    "Claude Opus 4.6":        {"color": COLOURS["blue"],   "marker": "o"},
    "Qwen 2.5 7B":            {"color": COLOURS["teal"],   "marker": "P"},
    "Gemma 4 E4B Q4":         {"color": COLOURS["cyan"],   "marker": "v"},
    "Gemma 4 E4B Q8":         {"color": COLOURS["green"],  "marker": "D"},
    "Llama 3.1 8B Q4":        {"color": COLOURS["yellow"], "marker": "^"},
    "Llama 3.1 8B Q5":        {"color": COLOURS["purple"], "marker": "<"},
    "GLM-4-9B-Chat Q4":       {"color": COLOURS["red"],    "marker": "s"},
    "DeepSeek-R1-Distill 7B": {"color": COLOURS["orange"], "marker": "X"},
}

# Short display names for crowded charts
SHORT_NAMES = {
    "Claude Opus 4.6":        "Opus 4.6",
    "Qwen 2.5 7B":            "Qwen 2.5",
    "Gemma 4 E4B Q4":         "Gemma Q4",
    "Gemma 4 E4B Q8":         "Gemma Q8",
    "Llama 3.1 8B Q4":        "Llama Q4",
    "Llama 3.1 8B Q5":        "Llama Q5",
    "GLM-4-9B-Chat Q4":       "GLM Q4",
    "DeepSeek-R1-Distill 7B": "DeepSeek",
}


def configure_mpl():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    })


def get_style(model_name):
    return MODEL_STYLE.get(model_name, {"color": COLOURS["grey"], "marker": "x"})


def load_all_scores(base="results"):
    """Load and combine scores from all four experiment directories.

    Deduplicates by (model, question_id, temperature, top_p): later runs take
    precedence over earlier runs for the same configuration.
    """
    base = Path(base)
    sources = [
        "full_sweep",
        "quant_comparison",
        "new_models_baseline",
        "cot_sweep",
    ]
    combined = {}
    for src in sources:
        path = base / src / "scores.json"
        if not path.exists():
            continue
        for s in json.load(open(path)):
            key = (s["model_name"], s["question_id"],
                   s["temperature"], s["top_p"])
            combined[key] = s
    return list(combined.values())


def load_all_answers(base="results"):
    base = Path(base)
    sources = ["full_sweep", "quant_comparison",
               "new_models_baseline", "cot_sweep"]
    combined = {}
    for src in sources:
        path = base / src / "raw_answers.json"
        if not path.exists():
            continue
        for a in json.load(open(path)):
            key = (a["model_name"], a["question_id"],
                   a["temperature"], a["top_p"])
            combined[key] = a
    return list(combined.values())


# --------------------------------------------------------------------------
# Figure 1: Score vs Temperature, four panels (one per category)
# --------------------------------------------------------------------------
def fig_score_vs_temperature(scores, out_dir):
    categories = ["factual", "reasoning", "synthesis", "cot"]
    cat_titles = {"factual": "Factual", "reasoning": "Reasoning",
                  "synthesis": "Synthesis", "cot": "Chain-of-thought"}

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for s in scores:
        composite = (s["accuracy"] + s["completeness"] + s["coherence"]) / 3
        cat = s["question_id"].split("_")[0]
        if cat in categories:
            data[s["model_name"]][cat][s["temperature"]].append(composite)

    # Order models: Opus first, then by type
    model_order = ["Claude Opus 4.6", "Qwen 2.5 7B", "Gemma 4 E4B Q4",
                   "Gemma 4 E4B Q8", "Llama 3.1 8B Q4", "Llama 3.1 8B Q5",
                   "GLM-4-9B-Chat Q4", "DeepSeek-R1-Distill 7B"]
    models_present = [m for m in model_order if m in data]

    n_models = len(models_present)
    jitter_width = 0.06
    offsets = np.linspace(-jitter_width / 2, jitter_width / 2, n_models)

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.5), sharex=True, sharey=True)

    for idx, cat in enumerate(categories):
        ax = axes[idx // 2][idx % 2]
        for i, model in enumerate(models_present):
            style = get_style(model)
            cat_data = data[model][cat]
            temps = sorted(cat_data.keys())
            if not temps:
                continue
            jittered = [t + offsets[i] for t in temps]
            means = [np.mean(cat_data[t]) for t in temps]
            n_vals = [len(cat_data[t]) for t in temps]
            sems = [np.std(cat_data[t]) / np.sqrt(n) if n > 1 else 0
                    for t, n in zip(temps, n_vals)]
            label = SHORT_NAMES.get(model, model) if idx == 0 else None
            ax.errorbar(jittered, means, yerr=sems, label=label,
                        color=style["color"], marker=style["marker"],
                        capsize=2, capthick=0.6, linewidth=0.9, markersize=4)

        ax.set_title(cat_titles[cat], fontsize=10)
        ax.set_xticks([0.0, 0.3, 0.7, 1.0])
        ax.set_ylim(2.0, 5.3)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if idx // 2 == 1:
            ax.set_xlabel("Temperature")
        if idx % 2 == 0:
            ax.set_ylabel("Composite score (1\u20135)")

    axes[0][0].legend(loc="lower right", framealpha=0.9, edgecolor="none",
                      fontsize=7, ncol=2, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "fig1_score_vs_temperature.pdf")
    plt.close()
    print("  fig1_score_vs_temperature.pdf")


# --------------------------------------------------------------------------
# Figure 2: Parameter heatmap for top 4 models
# --------------------------------------------------------------------------
def fig_param_heatmap(scores, out_dir):
    # Show the 4 most important models to keep the figure readable:
    # Opus (frontier), Qwen (new best local), Gemma Q4 (accuracy leader),
    # Llama Q4 (established baseline)
    featured = ["Claude Opus 4.6", "Qwen 2.5 7B",
                "Gemma 4 E4B Q4", "Llama 3.1 8B Q4"]
    # Use only baseline (non-CoT) scores
    filtered = [s for s in scores
                if s["model_name"] in featured
                and not s["question_id"].startswith("cot_")]

    temps = sorted(set(s["temperature"] for s in filtered))
    top_ps = sorted(set(s["top_p"] for s in filtered))

    n = len(featured)
    fig, axes = plt.subplots(1, n, figsize=(6.8, 2.8), sharey=True)

    for idx, model in enumerate(featured):
        sums = np.zeros((len(temps), len(top_ps)))
        counts = np.zeros_like(sums)
        for s in filtered:
            if s["model_name"] != model:
                continue
            ti = temps.index(s["temperature"])
            pi = top_ps.index(s["top_p"])
            composite = (s["accuracy"] + s["completeness"] + s["coherence"]) / 3
            sums[ti][pi] += composite
            counts[ti][pi] += 1
        matrix = np.zeros_like(sums)
        np.divide(sums, counts, out=matrix, where=counts > 0)

        im = axes[idx].imshow(matrix, cmap="RdYlGn", vmin=4.0, vmax=5.0,
                              aspect="auto")
        axes[idx].set_xticks(range(len(top_ps)))
        axes[idx].set_xticklabels([f"{p}" for p in top_ps], fontsize=9)
        axes[idx].set_yticks(range(len(temps)))
        if idx == 0:
            axes[idx].set_yticklabels([f"{t}" for t in temps], fontsize=9)
            axes[idx].set_ylabel("Temperature")
        else:
            axes[idx].set_yticklabels([])
        axes[idx].set_xlabel("Top-p", fontsize=9)
        short = SHORT_NAMES.get(model, model)
        axes[idx].set_title(short, fontsize=9, pad=3)

        for i in range(len(temps)):
            for j in range(len(top_ps)):
                val = matrix[i][j]
                if val > 0:
                    axes[idx].text(j, i, f"{val:.2f}", ha="center", va="center",
                                   fontsize=8, fontweight="bold",
                                   color="white" if val < 4.4 else "black")

    fig.subplots_adjust(right=0.90, wspace=0.08)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Score")
    fig.savefig(Path(out_dir) / "fig2_param_heatmap.pdf")
    plt.close()
    print("  fig2_param_heatmap.pdf")


# --------------------------------------------------------------------------
# Figure 3: Quality vs Latency for all models (baseline 12 questions)
# --------------------------------------------------------------------------
def fig_quality_vs_latency(scores, answers, out_dir):
    # Use only non-CoT scores/answers for the baseline Pareto view
    model_data = defaultdict(lambda: {"scores": [], "latencies": []})
    for s in scores:
        if s["question_id"].startswith("cot_"):
            continue
        composite = (s["accuracy"] + s["completeness"] + s["coherence"]) / 3
        model_data[s["model_name"]]["scores"].append(composite)
    for a in answers:
        if a["question_id"].startswith("cot_"):
            continue
        model_data[a["model_name"]]["latencies"].append(a["latency_seconds"])

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    points = []
    for model in sorted(model_data.keys()):
        d = model_data[model]
        if not d["scores"] or not d["latencies"]:
            continue
        style = get_style(model)
        avg_score = np.mean(d["scores"])
        avg_latency = np.mean(d["latencies"])
        sem_score = np.std(d["scores"]) / np.sqrt(len(d["scores"]))
        sem_latency = np.std(d["latencies"]) / np.sqrt(len(d["latencies"]))
        label = SHORT_NAMES.get(model, model)
        ax.errorbar(avg_latency, avg_score,
                    xerr=sem_latency, yerr=sem_score,
                    fmt="none", ecolor=style["color"], elinewidth=0.7,
                    capsize=2.5, capthick=0.7, alpha=0.6, zorder=4)
        ax.scatter(avg_latency, avg_score, s=55, color=style["color"],
                   marker=style["marker"], zorder=5,
                   edgecolors="black", linewidths=0.4)
        points.append((avg_latency, avg_score, label, style["color"]))

    # Manual label offsets to avoid overlap (dx, dy in points)
    # Qwen and GLM moved into chart interior to avoid overlapping y-axis
    label_offsets = {
        "Opus 4.6":  (10, 3),
        "Qwen 2.5":  (10, 14),
        "Gemma Q4":  (10, 0),
        "Gemma Q8":  (10, 0),
        "Llama Q4":  (10, 6),
        "Llama Q5":  (10, -10),
        "GLM Q4":    (10, -16),
        "DeepSeek":  (-58, -2),
    }
    for x, y, label, colour in points:
        dx, dy = label_offsets.get(label, (8, 0))
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color=colour,
                    fontweight="bold")

    ax.set_xlabel("Mean latency (s)")
    ax.set_ylabel("Composite score (1\u20135)")
    ax.set_ylim(4.2, 5.15)
    fig.savefig(Path(out_dir) / "fig3_quality_vs_latency.pdf")
    plt.close()
    print("  fig3_quality_vs_latency.pdf")


# --------------------------------------------------------------------------
# Figure 4: Token counts by model and category (including CoT)
# --------------------------------------------------------------------------
def fig_token_counts(answers, out_dir):
    data = defaultdict(lambda: defaultdict(list))
    for a in answers:
        cat = a["question_id"].split("_")[0]
        data[a["model_name"]][cat].append(a["tokens_generated"])

    model_order = ["Claude Opus 4.6", "Qwen 2.5 7B", "Gemma 4 E4B Q4",
                   "Gemma 4 E4B Q8", "Llama 3.1 8B Q4", "Llama 3.1 8B Q5",
                   "GLM-4-9B-Chat Q4", "DeepSeek-R1-Distill 7B"]
    models = [m for m in model_order if m in data]
    categories = ["factual", "reasoning", "synthesis", "cot"]
    cat_titles = ["Factual", "Reasoning", "Synthesis", "Chain-of-thought"]

    x = np.arange(len(categories))
    n = len(models)
    width = 0.8 / n
    offsets = np.arange(n) - (n - 1) / 2

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    for i, model in enumerate(models):
        style = get_style(model)
        means = [np.mean(data[model].get(cat, [0])) for cat in categories]
        label = SHORT_NAMES.get(model, model)
        ax.bar(x + offsets[i] * width, means, width, label=label,
               color=style["color"], edgecolor="white", linewidth=0.3)

    ax.set_ylabel("Mean tokens generated")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_titles)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none",
              fontsize=7, ncol=2, columnspacing=0.8, handletextpad=0.3)
    fig.savefig(Path(out_dir) / "fig4_token_counts.pdf")
    plt.close()
    print("  fig4_token_counts.pdf")


# --------------------------------------------------------------------------
# Figure 5: Dimension breakdown for all models (baseline only)
# --------------------------------------------------------------------------
def fig_dimension_breakdown(scores, out_dir):
    # Baseline only
    filtered = [s for s in scores if not s["question_id"].startswith("cot_")]

    model_order = ["Claude Opus 4.6", "Qwen 2.5 7B", "Gemma 4 E4B Q4",
                   "Gemma 4 E4B Q8", "Llama 3.1 8B Q4", "Llama 3.1 8B Q5",
                   "GLM-4-9B-Chat Q4", "DeepSeek-R1-Distill 7B"]
    models_present = [m for m in model_order
                      if any(s["model_name"] == m for s in filtered)]
    dimensions = ["accuracy", "completeness", "coherence"]

    data = defaultdict(lambda: defaultdict(list))
    for s in filtered:
        for dim in dimensions:
            data[s["model_name"]][dim].append(s[dim])

    matrix = []
    labels = []
    for model in models_present:
        row = [np.mean(data[model][dim]) for dim in dimensions]
        matrix.append(row)
        labels.append(SHORT_NAMES.get(model, model))

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=4.0, vmax=5.05, aspect="auto")

    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels([d.capitalize() for d in dimensions])
    ax.set_yticks(range(len(models_present)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(models_present)):
        for j in range(len(dimensions)):
            val = matrix[i][j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val < 4.3 else "black")

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(Path(out_dir) / "fig5_dimension_breakdown.pdf")
    plt.close()
    print("  fig5_dimension_breakdown.pdf")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-base", default="results")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    configure_mpl()
    scores = load_all_scores(args.results_base)
    answers = load_all_answers(args.results_base)
    print(f"Loaded {len(scores)} scores, {len(answers)} answers")

    print("Generating publication figures (v2):")
    fig_score_vs_temperature(scores, out)
    fig_param_heatmap(scores, out)
    fig_quality_vs_latency(scores, answers, out)
    fig_token_counts(answers, out)
    fig_dimension_breakdown(scores, out)
    print(f"Done. All figures in {out}/")


if __name__ == "__main__":
    main()
