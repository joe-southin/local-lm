# Good Enough? Quality, Latency, and Parameter Sensitivity of Quantized 7–9B LLMs on Consumer Hardware

This repository contains the evaluation harness, raw data, and analysis code supporting the paper of the same title (Southin, 2026). It evaluates five open-weights model families — Llama 3.1 8B, Gemma 4 E4B, GLM-4-9B, Qwen 2.5 7B, and DeepSeek-R1-Distill-Qwen-7B — at selected quantisation levels against a Claude Opus 4.6 baseline on 16 context-grounded Q&A items, then scores every response blind with two independent LLM judges (Claude Opus 4.6 and Gemini 2.5).

The paper PDF lives at [`paper/paper.pdf`](paper/paper.pdf); its Markdown source is [`paper/paper.md`](paper/paper.md).

## Repository layout

```
.
├── eval_harness.py            # Main harness: runs models, collects answers, judges with Opus
├── judge_gemini.py            # Secondary judge (Gemini 2.5) over existing raw_answers.json
├── multi_judge_analysis.py    # Compares Opus vs Gemini judges (correlation, self-preference)
├── analyse_results.py         # Charts and summary tables from a results dir
├── rejudge.py                 # Re-run judging on an existing results dir
├── eval_questions.json        # 16 context-grounded Q&A items (factual / reasoning / synthesis / CoT)
├── models.json                # Model configs (GGUF paths + context sizes)
├── setup.sh                   # Installs llama.cpp + downloads GGUFs to ~/models
├── requirements.txt           # Python deps
├── results/                   # All experimental data backing the paper
│   ├── full_sweep/            # Original 4 models × 12 questions × 12 params
│   ├── quant_comparison/      # Llama Q5 + Gemma Q4 quant variants
│   ├── new_models_baseline/   # Qwen + DeepSeek on original 12 questions
│   ├── cot_sweep/             # All 8 models on 4 CoT questions
│   ├── cot_sweep_deepseek_extended/
│   └── cot_deepseek_quick_validation/
├── paper/                     # Paper sources + final figures
└── tests/                     # pytest suite (schema + pure-function smoke tests)
```

## Quick start

### 1. Install Python deps

```bash
pip install -r requirements.txt
```

### 2. Download GGUFs (optional — only needed to reproduce inference)

```bash
bash setup.sh            # installs llama.cpp via Homebrew and pulls ~40 GB of GGUFs
```

### 3. Configure API keys

Copy `.env.example` to `.env` and fill in:

```
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

API keys are only needed to re-run the Claude Opus subject / judge or the Gemini judge; analysis of the cached `results/` works without them.

### 4. Re-run analysis on the shipped data

```bash
python analyse_results.py --results-dir results/full_sweep
python multi_judge_analysis.py results/full_sweep
```

### 5. Regenerate the paper figures

```bash
python paper/generate_figures.py --output-dir paper/figures
```

### 6. Run the full eval (GPU recommended)

```bash
python eval_harness.py --quick                         # 1 param combo, all models
python eval_harness.py --models "Qwen 2.5 7B"          # single model, full sweep
python eval_harness.py --category cot                  # just the CoT questions
```

## Reproducing the paper's results

Every chart and table in `paper/paper.pdf` is regenerated deterministically from the JSON/CSV files in `results/`. The harness was run on an Apple Silicon MacBook (M-series, 64 GB unified memory) via `llama-server` from llama.cpp; exact versions and hardware are documented in §3 of the paper.

## Tests

```bash
pip install pytest
pytest
```

The test suite validates data-file schemas and pure-function behaviour. It does **not** make any network calls or spin up a llama-server.

## Citation

If you use this code or data, please cite:

```bibtex
@misc{southin2026goodenough,
  title  = {Good Enough? Quality, Latency, and Parameter Sensitivity of
            Quantized 7--9B LLMs on Consumer Hardware},
  author = {Southin, J. E. A.},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
