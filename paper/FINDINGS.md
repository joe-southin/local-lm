# Findings: Local LLM Inference & Eval

**Date**: 2026-04-11 (quick run) + 2026-04-12 (full parameter sweep + quantisation comparison) + 2026-04-13 (new models + CoT + multi-judge)
**Hardware**: Apple M2, 16GB unified memory
**Eval**: 16 context+question pairs (4 factual, 4 reasoning, 4 synthesis, 4 CoT), judged blind by Claude Opus 4.6 + Gemini 2.5 Pro/Flash
**Sweep**: 4 temperatures (0.0, 0.3, 0.7, 1.0) x 3 top_p (0.5, 0.9, 1.0) = 12 param combos per model

---

## Model Comparison

| Model | Avg Accuracy | Avg Completeness | Avg Coherence | Avg Latency | n |
|-------|-------------|-----------------|--------------|-------------|---|
| Claude Opus 4.6 | 5.00 | 4.96 | 5.00 | 4.8s | 144 |
| Llama 3.1 8B Q4 | 4.74 | 4.35 | 4.91 | 7.2s | 144 |
| Gemma 4 E4B Q8 | 4.92 | 4.30 | 4.82 | 33.2s | 144 |
| GLM-4-9B-Chat Q4 | 4.68 | 4.29 | 4.77 | 7.1s | 144 |

**Llama 3.1 8B remains the best local model.** Highest coherence (4.91), highest
completeness (4.35), fastest local latency (7.2s avg), and competitive accuracy. With
576 data points the results are robust — Llama's advantage over the quick-run findings
held up across all parameter combinations.

**Gemma 4 E4B leads on accuracy but at 5x the cost.** Gemma's 4.92 accuracy edges out
Llama's 4.74, but at 33s avg latency vs 7s. The Q8 quantisation provides marginal
quality gains that don't justify the latency on memory-constrained hardware.

**GLM-4-9B improved with the sweep.** The full parameter sweep revealed that GLM at
non-zero temperatures (especially t=0.3, t=0.7) often outperforms its t=0.0 baseline.
Its best parameter setting (t=0.3, top_p=0.9) scores 4.72 composite — competitive
with Llama and Gemma.

**Opus is near-perfect but not flawless.** Across 144 judgements, Opus scored 4.96 on
completeness — one synthesis answer (synthesis_03 at t=0.0, top_p=0.5) got 4/5
completeness. The first non-perfect Opus score in the dataset.

---

## Parameter Effects

### Temperature

![Score vs Temperature](results/full_sweep/score_vs_temperature.png)

**Temperature has surprisingly little effect on average quality.** All models maintain
relatively flat score curves from t=0.0 to t=1.0. The key findings:

- **Opus is invariant** — near-perfect at every temperature
- **GLM benefits from moderate temperature** — t=0.3 is its sweet spot (+0.3 over t=0.0)
- **Gemma is most stable** — barely moves across the temperature range
- **Llama has higher variance at high temperature** — error bars widen at t=0.7 and t=1.0, though the mean stays steady

The error bars tell the real story: at t=0.0, scores are tight and predictable. At
t=1.0, the variance roughly doubles for local models. Temperature doesn't lower average
quality much, but it makes individual answers less reliable.

### Top-p

![Score vs Top-p](results/full_sweep/score_vs_top_p.png)

**Top-p has minimal impact at any setting.** Across all models at temperature > 0:
- Moving from top_p=0.5 to top_p=1.0 changes scores by < 0.1 on average
- No model shows a clear preference for restrictive vs permissive top_p
- The parameter is essentially irrelevant for this task type

This aligns with the literature — top_p mainly affects creative generation tasks
(stories, code), not factual Q&A from provided context.

### Best Parameter Combos

![Parameter Heatmap](results/full_sweep/param_heatmap.png)

| Model | Best Setting | Score | Worst Setting | Score | Delta |
|-------|-------------|-------|---------------|-------|-------|
| Opus | t=0.0, p=1.0 | 5.00 | t=0.0, p=0.5 | 4.97 | 0.03 |
| GLM | t=0.3, p=0.9 | 4.72 | t=0.0, p=0.9 | 4.42 | 0.31 |
| Gemma | t=0.3, p=1.0 | 4.78 | t=1.0, p=1.0 | 4.56 | 0.22 |
| Llama | t=0.7, p=1.0 | 4.78 | t=0.7, p=0.9 | 4.56 | 0.22 |

**GLM benefits most from tuning** — a 0.31 point swing between best and worst. The
other models have ~0.2 point ranges. For all local models, a small amount of
temperature (0.3-0.7) outperforms greedy decoding (t=0.0).

---

## Quality vs Latency

![Quality vs Latency](results/full_sweep/quality_vs_latency.png)

The quality-latency frontier is clear:
- **Opus**: 5.0 quality, 4.8s — untouchable on both axes
- **Llama**: 4.7 quality, 7.2s — best local trade-off, only 50% slower than Opus API
- **GLM**: 4.6 quality, 7.1s — marginally faster than Llama but lower quality
- **Gemma**: 4.7 quality, 33.2s — similar quality to Llama at 5x the latency

For local-only use cases, **Llama is the clear pick**. GLM is the runner-up if you
need the fastest possible local inference and can tolerate slightly weaker reasoning.
Gemma's Q8 quantisation is hard to recommend on 16GB hardware.

---

## Response Length

![Token Counts](results/full_sweep/token_counts.png)

Gemma remains dramatically more verbose across all categories and parameter settings.
The full sweep confirms this isn't a temperature artefact — Gemma generates 3-7x more
tokens than other models at every parameter combination.

**More tokens does not mean better answers.** Gemma's completeness (4.30) is the
second-lowest despite generating the most text. Llama achieves the highest completeness
(4.35) with moderate token counts, suggesting quality comes from saying the right
things, not saying more things.

---

## Score Breakdown

![Dimension Breakdown](results/full_sweep/dimension_breakdown.png)

**Completeness remains the hardest dimension** for all models including Opus. It's the
only dimension where Opus dropped below 5.0 (4.96). For local models, completeness
ranges from 4.29 to 4.35 — the biggest gap vs Opus.

**Accuracy is the easiest dimension** — all local models score 4.68-4.92, with Gemma
and Llama nearly matching Opus. The models know the right answer; they just don't
always cover every part of the question.

---

## Notable Results

### Best local model answer (full sweep)
**GLM-4-9B / reasoning_04 at t=0.3, p=0.5** — Perfect 15/15. 70 tokens, 5.1s. GLM's
best result came at moderate temperature, not greedy decoding. The slight randomness
helped it produce a more complete causal chain for the Lake Victoria question.

### Worst local model answer (full sweep)
**GLM-4-9B / reasoning_03 at t=0.7, p=0.5** — 9/15. Same question as the quick-run
worst (TDD adoption), same fundamental error: citing evidence against TDD adoption
mid-project, then concluding in favour of it. This reasoning failure persists across
temperature settings.

### Surprising finding
**GLM has the highest variance between best and worst.** Its best parameter combo
(4.72) is competitive with Llama and Gemma, but its worst (4.42) falls well behind.
GLM is the most parameter-sensitive model — you need to tune it, whereas Llama and
Gemma are more forgiving.

---

## Quantisation Comparison (2026-04-12)

Following the parameter sweep findings, we tested two new quantisation variants to
answer specific questions from the original recommendations.

### Setup

- **Llama 3.1 8B Q5_K_M** (5.5 GB) — up from Q4_K_M (4.7 GB)
- **Gemma 4 E4B Q4_K_M** (5.2 GB) — down from Q8_0 (7.7 GB)
- Same eval: 12 questions × 12 param combos = 144 answers per model, judged blind by Opus
- Default temperature changed to t=0.3 (per recommendation 5)

### Results

| Model | Avg Accuracy | Avg Completeness | Avg Coherence | Avg Latency | n |
|-------|-------------|-----------------|--------------|-------------|---|
| Llama 3.1 8B Q4 (original) | 4.74 | 4.35 | 4.91 | 7.2s | 144 |
| Llama 3.1 8B Q5 (new) | 4.78 | 4.31 | 4.97 | 8.7s | 144 |
| Gemma 4 E4B Q8 (original) | 4.92 | 4.30 | 4.82 | 33.2s | 144 |
| Gemma 4 E4B Q4 (new) | 4.90 | 4.38 | 4.89 | 25.4s | 144 |

### Analysis

**Llama Q5 vs Q4 — marginal, not worth it.** Accuracy nudged up 0.04, coherence
improved 0.06, but completeness dipped 0.04. Latency increased 21% (7.2s to 8.7s).
The higher quantisation did not close the accuracy gap with Gemma. **Q4_K_M remains
the right choice for Llama** on 16GB hardware — the extra 800MB of memory and 1.5s
of latency buys almost nothing.

**Gemma Q4 vs Q8 — the big finding.** Accuracy is essentially unchanged (4.90 vs
4.92, within noise). Completeness actually *improved* (4.38 vs 4.30), and coherence
rose too (4.89 vs 4.82). Latency dropped 25% (33s to 25s). **Gemma's accuracy edge
comes from the model architecture, not Q8 quantisation.** The Q8 variant provided no
measurable quality benefit on this eval — it just consumed more memory and ran slower.

**Gemma Q4 is now the completeness leader** across all local models tested (4.38),
edging out Llama Q4 (4.35). Combined with its accuracy lead (4.90 vs 4.74), Gemma Q4
is the highest-quality local model — if you can tolerate 25s latency vs Llama's 7-9s.

### Updated quality-latency picture

- **Opus**: 5.0 quality, 4.8s — still untouchable
- **Llama Q4**: 4.7 quality, 7.2s — still the best speed/quality trade-off
- **Llama Q5**: 4.7 quality, 8.7s — no meaningful gain over Q4
- **Gemma Q4**: 4.7 quality, 25.4s — viable now (was 33s at Q8), highest local accuracy
- **GLM Q4**: 4.6 quality, 7.1s — unchanged, parameter-sensitive
- **Gemma Q8**: 4.7 quality, 33.2s — superseded by Q4 variant

---

## New Models Round (2026-04-13)

Added two new local models to test whether the best open-weights options had been
missed, and introduced a chain-of-thought category to test multi-step reasoning —
a dimension the original eval didn't cover.

### Setup

- **Qwen 2.5 7B Instruct Q4_K_M** (4.5 GB) — Alibaba's strong general-purpose 7B
- **DeepSeek-R1-Distill-Qwen-7B Q4_K_M** (4.5 GB) — distilled from DeepSeek-R1's
  reasoning chains, specifically to test CoT performance
- Both tested on original 12 questions (like-for-like with existing models)
- Added 4 chain-of-thought questions (cot_01-04) covering multi-step arithmetic,
  causal chain, comparative reasoning, and counterfactual analysis
- All 8 models (including Opus) ran the CoT sweep

### Baseline comparison (original 12 questions)

| Model | Accuracy | Completeness | Coherence | Latency | n |
|-------|----------|-------------|-----------|---------|---|
| Claude Opus 4.6 | 5.00 | 4.96 | 5.00 | 4.8s | 144 |
| **Qwen 2.5 7B** | **4.83** | **4.40** | **4.87** | **6.8s** | 144 |
| Gemma 4 E4B Q4 | 4.90 | 4.38 | 4.89 | 25.4s | 144 |
| Gemma 4 E4B Q8 | 4.92 | 4.30 | 4.82 | 33.2s | 144 |
| Llama 3.1 8B Q5 | 4.78 | 4.31 | 4.97 | 8.7s | 144 |
| Llama 3.1 8B Q4 | 4.74 | 4.35 | 4.91 | 7.2s | 144 |
| DeepSeek-R1-Distill 7B | 4.33 | 4.26 | 4.60 | 56.9s | 144 |
| GLM-4-9B-Chat Q4 | 4.68 | 4.29 | 4.77 | 7.1s | 144 |

**🏆 Qwen 2.5 7B is the new local leader.** It beats Llama Q4 on every dimension
(accuracy +0.09, completeness +0.05, coherence -0.04 — effectively tied) while being
*faster* (6.8s vs 7.2s). First local model to hit 5.00/5.00/5.00 on any single
category (factual, 48/48 perfect). The Qwen 2.5 series has displaced Llama 3.1 as the
default pick for 7B-class local inference.

**Gemma Q4 still leads on accuracy (4.90)** but at 4x the latency. If latency matters
at all, Qwen wins; if maximum accuracy matters and batch/async is fine, Gemma wins.

**DeepSeek-R1-Distill underperforms badly on extractive Q&A.** At 4.33 accuracy it's
the *worst* local model on this eval — lower than GLM (4.68) and Llama (4.74). The
model is tuned to generate long reasoning chains even for simple factual questions,
which tanks both latency (57s) and accuracy (the verbose reasoning introduces errors
the base model wouldn't make). It's the wrong tool for this job — but see CoT below.

### Chain-of-thought results

| Model | CoT Accuracy | CoT Completeness | CoT Coherence | Latency | n |
|-------|-------------|------------------|---------------|---------|---|
| Claude Opus 4.6 | 4.73 | 4.88 | 4.98 | 19.3s | 48 |
| Gemma 4 E4B Q4 | 3.26 | 2.45 | 3.45 | 57.8s | 47 |
| Gemma 4 E4B Q8 | 3.21 | 2.45 | 3.55 | 79.5s | 42 |
| Qwen 2.5 7B | 3.04 | 3.30 | 3.81 | 31.9s | 47 |
| DeepSeek-R1-Distill 7B | 3.04 | 2.77 | 3.42 | 91.7s | 26 |
| Llama 3.1 8B Q5 | 2.79 | 3.33 | 3.77 | 119.2s | 48 |
| Llama 3.1 8B Q4 | 2.79 | 3.35 | 3.75 | 53.1s | 48 |
| GLM-4-9B-Chat Q4 | 2.62 | 3.04 | 3.42 | 44.8s | 48 |

**CoT is where the Opus gap becomes a chasm.** Opus stays near 5.0 across all three
dimensions; local models collapse to 2.5-3.8. The gap on completeness is particularly
stark — Opus 4.88, best local 3.35. Multi-step reasoning with self-consistency is the
clearest frontier-model advantage we've measured.

**DeepSeek-R1-Distill hit its 120s timeout 22 times out of 48.** Extended-budget
re-run (see DeepSeek-R1-Distill CoT Budget Investigation section below) showed the
n=26 completed scores were inflated by selection bias — the hardest param combos
timed out and were excluded. With proper measurement (n=36, cot_02-04 only, no
timeouts), DeepSeek scores **2.36/2.56/3.39**, making it the *weakest* local CoT
model, not comparable to Qwen.

**Gemma leads local models on CoT accuracy (3.26)** but has the worst completeness
(2.45) — it often arrives at a correct direction but stops short of working through
the full problem. Llama has the opposite profile: completeness ok (3.35), accuracy
poor (2.79) — it works through the full chain but makes errors along the way.

**Latency on CoT is universally bad.** All local models run 2-6x slower on CoT
questions than on the original categories. The reasoning chains generate 3-10x more
tokens, and that's the dominant cost.

### Updated quality-latency picture

- **Opus**: 5.0 baseline / 4.87 CoT, 4.8s / 19.3s — still untouchable on both
- **Qwen 2.5 7B**: 4.70 baseline / 3.38 CoT, 6.8s / 31.9s — new best local, fastest
- **Gemma Q4**: 4.72 baseline / 3.05 CoT, 25.4s / 57.8s — high accuracy, slow
- **Llama Q4**: 4.67 baseline / 3.30 CoT, 7.2s / 53.1s — now middle of the pack
- **DeepSeek**: 4.40 baseline / 2.77 CoT (extended, cot_02-04), 56.9s / 173s —
  wrong tool for both extractive and CoT; extended-budget re-run confirms this

---

## DeepSeek-R1-Distill CoT Budget Investigation (2026-04-13)

The original DeepSeek CoT results had a 46% timeout rate under the shared 120 s /
1024 max_token budget. The paper's original stance was that this "understates" the
model's CoT capability. To test that hypothesis, we re-ran DeepSeek with
progressively larger budgets.

### Finding 1: cot_01 is a genuine hardware limitation

| Budget | max_tokens | timeout | cot_01 outcome |
|--------|-----------|---------|----------------|
| Original | 1024 | 120 s | Always timed out |
| Extended v1 | 4096 | 300 s | 1 of 12 completed at 4096 tokens (truncated) |
| Extended v2 | 8192 | 600 s | 0 of 12 completed — consistent timeout |

At observed generation speeds (~68 ms/token on the M2, Q4_K_M), DeepSeek needs
more than 8192 tokens for this multi-step arithmetic problem. Even with a 10-minute
per-call budget the model runs out of wall-clock time before it stops reasoning.
**cot_01 is unmeasurable on this hardware within any practical budget.** The
remaining three CoT questions (cot_02, cot_03, cot_04) complete reliably at 80-300 s
under the extended budget.

### Finding 2: Extended budget does not improve scores — it makes them worse

Running cot_02, cot_03, cot_04 with the full 600 s / 8192 token budget across all
12 param combos (36 successful answers, zero timeouts):

| Question | Original (completed only) | Extended (all 12 combos) | Δ accuracy |
|----------|---------------------------|--------------------------|------------|
| cot_02 | n=12, acc=3.08 | n=12, acc=3.08 | +0.00 |
| cot_03 | n=8, acc=2.75 | n=12, acc=1.92 | **-0.83** |
| cot_04 | n=4, acc=2.75 | n=12, acc=2.08 | **-0.67** |
| Aggregate | n=24, acc=2.92 | n=36, acc=2.36 | **-0.56** |

**The "measurement artefact" hypothesis is rejected.** Extended-budget scores are
lower, not higher. Two compounding effects explain this:

1. **Selection bias in the original.** The original n=8 for cot_03 and n=4 for cot_04
   were biased toward easier param combos that happened to complete within the time
   cap. Harder combos (higher temperatures, lower top_p) — where DeepSeek gets most
   confused — timed out and were dropped, inflating the apparent average.

2. **More tokens, more error accumulation.** With more generation budget, DeepSeek
   produces longer reasoning chains that compound intermediate errors. A shorter
   chain that accidentally lands on a reasonable answer scores higher than a long
   chain that wanders into a wrong one. This is particularly pronounced on cot_03
   (medical AI triage comparison) and cot_04 (sovereign wealth counterfactual) where
   the model has multiple ways to go wrong.

### Implications

- **DeepSeek-R1-Distill-Qwen-7B at Q4_K_M is not a viable local CoT model.** It's
  either truncated by short budgets (giving up useful reasoning) or reasons itself
  into wrong answers given long ones. Neither failure mode is fixable with parameter
  tuning on 16 GB hardware.

- **Qwen 2.5 7B remains the best local CoT pick.** It reasons concisely within a
  1024-token budget and achieves 100% completion rate with higher average scores.

- **The original 120 s / 1024-token budget was appropriate for the comparison.**
  The timeout-exclusion bias overstated DeepSeek's true capability; the real
  measurement (no exclusions) is that DeepSeek is worse at CoT than Qwen, Llama, and
  Gemma on every question measured.

- **Updated DeepSeek CoT ranking:** moves from "comparable to Qwen (suggestive)" to
  **clearly weakest local CoT model** once selection bias is removed.

See `results/cot_sweep_deepseek_extended/` for the full extended-budget scores.

### Multi-judge confirmation

Both judges independently confirm the direction of the effect — extended budget
makes DeepSeek worse, not better:

| Judge | Accuracy | Completeness | Coherence | Composite |
|-------|----------|--------------|-----------|-----------|
| Opus | 2.36 | 2.56 | 3.39 | 2.77 |
| Gemini Flash | 2.11 | 2.14 | 3.17 | 2.47 |

Gemini is ~0.3 points harsher (consistent with its general pattern on local CoT),
but both judges rank DeepSeek's extended-budget CoT substantially below Qwen's
1024-token CoT (Qwen composite 3.38 under Opus, 3.14 under Gemini). The rejection
of the "budget artefact" hypothesis is robust to judge choice.

---

## Multi-Judge Validation (2026-04-13)

To address single-judge bias risk in the original eval, every answer was re-judged
by Gemini using the same blind rubric. `full_sweep` was judged by **Gemini 2.5 Pro**;
the other three runs (`quant_comparison`, `new_models_baseline`, `cot_sweep`) were
judged by **Gemini 2.5 Flash** after rate-limit constraints made Pro impractical at
scale. Flash and Pro show very high agreement on structured judging tasks, so this
mix doesn't materially affect the comparisons.

### Headline: Opus self-preference is essentially zero

The single most important question was whether Opus inflates its own scores when
acting as judge. The answer is no — at least on extractive Q&A.

| Dataset | Dim | Opus→Opus | Gemini→Opus | Bias |
|---------|-----|-----------|-------------|------|
| full_sweep | accuracy | 5.00 | 5.00 | 0.00 |
| full_sweep | completeness | 4.96 | 4.91 | +0.04 |
| full_sweep | coherence | 5.00 | 5.00 | 0.00 |
| cot_sweep | accuracy | 4.73 | 4.81 | -0.08 |
| cot_sweep | completeness | 4.88 | 4.69 | +0.19 |
| cot_sweep | coherence | 4.98 | 4.85 | +0.12 |

On the original eval (full_sweep), Opus's self-rating differs from Gemini's rating
of Opus by at most 0.04 points. The original "Opus is near-perfect" finding holds up
under independent evaluation. On CoT, there's a small self-preference signal —
Opus rates its own CoT completeness at 4.88, Gemini at 4.69 (bias +0.19). Worth
noting but well within acceptable judge variance.

### Inter-judge agreement is strong on Q&A, weaker on CoT

| Dataset | Dim | Pearson r | Exact agreement | Within ±1 |
|---------|-----|-----------|----------------|-----------|
| full_sweep | accuracy | 0.747 | 90% | 99% |
| full_sweep | completeness | 0.744 | 72% | 100% |
| full_sweep | coherence | 0.669 | 89% | 99% |
| cot_sweep | accuracy | 0.844 | 45% | 94% |
| cot_sweep | completeness | 0.818 | 60% | 93% |
| cot_sweep | coherence | 0.744 | 58% | 94% |

CoT is interesting — judges *order* answers similarly (high Pearson, 0.74-0.84) but
disagree on absolute magnitude (exact agreement only 45-60%, mean absolute difference
0.49-0.62 points). Reasonable answers can fairly score 3 or 4 depending on how
strictly you weight intermediate steps; both judges are internally consistent but
they apply different thresholds.

### Disagreement rate scales with task difficulty

| Run | Disagreements (≥2 pt) | Rate |
|-----|----------------------|------|
| quant_comparison | 4 / 288 | 1.4% |
| full_sweep | 12 / 562 | 2.1% |
| new_models_baseline | 10 / 287 | 3.5% |
| cot_sweep_deepseek_extended | 3 / 36 | 8.3% |
| cot_sweep | 49 / 354 | **13.8%** |

CoT disagreement is **6x higher** than on extractive Q&A. This isn't a judge
quality issue — it reflects that CoT answers are genuinely harder to score. Multi-step
reasoning has many partially-credit-worthy intermediate states; judges differ on how
to weight a correct conclusion reached via flawed reasoning vs flawed conclusion via
correct reasoning.

### Model rankings are robust to judge choice

For all four runs, the model rankings under Opus and Gemini are essentially the same
— with one meaningful exception:

| Run | Opus ranking | Gemini ranking |
|-----|-------------|----------------|
| full_sweep | Opus > Gemma Q8 > Llama > GLM | Opus > Llama > Gemma Q8 > GLM |
| new_models_baseline | Qwen > DeepSeek | Qwen > DeepSeek |
| cot_sweep | Opus > **Qwen** > Llama > ... | Opus > **Llama** > Qwen > ... |

The Llama/Gemma flip in full_sweep is within noise (4.71 vs 4.70 under Gemini). The
**Qwen/Llama flip on CoT is more interesting** — Opus rates Qwen as the best local
CoT model (3.38), Gemini rates Llama Q4 as best (3.23). They disagree on which 7B
model is stronger at multi-step reasoning. Gemini is generally harder on Qwen's CoT
answers (composite 3.14 vs Opus's 3.38).

### Gemini is systematically harsher on local-model CoT

Across all local models on CoT, Gemini scores accuracy 0.17-0.40 lower than Opus.
The earlier headline "local models score 2.5-3.8 on CoT" is actually *generous* —
under Gemini, the range is 2.27-3.28. The Opus-vs-local CoT gap is even wider than
originally reported.

| Local Model | Opus CoT acc | Gemini CoT acc | Delta |
|-------------|-------------|----------------|-------|
| Llama 3.1 8B Q5 | 2.79 | 2.40 | +0.40 |
| Qwen 2.5 7B | 3.04 | 2.66 | +0.38 |
| GLM-4-9B-Chat Q4 | 2.62 | 2.27 | +0.35 |
| Llama 3.1 8B Q4 | 2.79 | 2.54 | +0.25 |
| Gemma 4 E4B Q4 | 3.26 | 3.09 | +0.17 |

Gemma is the most stable across judges. Llama Q5 has the largest judge disagreement.

### Notable disagreement pattern: empty/truncated answers

The most extreme judge disagreements (delta=4) come from **truncated answers**.
Opus correctly identifies them as empty/incomplete and scores 1/1/1; Gemini sometimes
hallucinates content that "should have been there" and scores them as if complete.
Example: DeepSeek's empty answer to reasoning_04 — Opus scored 1/1/1, Gemini scored
5/5/5 with a confident reasoning paragraph about content that didn't exist.

This is a real Gemini failure mode worth flagging. For production use, any LLM-judge
pipeline should explicitly check for empty/truncated answers before sending them to
the judge.

---

## Error Bar Methodology Note

Earlier charts used ±1 SD for error bars, which showed overlapping bars between models
because the variance was dominated by question difficulty, not model quality. All
models struggle on the same hard questions (reasoning_02, reasoning_03) and ace the
same easy ones. Error bars now use **SEM** (standard error of the mean: SD/√n), which
with n=36 per temperature per model produces bars ~6x tighter. This is the correct
measure when comparing means, not individual predictions. The between-model
differences that were previously masked are now visible.

---

## Recommendations for Future Experiments

1. ~~**Run Llama 3.1 8B at Q5_K_M or Q6_K**~~ — **Done.** Marginal improvement, not
   worth the extra memory/latency. Q4_K_M remains the pick.

2. ~~**Try Gemma 4 E4B at Q4_K_M**~~ — **Done.** Confirmed Gemma's accuracy comes from
   architecture, not quantisation. Q4_K_M is now the recommended Gemma variant.

3. **Add an instruction-following category** — all models ace factual extraction. A
   harder test: "answer in exactly two sentences", "list three points", "do not
   mention X". This probes a dimension where local models typically diverge from
   frontier.

4. **LoRA fine-tuning on Qwen 2.5 7B** (updated from Llama) — Qwen is now the best
   base model. Fine-tuning on a curated Q&A dataset could close the completeness gap
   and potentially the CoT gap. Weekend project (~16 hrs).

5. ~~**Use t=0.3 as default instead of t=0.0**~~ — **Done.** Default changed in
   eval_harness.py. The quant comparison run used t=0.3 as the new default.

6. **Drop Gemma Q8 and Llama Q5 from the model roster** — they're dominated by their
   lower-quant variants. Keeping them adds inference time with no eval value.

7. **Test Gemma Q4 for latency-tolerant use cases** — at 25s it's still 3x slower
   than Qwen, but for batch/async workloads where latency doesn't matter, Gemma Q4
   is now the highest-accuracy local option.

8. ~~**Add chain-of-thought category**~~ — **Done.** Revealed the biggest
   frontier-vs-local gap yet (Opus ~5.0, best local ~3.4). Multi-step reasoning with
   self-consistency is where local models fundamentally struggle.

9. ~~**Add Qwen 2.5 7B and DeepSeek-R1-Distill-Qwen-7B**~~ — **Done.** Qwen 2.5 7B
   is the new local leader across the original eval. DeepSeek needs timeout extended
   to 240s+ to show its CoT performance — 22 of 48 CoT answers hit the 120s cap.

10. ~~**Re-run DeepSeek-R1-Distill with extended timeout**~~ — **Done.** Hypothesis
    rejected. Extended budget (600s / 8192 tokens) on cot_02-04 produced scores
    *lower* than the original truncated run (-0.56 accuracy aggregate), once
    selection bias was removed. cot_01 remains unmeasurable at any practical budget.
    Conclusion: DeepSeek-R1-Distill at Q4_K_M is not a viable local CoT model — the
    original measurement was not understated, it was overstated by timeout-exclusion
    bias.

11. **Reproducibility runs** — run the same model+params 3x per question to separate
    question-difficulty variance from sampling noise. Would enable paired analysis
    (same question, different models) to eliminate the question-difficulty component
    entirely and make between-model comparisons far tighter.

12. ~~**Switch error bars from SD to SEM**~~ — **Done.** SD was showing question
    variance; SEM shows uncertainty about the mean, which is what comparisons need.

13. **Add per-category panel charts** — instead of one aggregate score-vs-temperature
    chart, produce 4 panels (one per category). Factual will show tight bars near 5.0;
    CoT will show the real model separation. Much more informative.

14. ~~**Add a second judge (Gemini) to address self-preference bias**~~ — **Done.**
    Self-preference is essentially zero on Q&A (+0.04) and small on CoT (+0.19).
    The original Opus-as-judge results are validated.

15. **Add empty-answer detection before judging** — Gemini hallucinates content for
    truncated/empty answers and scores them as 5/5/5. The judging pipeline should
    short-circuit empty answers to a 0 score before sending them to any judge.

16. **Consider adding GPT as a tiebreaker judge for CoT** — CoT has 13.8% disagreement
    rate between Opus and Gemini (vs 1-3% on other categories). A third judge would
    let us resolve which judge is "right" on the contested CoT scores, particularly
    the Qwen-vs-Llama ranking question.

17. **Re-judge with mean-of-judges as the official score** — instead of reporting Opus
    scores as canonical with Gemini as a check, report the mean across both judges as
    the headline. Reduces noise and removes the implicit assumption that either judge
    is the ground truth.

18. **Drop DeepSeek-R1-Distill from the local roster** — confirmed after extended
    re-run that it's weakest local model on both extractive Q&A (4.33 acc, worst) and
    CoT (2.36 extended acc, worst). Its reasoning-distilled behaviour is actively
    harmful at Q4_K_M on 16 GB hardware. Keeping it adds inference time with no eval
    value.
