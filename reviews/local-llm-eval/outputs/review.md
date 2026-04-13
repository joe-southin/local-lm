# Literature Review: Quantized Small LLM Performance on Consumer Hardware

## 1. Topic Overview

This review examines the empirical landscape around quantized small language models (7–9B parameters) running on consumer hardware, with particular focus on sampling parameter effects and LLM-as-judge evaluation methodology. The analysis was decomposed thematically across five sub-areas: quantization effects on small models, sampling parameter sensitivity, LLM-as-judge reliability, edge inference performance on Apple Silicon, and the relationship between response verbosity and answer quality. The evidence landscape is unevenly developed: quantization benchmarking and LLM-as-judge methodology are well-studied with multiple surveys and systematic evaluations, while the interaction between quantization, sampling parameters, and task-specific quality on consumer hardware remains sparse. Across 64 references spanning peer-reviewed papers, preprints, technical reports, and open-source tools, the literature converges on several findings directly relevant to contextualising local inference experiments — but leaves measurable gaps that targeted empirical work can address.

## 2. Introduction

The democratisation of large language models through quantization and efficient inference frameworks has made it feasible to run 7–9B parameter models on consumer laptops with unified memory architectures. This creates a practical question: how much quality do you sacrifice for local, private, low-latency inference — and can parameter tuning recover some of that gap?

This review addresses five research questions:

1. How do quantized 7–9B parameter open-weights models compare to frontier models on context-grounded Q&A tasks?
2. What effect do sampling parameters (temperature, top_p) have on factual Q&A quality, and does the effect differ between model families?
3. How reliable and valid is the LLM-as-judge methodology for evaluating model output quality?
4. What are the quality-latency-memory tradeoffs for local inference on consumer-grade Apple Silicon hardware?
5. Does response verbosity correlate with answer quality, and how do quantization levels affect this?

The scope covers 2020–2026, spanning the emergence of efficient quantization formats (GGUF, GPTQ, AWQ), the maturation of local inference frameworks (llama.cpp, MLX, Ollama), and the rapid development of LLM-as-judge methodology.

## 3. Methodology

This review employed a multi-LLM parallel search approach to maximise coverage. Research demonstrates that different AI assistants searching the same databases find substantially non-overlapping result sets, making complementary search a practical necessity.

**Search tracks:**

| Source | Found | Unique | Overlap with others |
|--------|-------|--------|---------------------|
| Claude web search | 28 | 22 | 6 |
| Google Scholar (via Claude) | 7 | 3 | 4 |
| Supplementary academic (arXiv, Semantic Scholar) | 13 | 10 | 3 |
| Gemini | 22 | 18 | 4 |
| **Total unique** | **64** | | |

Search terms included: quantized LLM benchmark, small language model evaluation, LLM-as-judge reliability, temperature sampling parameter effect, Apple Silicon inference, GGUF quantization, and competitive terms (Ollama, LM Studio, MLX benchmark). Inclusion criteria required empirical evaluation or benchmarking of language models, studies of sampling parameters, LLM-as-judge methodology, quantization effect studies, or edge inference performance. Marketing materials and pure training/fine-tuning papers were excluded.

## 4. How the Field Got Here

The trajectory from "LLMs require datacentre hardware" to "LLMs run on your laptop" unfolded across three parallel threads: model compression, inference optimisation, and evaluation methodology.

Nucleus sampling was introduced by Holtzman et al. (2020) at ICLR, establishing top-p as the dominant alternative to greedy decoding by showing that maximisation-based approaches produce degenerate text. This foundational work motivated the sampling parameter space that local inference frameworks now expose. Quantization became practical with the GGUF format (2023) and llama.cpp's C++ inference engine, which combined model compression with Metal GPU acceleration on Apple Silicon. By 2024, the open-weights frontier was trailing closed models by roughly three months on average (Epoch AI, 2025), and frontier capabilities were becoming reproducible on consumer hardware within approximately one year of their initial release.

The LLM-as-judge paradigm was formalised by Zheng et al. (2023) with MT-Bench and Chatbot Arena, establishing that GPT-4 as evaluator achieves >80% agreement with human preferences. This opened the door for automated evaluation of local models against frontier baselines — but also introduced systematic biases that subsequent work has documented extensively.

**Milestone Timeline:**

| Year | Milestone | Significance |
|------|-----------|--------------|
| 2020 | Holtzman et al. — nucleus (top-p) sampling | Established theoretical basis for sampling parameter control in generation |
| 2023 | GGUF format + llama.cpp | Made quantized local inference practical on consumer hardware |
| 2023 | Zheng et al. — MT-Bench / LLM-as-judge | Formalised automated evaluation methodology now used across the field |
| 2024 | ACL — Comprehensive quantization evaluation | Showed 4-bit quantized LLMs retain comparable performance to full precision |
| 2024 | Renze & Guven — Temperature effect on problem-solving | First systematic evidence that temperature has no statistically significant effect on LLM problem-solving |
| 2024 | Panickssery et al. — Self-preference bias (NeurIPS) | Proved LLM judges recognise and favor their own outputs |
| 2025 | SLMQuant — Small model quantization benchmark | Revealed fundamental disparities in how small models respond to quantization vs. large models |
| 2026 | Unified llama.cpp quantization eval on Llama-3.1-8B | First comprehensive GGUF-specific evaluation on the most widely-deployed 8B model |

**Terminology evolution:** Early work (2020–2022) discussed "model compression" and "knowledge distillation"; current literature uses "quantization-aware deployment" and "edge LLM inference." The shift from "small models" to "SLMs" (small language models) as a distinct research category solidified in 2024 with dedicated surveys and benchmarks.

## 5. Key Research Groups

**Renze & Guven** — Authors of the two most directly relevant papers on temperature effects and non-deterministic evaluation. Their work on "The Effect of Sampling Temperature on Problem Solving in Large Language Models" (2024) and "The Good, The Bad, and The Greedy" (NAACL 2025) are the closest comparators to parameter sweep methodology.

**Panickssery, Bowman & Feng** — NeurIPS 2024 work on self-preference bias in LLM judges. Directly relevant to any study using a frontier model (like Opus) to judge its own outputs.

**Georgi Gerganov / ggml-org** — Creator of llama.cpp and the GGUF format. The foundational infrastructure that enables all local quantized inference work.

**Wang, Jiacheng et al.** — SLMQuant team. First systematic benchmark comparing quantization approaches specifically on small language models, establishing that LLM-optimised compression techniques don't transfer directly to the 7–9B parameter range.

**Apple MLX team (Hannun et al.)** — Developing the competing inference framework to llama.cpp that exploits unified memory architecture. Their benchmarks provide the hardware-specific context for Apple Silicon inference studies.

## 6. Thematic Analysis

### 6.1 Quantized Small Models vs. Frontier: The Narrowing Gap

The performance gap between quantized small models and frontier systems has been extensively benchmarked, though rarely on context-grounded Q&A specifically. SLMQuant (Wang et al., 2025) establishes that small language models respond to quantization fundamentally differently from larger models — "direct transfer of LLM-optimised techniques leads to suboptimal results." This is corroborated by the IJCAI 2025 finding that smaller models at higher bitwidths (e.g., Q4_K_M at 7B) outperform larger models at extreme quantization (e.g., 2-bit at 65B).

The most directly comparable benchmark to a Llama-3.1-8B local evaluation is the unified llama.cpp quantization study (arXiv 2601.14277, 2026), which tested every GGUF quantization format on Llama-3.1-8B-Instruct across GSM8K, HellaSwag, IFEval, MMLU, and TruthfulQA. That study confirms Q4_K_M as the optimal balance between size reduction and quality retention for this specific model — consistent with the ~75% size reduction / 95% quality retention heuristic from the broader quantization taxonomy (arXiv 2502.13178).

Red Hat's large-scale study (2024), running over 500,000 evaluations across quantized models, found that 8B models "experience slight variability when quantized but preserve core semantic meaning and structural coherence." This supports the observation that quantized 7–9B models remain competitive on factual tasks even at Q4 precision.

The Epoch AI analyses provide broader context: open-weights models now trail the frontier by approximately three months, and frontier capabilities become available on consumer hardware within roughly one year. For context-grounded Q&A — a task that doesn't require the frontier's most advanced reasoning — the gap is likely even smaller than the aggregate benchmarks suggest.

**What's missing:** Most benchmarks evaluate quantized models on standard tasks (MMLU, GSM8K, HumanEval) using fixed parameters. Studies comparing quantized small models to frontier specifically on **context-grounded Q&A with passage comprehension** — as opposed to knowledge recall or code generation — are notably absent. This is a gap that targeted evaluation can fill.

### 6.2 Temperature and Top-p: Less Important Than You'd Think

The effect of sampling parameters on factual Q&A quality is the area where the literature most directly addresses (and largely confirms) the finding that temperature has minimal effect on mean quality.

Renze & Guven (2024) provide the canonical result: "changes in temperature from 0.0 to 1.0 do not have a statistically significant impact on LLM performance for problem-solving tasks." Their results "appear to generalise across LLMs, prompt-engineering techniques, and problem domains." This aligns with clinical temperature studies (medRxiv, 2024) showing GPT-4 and Llama-3-70b maintain "remarkable consistency in performance across a variety of clinical tasks regardless of temperature settings."

The "Hot or Cold?" study (arXiv 2506.07295, 2025) adds an important nuance by size: small models (1–4B) are more temperature-sensitive than medium (6–13B) or large (40–80B) models. They introduce the "mutation temperature" concept — the threshold at which performance degrades — and show it increases with model size. For the 7–9B range, this places models at the transition boundary where temperature sensitivity is moderate but measurable.

Renze & Guven's companion paper "The Good, The Bad, and The Greedy" (NAACL 2025) argues forcefully that **benchmarks using only greedy decoding miss important information.** Sampling-based evaluation reveals performance differences between models that are invisible under deterministic decoding. This directly supports the methodological choice of running parameter sweeps rather than single-setting benchmarks.

On top-p specifically, the literature is sparser. Holtzman et al. (2020) established nucleus sampling's value for open-ended generation, but for factual tasks the consensus is that top-p has minimal effect. No paper in the survey explicitly tests top-p's irrelevance on small quantized models running factual Q&A — this is assumed rather than empirically demonstrated.

**What's missing:** The interaction between **quantization level and temperature sensitivity** is unstudied. Does Q4 quantization make a model more or less robust to temperature variation compared to Q8? Does the "mutation temperature" threshold shift with quantization? Additionally, no study compares temperature sensitivity **across model families** at the same parameter count — the finding that GLM is more parameter-sensitive than Llama or Gemma appears to be novel.

### 6.3 LLM-as-Judge: Reliable but Biased

The LLM-as-judge methodology has received extensive scrutiny since Zheng et al. (2023). Two comprehensive surveys (arXiv 2411.15594; arXiv 2412.05579) and multiple focused studies establish both its utility and its limitations.

The most relevant bias for a setup where Opus judges all models including itself is **self-preference bias**. Panickssery et al. (NeurIPS 2024) demonstrate that "LLM evaluators recognise and favour their own generations" with a "linear correlation between self-recognition capability and self-preference bias strength." The root cause is perplexity-based: models find text matching their own training distribution more "natural" and score it higher. GPT-4o and Claude 3.5 Sonnet both exhibit this bias, and models also show **family bias**, assigning higher ratings to outputs from the same model family.

This directly implies that an Opus-judged evaluation will systematically overrate Opus's own outputs. The magnitude of this effect depends on whether the evaluation uses pairwise comparison or absolute scoring. The empirical study by arXiv 2506.13639 (2025) finds that **providing reference answers and score descriptions is crucial for reliable evaluation** — omitting either "significantly degrades alignment with human judgements." This supports the design choice of including reference answers in the judge prompt. The same study finds non-deterministic sampling improves alignment with human preferences over deterministic evaluation, and that chain-of-thought reasoning offers "minimal gains when clear evaluation criteria are present."

Position bias (Wang et al., 2024) and verbosity bias (Saito, 2023; arXiv 2407.01085, 2024) are additional documented concerns. LLM judges tend to prefer longer outputs independent of quality, which means any evaluation using an LLM judge must account for the possibility that verbose models receive inflated scores.

**What's missing:** Self-preference bias has been studied for GPT-4o and Claude 3.5 Sonnet but **not for Opus 4.6 specifically**. The magnitude of bias may differ across model versions. No study has examined whether using a frontier model to judge itself *on context-grounded Q&A specifically* (where the answer is verifiable against a passage) produces less self-preference bias than on subjective tasks — a plausible hypothesis given that factual accuracy is less distribution-dependent.

### 6.4 Edge Inference on Apple Silicon: A Maturing Ecosystem

The Apple Silicon inference landscape has been comprehensively benchmarked in 2025–2026. The production-grade comparison (arXiv 2511.05502) tested five frameworks on Apple Silicon: MLX achieved ~230 tok/s, MLC-LLM ~190 tok/s, llama.cpp ~150 tok/s, Ollama 20–40 tok/s, and PyTorch MPS 7–9 tok/s. The peer-reviewed ACM SIGMETRICS study (2025) characterises memory bandwidth and GPU utilisation patterns.

The native Apple Silicon inference study (arXiv 2601.19139, 2026) evaluated across M2 Ultra, M2 Max, and M4 Pro, with the M4 Max reaching 525 tok/s. This provides hardware-specific context: on M2 with 16GB unified memory, performance is constrained by memory bandwidth, and only one model fits in memory at a time — a practical constraint that shapes experimental design.

The mobile platforms study (arXiv 2410.03613, 2024) adds power consumption and thermal data, showing that sustained inference on consumer devices involves thermal management tradeoffs not captured in throughput-only benchmarks.

**What's missing:** The inference framework benchmarks focus on throughput and latency but **do not simultaneously measure output quality across quantization levels**. Knowing that MLX is faster than llama.cpp is useful, but practitioners need to know whether the same model at the same quantization produces different quality outputs across frameworks (it shouldn't, but this hasn't been verified). The quality-vs-latency Pareto frontier that maps **different models at different quantizations** against each other on the same hardware is not present in the literature.

### 6.5 Verbosity and Quality: More Is Not Better

The relationship between response length and quality is one of the best-documented biases in LLM evaluation. "Verbosity ≠ Veracity" (arXiv 2411.07858, 2024) demonstrates "strong correlation between conciseness and correctness" — verbose responses correlate with high uncertainty and lower accuracy, as "models compensate for low confidence with extra text."

This finding is reinforced from the evaluation side: LLM judges exhibit verbosity bias, preferring longer outputs independent of quality (Saito, 2023; arXiv 2407.01085). The DPO literature (arXiv 2403.19159) shows that preference optimisation without length correction systematically favors longer outputs. Multiple papers propose causal frameworks for understanding and mitigating this bias in RLHF (arXiv 2511.12573; NAACL 2025).

The Chang et al. survey (ACM TIST, 2024) provides the broadest treatment, reviewing evaluation metrics and biases including verbosity-preference correlations across multiple LLM evaluation settings.

**What's missing:** The verbosity literature studies full-precision models. Whether **quantization changes verbosity patterns** — and whether the verbosity-quality anticorrelation holds for quantized small models specifically — has not been examined. The observation that Gemma 4 E4B at Q8 generates 3–7x more tokens than Llama 3.1 8B at Q4 while achieving lower completeness scores would be a novel data point connecting quantization format, model architecture, and the verbosity-quality relationship.

## 7. Tools, Software & Companies

### Inference Frameworks

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — C++ inference engine with GGUF quantization and Metal acceleration. Foundational infrastructure for local LLM inference. Open source, production-grade, actively maintained by Georgi Gerganov and community.

- **[MLX](https://github.com/ml-explore/mlx)** — Apple's ML framework optimised for unified memory on Apple Silicon. 20–87% faster than llama.cpp for models under 14B. Open source, production-grade.

- **[Ollama](https://ollama.com)** — Local LLM runner with developer-friendly CLI. Transitioned to MLX backend in v0.19 (2026). Free, open source. Lower throughput than direct llama.cpp/MLX but simpler setup.

- **[vllm-mlx](https://github.com/vllm-project/vllm-mlx)** — vLLM architecture ported to Apple MLX for high-throughput local inference. Open source, beta.

### Desktop Applications

- **[LM Studio](https://lmstudio.ai)** — GUI for running and benchmarking local LLMs. Supports llama.cpp and MLX backends. Free for personal use, proprietary.

- **[Jan](https://jan.ai)** — Open-source offline ChatGPT alternative supporting GGUF and TensorRT. Free.

- **[Faraday](https://faraday.dev)** — Native desktop app for local LLMs focused on Apple Silicon usability. Freemium.

### Model Providers

- **[Meta AI](https://ai.meta.com)** — Developer of the Llama series. Llama 3.1 8B is the most deployed model in the 8B range.

- **[Google DeepMind](https://deepmind.google)** — Developer of the Gemma series. Gemma 4 E4B uses Per-Layer Embedding architecture for edge efficiency.

- **[Zhipu AI](https://zhipu-ai.com)** — Developer of the GLM family. GLM-4-9B-Chat provides bilingual (Chinese-English) capability in the 9B range.

### Standards & Specifications

- **[GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)** — Binary format standard for distributing quantized models. De facto standard for local inference.

## 8. Gaps & Future Directions

Based on this review, the following gaps represent opportunities for novel contribution:

### Gap 1: Quantization × Sampling Parameter Interaction (Strong gap)
No study examines how **quantization level affects sampling parameter sensitivity**. Does Q4_K_M quantization make a model more or less robust to temperature variation than Q8? Does the "mutation temperature" threshold shift with quantization? An experiment comparing temperature sweeps on the same model at different quantization levels would directly address this.

### Gap 2: Model-Family Parameter Sensitivity at Matched Size (Moderate gap)
The finding that GLM-4-9B is significantly more parameter-sensitive than Llama 3.1 8B and Gemma 4 E4B has no direct comparator in the literature. Temperature studies either test one model family across sizes (Hot or Cold?, 2025) or test multiple models at one temperature (standard benchmarks). A **controlled comparison of parameter sensitivity across architectures at the same parameter count** would be novel.

### Gap 3: Context-Grounded Q&A as a Benchmark Task (Moderate gap)
Most quantization benchmarks use MMLU, GSM8K, HumanEval, or TruthfulQA. Context-grounded Q&A — where the answer must be extracted or synthesised from a provided passage — is underrepresented as an evaluation task for quantized small models despite being one of the most practical use cases (RAG, document Q&A).

### Gap 4: Quality-Latency Pareto Frontier on Consumer Hardware (Moderate gap)
Inference benchmarks report throughput. Quality benchmarks report scores. No study maps both simultaneously to produce a **quality-vs-latency Pareto frontier** for multiple models on the same consumer hardware. This is the chart a practitioner actually needs.

### Gap 5: Self-Preference Bias in Factual Evaluation (Narrow gap)
Self-preference bias is documented for subjective tasks. Whether it's attenuated for **factual Q&A with verifiable answers** — where correctness is less distribution-dependent — is untested. An Opus-judging-Opus setup on context-grounded Q&A provides a natural test case.

### Gap 6: Verbosity Patterns Under Quantization (Narrow gap)
The verbosity-quality anticorrelation is established for full-precision models. Whether quantization affects verbosity behaviour — and whether certain architectures (e.g., Gemma's observed 3–7x verbosity) change their output length distribution under different quantization levels — is unexplored.

### What a well-designed study would look like
A study addressing Gaps 1–4 simultaneously would:
- Select 3+ models from different families at matched parameter counts (7–9B)
- Run each at 2+ quantization levels (Q4_K_M and Q8_0 minimum)
- Sweep temperature (0.0, 0.3, 0.7, 1.0) and top_p (0.5, 0.9, 1.0) at each quantization
- Use context-grounded Q&A as the primary task (12+ items across factual, reasoning, synthesis)
- Report quality, latency, and token count jointly
- Use multi-judge evaluation or at minimum acknowledge and measure self-preference bias
- Produce a Pareto frontier chart mapping quality against latency

This design matches closely the experimental setup in the FINDINGS.md of the current project — suggesting the work already done partially fills these gaps.

## 9. Conclusion

The literature on quantized small LLM inference is mature in its component parts — quantization benchmarking, sampling parameter effects, LLM-as-judge methodology, and edge inference performance are each well-studied — but the **intersections between these areas remain sparse**. The most significant gap is the interaction between quantization and sampling parameters: nobody has systematically studied whether Q4 models respond differently to temperature tuning than Q8 models. Similarly, while the quality-vs-latency tradeoff is the most practical concern for local inference practitioners, no published study maps both axes simultaneously across multiple models on consumer hardware. The finding that t=0.3 consistently outperforms greedy decoding across three different model families is consistent with but extends beyond the existing evidence base, which has studied temperature effects primarily on individual model families or larger models. For a paper contextualising local inference experiments, the framing should emphasise these intersection gaps rather than competing on any single axis — the contribution is in the cross-cutting empirical evidence, not in any one benchmark result.

## 10. References

See `refs.csv` for the full scored and tagged reference database, and `refs.bib` for BibTeX export.
