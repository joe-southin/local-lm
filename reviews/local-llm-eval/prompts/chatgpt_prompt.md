I am conducting a multi-source systematic literature review. I need your help finding relevant references. This is part of a pipeline where results from multiple AI assistants are merged and deduplicated, so please follow the output format EXACTLY.

TOPIC: Performance of quantized small LLMs (7-9B parameters) on consumer hardware compared to frontier models, with emphasis on sampling parameter effects (temperature, top_p) on Q&A task quality, and LLM-as-judge evaluation methodology.

RESEARCH QUESTIONS:
1. How do quantized 7-9B parameter open-weights models (Llama 3.1 8B, Gemma 4 E4B, GLM-4-9B) compare to frontier models on context-grounded Q&A tasks?
2. What effect do sampling parameters (temperature, top_p) have on factual Q&A quality, and does the effect differ between model families?
3. How reliable and valid is the LLM-as-judge methodology for evaluating model output quality?
4. What are the quality-latency-memory tradeoffs for local inference on consumer-grade Apple Silicon hardware?
5. Does response verbosity correlate with answer quality, and how do quantization levels affect this?

TIME RANGE: 2020-01-01 to 2026-04-12

KEY SEARCH TERMS: local LLM inference, small language models, quantized LLM, LLM benchmarking, LLM-as-judge, sampling parameters, temperature decoding, edge AI inference, on-device LLM, GGUF quantization, model compression, nucleus sampling top-p, greedy decoding, consumer hardware AI, Apple Silicon inference, llama.cpp

INCLUSION CRITERIA:
- Empirical evaluation or benchmarking of language models (any size)
- Studies of LLM sampling/decoding parameters and their effect on output quality
- LLM-as-judge methodology papers (reliability, bias, validity)
- Quantization effect studies (GPTQ, GGUF, AWQ, etc.)
- Edge/consumer/on-device LLM inference performance studies
- Systematic reviews or meta-analyses of LLM evaluation methods
- Tools and frameworks for local LLM inference (llama.cpp, Ollama, MLX, etc.)

EXCLUSION CRITERIA:
- Papers focused solely on training or fine-tuning (not inference)
- Studies using only proprietary/closed models with no local inference component
- Papers on models >70B parameters with no relevance to consumer hardware
- Non-English language papers
- Marketing materials disguised as benchmarks

---

YOUR TASK:

Search broadly for relevant references. I specifically want you to:
1. Find peer-reviewed papers directly on this topic
2. Look in ADJACENT fields that study similar problems with different terminology
3. Find relevant tools, software platforms, and companies working in this space
4. Include government reports, institutional publications, and standards documents
5. Include 2-3 foundational/seminal papers even if outside the time range
6. Search for tools and platforms that the END-USER of this research might buy or use — even if they don't use the same terminology as the research literature. Think about what a practitioner, buyer, or decision-maker would search for.
   Also specifically search for: local AI assistant, private LLM deployment, offline AI chatbot, llama.cpp benchmark, Ollama benchmark comparison, LM Studio performance, MLX inference benchmark

---

OUTPUT FORMAT — FOLLOW THIS EXACTLY:

Return ONLY a plain text list. One reference per line. Eight fields separated by pipe characters ( | ).

The format is:
TITLE | AUTHORS | YEAR | SOURCE | DOI | URL | TYPE | DESCRIPTION

Field rules:
- TITLE: Full title. No quotes. No truncation.
- AUTHORS: Semicolon-separated. Use "Last, First" format. Example: Smith, John; Doe, Jane
- YEAR: 4 digits only. Example: 2023
- SOURCE: Journal name, conference, website, or publisher
- DOI: The DOI string only (example: 10.1038/s41586-023-06185-3). Leave blank if none.
- URL: Full URL starting with https://. For DOI papers use https://doi.org/DOI. For tools use homepage.
- TYPE: Exactly one of these words: paper, preprint, tool, report, blog, book, software, company
- DESCRIPTION: One or two sentences. MUST NOT contain the pipe character |

EXAMPLE LINES (copy this format exactly):

Flood risk reduction using wetland restoration | Martinez, Elena; Chen, Wei | 2022 | Journal of Hydrology | 10.1016/j.jhydrol.2022.128456 | https://doi.org/10.1016/j.jhydrol.2022.128456 | paper | Quantifies flood peak reduction from restored wetlands across 12 UK catchments using paired analysis.
InVEST Natural Capital Model | Sharp, Richard; Tallis, Heather | 2020 | Stanford Natural Capital Project | | https://naturalcapitalproject.stanford.edu/software/invest | tool | Open-source suite of models for mapping and valuing ecosystem services including flood mitigation.
Treeconomy | | 2020 | | | https://www.treeconomy.co | company | Platform for natural capital project origination and monitoring using satellite data.

---

CRITICAL RULES:
1. DO NOT wrap your response in markdown code blocks (no ``` marks)
2. DO NOT add a header row (no "TITLE | AUTHORS | ..." header line)
3. DO NOT number the lines
4. DO NOT add any preamble text before the first reference line
5. DO NOT add commentary between reference lines
6. Start your response with the first reference line immediately
7. After ALL reference lines, add a blank line and then you may add:
   - THEMES: 2-3 major themes you noticed
   - GAPS: areas that seem under-researched
   - SUGGESTED_TERMS: 2-3 additional search terms I should try
8. Only include references you are CONFIDENT actually exist. If unsure, omit it.
9. Aim for 15-25 references. Quality over quantity.

Begin.
