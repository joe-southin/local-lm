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

BOOLEAN QUERIES TO TRY:
1. (quantized OR quantised) AND (LLM OR "language model") AND (benchmark OR evaluation)
2. "LLM as judge" OR "LLM judge" OR "model-based evaluation"
3. "small language model" AND (temperature OR "top-p" OR "sampling parameters")
4. ("local inference" OR "edge inference" OR "on-device") AND (LLM OR transformer) AND (performance OR latency)
5. "GGUF" OR "llama.cpp" AND (benchmark OR comparison)

---

YOUR TASK:

Please search Google Scholar and other sources. I specifically want you to focus on:
1. RECENT papers (last 1-2 years) that may not be widely cited yet
2. Preprints on arXiv, bioRxiv, SSRN, EarthArXiv, or other discipline-appropriate servers
3. Tools, software platforms, open-source projects, and SaaS products relevant to this topic
4. Companies and organisations active in this space
5. Highly-cited foundational papers that define the field
6. Government reports, technical standards, and guidelines
7. Tools and platforms that the END-USER of this research might buy or use — even if they don't use the same terminology as the research literature. Think about what a practitioner or buyer would search for.
   Also specifically search for: local AI assistant, private LLM deployment, offline AI chatbot, llama.cpp benchmark, Ollama benchmark comparison, LM Studio performance, MLX inference benchmark

---

OUTPUT FORMAT — FOLLOW THIS EXACTLY:

Return ONLY a plain text list. One reference per line. Eight fields separated by pipe characters ( | ).

The format is:
TITLE | AUTHORS | YEAR | SOURCE | DOI | URL | TYPE | DESCRIPTION

Field rules:
- TITLE: Full title. No quotes. No truncation.
- AUTHORS: Semicolon-separated. Use "Last, First" format. Example: Kumar, Prashant; Lee, Su-Jin
- YEAR: 4 digits only. Example: 2024
- SOURCE: Journal name, conference, website, or publisher
- DOI: The DOI string only (example: 10.1016/j.ecoleng.2024.107289). Leave blank if none.
- URL: Full URL starting with https://. For DOI papers use https://doi.org/DOI. For tools use project homepage.
- TYPE: Exactly one of these words: paper, preprint, tool, report, blog, book, software, company
- DESCRIPTION: One or two sentences. MUST NOT contain the pipe character |

EXAMPLE LINES (copy this exact formatting):

Remote sensing of riparian buffer effectiveness | Park, Soo-Hyun; Williams, Tom | 2024 | Remote Sensing of Environment | 10.1016/j.rse.2024.114087 | https://doi.org/10.1016/j.rse.2024.114087 | paper | Uses Sentinel-2 time series to quantify flood attenuation by riparian buffers in three Welsh catchments.
Vizzuality | | 2015 | | | https://www.vizzuality.com | company | Design and technology company building data visualisation tools for conservation and climate organisations.
Catchment flood management using ML predictions | Abbas, Syed; O'Brien, Niamh | 2025 | EarthArXiv | 10.31223/X5KW9T | https://doi.org/10.31223/X5KW9T | preprint | Proposes a gradient-boosted model for predicting NbS intervention effectiveness at catchment scale.

---

CRITICAL RULES:
1. DO NOT wrap your response in markdown code blocks (no ``` marks)
2. DO NOT add a header row
3. DO NOT number the lines
4. DO NOT add any preamble text before the first reference line
5. DO NOT add commentary between reference lines
6. Start your response immediately with the first reference line
7. After ALL reference lines, add a blank line then you may add:
   - LANDSCAPE: 3-5 sentence summary of the state of the field
   - KEY_RESEARCHERS: Names and affiliations of leading groups
   - VENUES: Conferences or journals where this work appears
   - SUGGESTED_TERMS: 2-3 additional search terms to try
8. Only include references you are CONFIDENT actually exist. Do NOT fabricate.
9. Aim for 15-25 references. Quality over quantity.

Begin.
