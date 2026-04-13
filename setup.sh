#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$HOME/models"

echo "=== Local LLM Eval: Environment Setup ==="
echo ""

# 1. Install llama.cpp via Homebrew
if command -v llama-server &>/dev/null; then
    echo "[ok] llama-server already installed: $(which llama-server)"
else
    echo "[install] Installing llama.cpp via Homebrew..."
    brew install llama.cpp
    echo "[ok] llama-server installed: $(which llama-server)"
fi

# 2. Install huggingface hub CLI
if command -v hf &>/dev/null; then
    echo "[ok] hf CLI already installed"
else
    echo "[install] Installing huggingface-hub..."
    pip install huggingface-hub
    echo "[ok] hf CLI installed"
fi

# 3. Create model directory
mkdir -p "$MODEL_DIR"
echo "[ok] Model directory: $MODEL_DIR"
echo ""

# 4. Download models
echo "=== Downloading models (this may take a while) ==="
echo ""

# GLM-4-9B-Chat Q4_K_M (~5.5 GB)
if ls "$MODEL_DIR"/glm-4-9b-chat-Q4_K_M.gguf &>/dev/null; then
    echo "[ok] GLM-4-9B-Chat already downloaded"
else
    echo "[download] GLM-4-9B-Chat Q4_K_M..."
    hf download bartowski/glm-4-9b-chat-GGUF \
        --include "glm-4-9b-chat-Q4_K_M.gguf" \
        --local-dir "$MODEL_DIR"
fi

# Gemma 4 E4B-it Q8_0 (~8.0 GB)
if ls "$MODEL_DIR"/google_gemma-4-E4B-it-Q8_0.gguf &>/dev/null; then
    echo "[ok] Gemma 4 E4B-it already downloaded"
else
    echo "[download] Gemma 4 E4B-it Q8_0..."
    hf download bartowski/google_gemma-4-E4B-it-GGUF \
        --include "google_gemma-4-E4B-it-Q8_0.gguf" \
        --local-dir "$MODEL_DIR"
fi

# Llama 3.1 8B Instruct Q4_K_M (~5.0 GB)
if ls "$MODEL_DIR"/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf &>/dev/null; then
    echo "[ok] Llama 3.1 8B Instruct Q4 already downloaded"
else
    echo "[download] Llama 3.1 8B Instruct Q4_K_M..."
    hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
        --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
        --local-dir "$MODEL_DIR"
fi

# Llama 3.1 8B Instruct Q5_K_M (~5.7 GB)
if ls "$MODEL_DIR"/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf &>/dev/null; then
    echo "[ok] Llama 3.1 8B Instruct Q5 already downloaded"
else
    echo "[download] Llama 3.1 8B Instruct Q5_K_M..."
    hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
        --include "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf" \
        --local-dir "$MODEL_DIR"
fi

# Gemma 4 E4B-it Q4_K_M (~4.5 GB)
if ls "$MODEL_DIR"/google_gemma-4-E4B-it-Q4_K_M.gguf &>/dev/null; then
    echo "[ok] Gemma 4 E4B-it Q4 already downloaded"
else
    echo "[download] Gemma 4 E4B-it Q4_K_M..."
    hf download bartowski/google_gemma-4-E4B-it-GGUF \
        --include "google_gemma-4-E4B-it-Q4_K_M.gguf" \
        --local-dir "$MODEL_DIR"
fi

# Qwen 2.5 7B Instruct Q4_K_M (~4.7 GB)
if ls "$MODEL_DIR"/Qwen2.5-7B-Instruct-Q4_K_M.gguf &>/dev/null; then
    echo "[ok] Qwen 2.5 7B Instruct already downloaded"
else
    echo "[download] Qwen 2.5 7B Instruct Q4_K_M..."
    hf download bartowski/Qwen2.5-7B-Instruct-GGUF \
        --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
        --local-dir "$MODEL_DIR"
fi

# DeepSeek-R1-Distill-Qwen-7B Q4_K_M (~4.7 GB)
if ls "$MODEL_DIR"/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf &>/dev/null; then
    echo "[ok] DeepSeek-R1-Distill-Qwen-7B already downloaded"
else
    echo "[download] DeepSeek-R1-Distill-Qwen-7B Q4_K_M..."
    hf download bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF \
        --include "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf" \
        --local-dir "$MODEL_DIR"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Models downloaded to $MODEL_DIR:"
ls -lh "$MODEL_DIR"/*.gguf 2>/dev/null || echo "  (no .gguf files found — check download logs above)"
echo ""
echo "Next: run 'python eval_harness.py --quick' after completing Phase 2"
