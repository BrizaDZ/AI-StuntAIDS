#!/bin/bash

# 1. Jalankan Ollama
ollama serve &

# 2. Load model LLaMA 3 (optional, jika belum di-pull)
ollama pull llama3 &

# 3. Jalankan FastAPI
cd /workspace
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
