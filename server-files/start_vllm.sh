#!/usr/bin/env bash
set -e

export PYTHONPATH=/workspace/python_pkgs
export HF_HOME=/workspace/hf
export TRANSFORMERS_CACHE=/workspace/hf

# Use environment variables with defaults
API_KEY="${VLLM_API_KEY:-rk_PLACEHOLDER}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# Log file for diagnostics
LOG_FILE="/workspace/vllm.log"

echo "Starting vLLM server..." | tee "$LOG_FILE"
echo "Model: /workspace/models/mistral-7b-instruct-awq" | tee -a "$LOG_FILE"
echo "Tensor Parallel Size: $TP_SIZE" | tee -a "$LOG_FILE"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/mistral-7b-instruct-awq \
  --gpu-memory-utilization 0.7 \
  --tensor-parallel-size "$TP_SIZE" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$API_KEY" \
  2>&1 | tee -a "$LOG_FILE"
