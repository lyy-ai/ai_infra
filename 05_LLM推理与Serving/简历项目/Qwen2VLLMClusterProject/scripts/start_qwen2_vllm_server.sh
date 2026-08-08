#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/liyangyang/models/Qwen2-0.5B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen2-0.5b-instruct}
PORT=${PORT:-8000}
GPU_ID=${GPU_ID:-0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
PYTHON_BIN=${PYTHON_BIN:-/data/liyangyang/qwen35_env/bin/python}

PREFIX_FLAG=""
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  PREFIX_FLAG="--enable-prefix-caching"
fi

echo "Starting vLLM server"
echo "  model: ${MODEL_PATH}"
echo "  served name: ${SERVED_MODEL_NAME}"
echo "  gpu: ${GPU_ID}, port: ${PORT}"
echo "  max_num_seqs: ${MAX_NUM_SEQS}, gpu_mem_util: ${GPU_MEM_UTIL}, max_model_len: ${MAX_MODEL_LEN}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PATH=/data/liyangyang/qwen35_env/bin:$PATH \
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --dtype float16 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    ${PREFIX_FLAG} \
    --port "${PORT}" \
    ${EXTRA_ARGS:-}
