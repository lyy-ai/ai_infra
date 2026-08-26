#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/models/Qwen2-0.5B-Instruct}
GPU_IDS=${GPU_IDS:-"0"}
PORTS=${PORTS:-"8000"}
PID_FILE=${PID_FILE:-vllm_qwen2_cluster.pids}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

read -r -a GPU_ARR <<< "${GPU_IDS}"
read -r -a PORT_ARR <<< "${PORTS}"

if [[ "${#GPU_ARR[@]}" -ne "${#PORT_ARR[@]}" ]]; then
  echo "GPU_IDS and PORTS must have the same length"
  exit 1
fi

: > "${PID_FILE}"
for i in "${!GPU_ARR[@]}"; do
  gpu_id="${GPU_ARR[$i]}"
  port="${PORT_ARR[$i]}"
  log_file="vllm_qwen2_${port}.log"
  echo "starting instance: gpu=${gpu_id}, port=${port}, log=${log_file}"
  MODEL_PATH="${MODEL_PATH}" GPU_ID="${gpu_id}" PORT="${port}" \
    "${SCRIPT_DIR}/start_qwen2_vllm_server.sh" > "${log_file}" 2>&1 &
  echo $! >> "${PID_FILE}"
done

echo "cluster started; pids saved to ${PID_FILE}"
echo "router example: upstream 127.0.0.1:${PORT_ARR[*]}"
echo "stop with: xargs -a ${PID_FILE} kill"
