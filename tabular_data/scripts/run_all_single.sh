#!/usr/bin/env bash

# Run single runs for all supported models/backends on one dataset.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET="downscaled_mnist_pca_10"

MODELS=(
  "dressed_quantum_circuit"
  "dressed_quantum_circuit_reservoir"
  "multiple_paths_model"
  "multiple_paths_model_reservoir"
  "data_reuploading"
  "q_kernel_method"
  "q_kernel_method_reservoir"
  "q_rks"
  "mlp"
  "rbf_svc"
  "rks"
)

declare -A MODEL_BACKENDS=(
  ["dressed_quantum_circuit"]="photonic gate"
  ["dressed_quantum_circuit_reservoir"]="photonic gate"
  ["multiple_paths_model"]="photonic gate"
  ["multiple_paths_model_reservoir"]="photonic gate"
  ["data_reuploading"]="photonic gate"
  ["q_kernel_method"]="photonic gate"
  ["q_kernel_method_reservoir"]="photonic gate"
  ["q_rks"]="photonic gate"
  ["mlp"]="classical"
  ["rbf_svc"]="classical"
  ["rks"]="classical"
)

echo "====== Single Run Sweep ======"
echo "Dataset: ${DATASET}"
echo "Models: ${MODELS[*]}"
echo

for model in "${MODELS[@]}"; do
  if [[ ! -v MODEL_BACKENDS["$model"] ]]; then
    echo "    [SKIP] No backend configuration for model '${model}'."
    continue
  fi
  backends="${MODEL_BACKENDS["$model"]}"
  for backend in ${backends}; do
    echo "-> Model: ${model}, Backend: ${backend}"
    python main.py --dataset "${DATASET}" --model "${model}" --backend "${backend}" --run_type single
  done
done

echo "====== Sweep completed ======"
