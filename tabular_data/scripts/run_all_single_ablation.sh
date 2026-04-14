#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

# Run single runs for all supported models/backends on one dataset,
# plus all compatible ablation variants.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
SCRIPT_NAME="$(basename "$0" .sh)"
BATCH_TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

DATASET="hidden_manifold_2_6"
USE_WANDB=false
BATCH_RUN_GROUP="${SCRIPT_NAME}/${DATASET}_${BATCH_TIMESTAMP}"

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

echo "====== Single Run Sweep (with Ablations) ======"
echo "Dataset: ${DATASET}"
echo "Use Weights & Biases: ${USE_WANDB}"
echo "Run group: ${BATCH_RUN_GROUP}"
echo "Models: ${MODELS[*]}"
echo

WANDB_FLAG=()
if [[ "${USE_WANDB}" != "true" ]]; then
  WANDB_FLAG+=(--no-wandb)
fi

for model in "${MODELS[@]}"; do
  if [[ ! -v MODEL_BACKENDS["$model"] ]]; then
    echo "    [SKIP] No backend configuration for model '${model}'."
    continue
  fi
  backends="${MODEL_BACKENDS["$model"]}"
  for backend in ${backends}; do
    variants=("${model}")

    # Add all compatible single-run ablation variants.
    if [[ "${backend}" == "photonic" ]]; then
      case "${model}" in
        dressed_quantum_circuit|multiple_paths_model)
          variants+=("${model}_abla_q" "${model}_abla_c")
          ;;
        dressed_quantum_circuit_reservoir|multiple_paths_model_reservoir|q_rks)
          variants+=("${model}_abla_q")
          ;;
      esac
    fi

    for variant_model in "${variants[@]}"; do
      if [[ "${backend}" == "classical" ]]; then
        echo "-> Model: ${variant_model}, Backend: auto (classical)"
        python main.py --dataset "${DATASET}" --model "${variant_model}" --run_type single --big_script_name "${BATCH_RUN_GROUP}" "${WANDB_FLAG[@]}"
      else
        echo "-> Model: ${variant_model}, Backend: ${backend}"
        python main.py --dataset "${DATASET}" --model "${variant_model}" --backend "${backend}" --run_type single --big_script_name "${BATCH_RUN_GROUP}" "${WANDB_FLAG[@]}"
      fi
    done
  done
done

echo "====== Sweep completed ======"
