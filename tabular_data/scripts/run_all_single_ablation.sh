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
RANDOM_STATE="${RANDOM_STATE:-42}"
MAX_TRAIN_TIME_SECONDS="${MAX_TRAIN_TIME_SECONDS:-1800}"
BATCH_RUN_GROUP="${SCRIPT_NAME}/${DATASET}_${BATCH_TIMESTAMP}"
RESULTS_ROOT="results/${BATCH_RUN_GROUP}"
LONG_TRAINING_FILE="${RESULTS_ROOT}/long_training_${DATASET}.jsonl"
mkdir -p "${RESULTS_ROOT}"

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
echo "Random state: ${RANDOM_STATE}"
echo "Max single-train time (s): ${MAX_TRAIN_TIME_SECONDS}"
echo "Run group: ${BATCH_RUN_GROUP}"
echo "Models: ${MODELS[*]}"
echo

WANDB_FLAG=()
if [[ "${USE_WANDB}" != "true" ]]; then
  WANDB_FLAG+=(--no-wandb)
fi

run_single_cmd() {
  local model_label="$1"
  local backend_label="$2"
  shift
  shift
  local -a cmd=( "$@" )
  if command -v timeout >/dev/null 2>&1; then
    if timeout "${MAX_TRAIN_TIME_SECONDS}" "${cmd[@]}"; then
      return 0
    fi
    local exit_code=$?
    if [[ ${exit_code} -eq 124 ]]; then
      echo "   [TIMEOUT] ${model_label} exceeded ${MAX_TRAIN_TIME_SECONDS}s. Nothing is returned for this run (process terminated by timeout)."
      printf '{"timestamp":"%s","dataset":"%s","model":"%s","backend":"%s","run_type":"single","status":"skipped","event_type":"process_timeout","source":"shell_timeout","reason":"Process terminated by timeout command before run completion.","max_train_time_seconds":%s,"hyperparameters":{"dataset":"%s","model":"%s","backend":"%s","random_state":%s,"max_train_time_seconds":%s}}\n' \
        "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "${DATASET}" "${model_label}" "${backend_label}" "${MAX_TRAIN_TIME_SECONDS}" \
        "${DATASET}" "${model_label}" "${backend_label}" "${RANDOM_STATE}" "${MAX_TRAIN_TIME_SECONDS}" >> "${LONG_TRAINING_FILE}"
    fi
    return ${exit_code}
  fi
  echo "   [WARN] 'timeout' not found; relying on in-training time budget only."
  "${cmd[@]}"
}

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
        run_single_cmd "${variant_model}" "classical" python main.py --dataset "${DATASET}" --model "${variant_model}" --run_type single --random_state "${RANDOM_STATE}" --max_train_time_seconds "${MAX_TRAIN_TIME_SECONDS}" --big_script_name "${BATCH_RUN_GROUP}" "${WANDB_FLAG[@]}"
      else
        echo "-> Model: ${variant_model}, Backend: ${backend}"
        run_single_cmd "${variant_model}" "${backend}" python main.py --dataset "${DATASET}" --model "${variant_model}" --backend "${backend}" --run_type single --random_state "${RANDOM_STATE}" --max_train_time_seconds "${MAX_TRAIN_TIME_SECONDS}" --big_script_name "${BATCH_RUN_GROUP}" "${WANDB_FLAG[@]}"
      fi
    done
  done
done

echo "====== Sweep completed ======"
