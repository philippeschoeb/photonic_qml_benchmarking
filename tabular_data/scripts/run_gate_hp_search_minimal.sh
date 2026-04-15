#!/usr/bin/env bash

# Run minimal hyperparameter searches for all supported models/backends on one dataset.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
SCRIPT_NAME="$(basename "$0" .sh)"
RUN_TS="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_GROUP="${SCRIPT_NAME}/${RUN_TS}"
RESULTS_ROOT="results/${RUN_GROUP}"
LOG_DIR="${RESULTS_ROOT}/logs"
mkdir -p "${LOG_DIR}"

DATASET="${1:-${DATASET:-hidden_manifold_2_6}}"
RANDOM_STATE="${RANDOM_STATE:-42}"
MAX_TRAIN_TIME_SECONDS="${MAX_TRAIN_TIME_SECONDS:-1800}"
USE_WANDB=0
WANDB_FLAG="--no-wandb"
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_FLAG=""
fi

MODELS=(
  "dressed_quantum_circuit"
  "dressed_quantum_circuit_reservoir"
  "multiple_paths_model"
  "multiple_paths_model_reservoir"
  "data_reuploading"
  "q_kernel_method_reservoir"
  "q_rks"
)

declare -A MODEL_BACKENDS=(
  ["dressed_quantum_circuit"]="gate"
  ["dressed_quantum_circuit_reservoir"]="gate"
  ["multiple_paths_model"]="gate"
  ["multiple_paths_model_reservoir"]="gate"
  ["data_reuploading"]="gate"
  ["q_kernel_method_reservoir"]="gate"
  ["q_rks"]="gate"
)

echo "====== Hyperparameter Search (minimal) Sweep ======"
echo "Dataset: ${DATASET}"
echo "Models: ${MODELS[*]}"
echo "Random state: ${RANDOM_STATE}"
echo "Max single-train time (s): ${MAX_TRAIN_TIME_SECONDS}"
echo "Output root: tabular_data/${RESULTS_ROOT}"
echo "Summary CSV: tabular_data/${RESULTS_ROOT}/hp_search_${DATASET}.csv"
echo "Logs dir: tabular_data/${LOG_DIR}"
echo

success_count=0
failure_count=0
failed_runs=()

run_single() {
  local model="$1"
  local backend_label="$2"
  shift 2
  local -a cmd=( "$@" )
  local safe_backend="${backend_label// /_}"
  local log_file="${LOG_DIR}/${model}__${safe_backend}.log"

  echo "-> Model: ${model}, Backend: ${backend_label}"
  printf '   Command:'
  printf ' %q' "${cmd[@]}"
  echo
  echo "   Log: tabular_data/${log_file}"

  "${cmd[@]}" 2>&1 | tee "${log_file}"
  local exit_code=${PIPESTATUS[0]}

  if [[ ${exit_code} -eq 0 ]]; then
    echo "   [OK] Completed: ${model} (${backend_label})"
    success_count=$((success_count + 1))
  else
    echo "   [FAIL] ${model} (${backend_label}) exited with code ${exit_code}"
    if [[ ${exit_code} -eq 137 ]]; then
      echo "   [INFO] Exit code 137 often indicates the process was killed (e.g., out-of-memory)."
    fi
    failed_runs+=("${model}|${backend_label}|${exit_code}|${log_file}")
    failure_count=$((failure_count + 1))
  fi
  echo
}

for model in "${MODELS[@]}"; do
  if [[ -z "${MODEL_BACKENDS[$model]+x}" ]]; then
    echo "    [SKIP] No backend configuration for model '${model}'."
    continue
  fi
  backends="${MODEL_BACKENDS["$model"]}"
  for backend in ${backends}; do
    cmd=(python -u main.py --dataset "${DATASET}" --model "${model}" --run_type hyperparam_search --hp_profile minimal --random_state "${RANDOM_STATE}" --max_train_time_seconds "${MAX_TRAIN_TIME_SECONDS}" --big_script_name "${RUN_GROUP}")
    if [[ -n "${WANDB_FLAG}" ]]; then
      cmd+=("${WANDB_FLAG}")
    fi

    if [[ "${backend}" == "classical" ]]; then
      run_single "${model}" "auto_classical" "${cmd[@]}"
    else
      run_single "${model}" "${backend}" "${cmd[@]}" --backend "${backend}"
    fi
  done
done

echo "====== Sweep completed ======"
echo "Successful runs: ${success_count}"
echo "Failed runs: ${failure_count}"

if [[ ${failure_count} -gt 0 ]]; then
  echo
  echo "Failed run summary:"
  for failure in "${failed_runs[@]}"; do
    IFS="|" read -r model backend_label exit_code log_file <<< "${failure}"
    echo " - ${model} (${backend_label}): exit ${exit_code}, log: tabular_data/${log_file}"
  done
  exit 1
fi
