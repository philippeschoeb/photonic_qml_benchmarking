#!/usr/bin/env bash

# Run minimal hyperparameter searches for all supported models/backends on one dataset,
# plus ablation variants where supported.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
SCRIPT_NAME="$(basename "$0" .sh)"
RUN_TS="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_GROUP="${RUN_GROUP_OVERRIDE:-${SCRIPT_NAME}/${RUN_TS}}"
RESULTS_ROOT="results/${RUN_GROUP}"
LOG_DIR="${RESULTS_ROOT}/logs"
mkdir -p "${LOG_DIR}"

DATASET="${1:-${DATASET:-hidden_manifold_10_10}}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
if [[ "${2:-}" == "--with-ablation" ]]; then
  RUN_ABLATIONS=1
fi
HP_PROFILE="${HP_PROFILE:-minimal}"
MODELS_OVERRIDE="${MODELS_OVERRIDE:-all}"
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
  "q_kernel_method"
  "q_kernel_method_reservoir"
  "q_rks"
  "mlp"
  "rbf_svc"
  "rks"
)

if [[ -n "${MODELS_OVERRIDE}" && "${MODELS_OVERRIDE}" != "all" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODELS_OVERRIDE}"
fi

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

can_run_quantum_ablation() {
  local model="$1"
  local backend="$2"
  if [[ "${backend}" != "photonic" ]]; then
    return 1
  fi
  case "${model}" in
    dressed_quantum_circuit|dressed_quantum_circuit_reservoir|multiple_paths_model|multiple_paths_model_reservoir|q_rks)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

can_run_classical_ablation() {
  local model="$1"
  local backend="$2"
  if [[ "${backend}" != "photonic" ]]; then
    return 1
  fi
  case "${model}" in
    dressed_quantum_circuit|multiple_paths_model)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

echo "====== Hyperparameter Search (minimal) + Ablations Sweep ======"
echo "Dataset: ${DATASET}"
echo "Run ablations: ${RUN_ABLATIONS} (set arg '--with-ablation' or env RUN_ABLATIONS=1)"
echo "HP profile: ${HP_PROFILE}"
echo "Models: ${MODELS[*]}"
echo "Output root: tabular_data/${RESULTS_ROOT}"
echo "Summary CSV: tabular_data/${RESULTS_ROOT}/hp_search_${DATASET}.csv"
echo "Logs dir: tabular_data/${LOG_DIR}"
echo

success_count=0
failure_count=0
failed_runs=()

run_single() {
  local model_label="$1"
  local backend_label="$2"
  shift 2
  local -a cmd=( "$@" )
  local safe_backend="${backend_label// /_}"
  local log_file="${LOG_DIR}/${DATASET}__${model_label}__${safe_backend}.log"

  echo "-> Model: ${model_label}, Backend: ${backend_label}"
  printf '   Command:'
  printf ' %q' "${cmd[@]}"
  echo
  echo "   Log: tabular_data/${log_file}"

  "${cmd[@]}" 2>&1 | tee "${log_file}"
  local exit_code=${PIPESTATUS[0]}

  if [[ ${exit_code} -eq 0 ]]; then
    echo "   [OK] Completed: ${model_label} (${backend_label})"
    success_count=$((success_count + 1))
  else
    echo "   [FAIL] ${model_label} (${backend_label}) exited with code ${exit_code}"
    if [[ ${exit_code} -eq 137 ]]; then
      echo "   [INFO] Exit code 137 often indicates the process was killed (e.g., out-of-memory)."
    fi
    failed_runs+=("${model_label}|${backend_label}|${exit_code}|${log_file}")
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
    cmd=(
      python -u main.py
      --dataset "${DATASET}"
      --model "${model}"
      --run_type hyperparam_search
      --hp_profile "${HP_PROFILE}"
      --big_script_name "${RUN_GROUP}"
    )
    if [[ -n "${WANDB_FLAG}" ]]; then
      cmd+=("${WANDB_FLAG}")
    fi
    if [[ "${backend}" == "classical" ]]; then
      run_single "${model}" "auto_classical" "${cmd[@]}"
    else
      run_single "${model}" "${backend}" "${cmd[@]}" --backend "${backend}"
    fi

    if [[ "${RUN_ABLATIONS}" == "1" ]]; then
      if can_run_quantum_ablation "${model}" "${backend}"; then
        abla_model_q="${model}_abla_q"
        cmd_abla_q=(
          python -u main.py
          --dataset "${DATASET}"
          --model "${abla_model_q}"
          --run_type hyperparam_search
          --hp_profile "${HP_PROFILE}"
          --big_script_name "${RUN_GROUP}"
        )
        if [[ -n "${WANDB_FLAG}" ]]; then
          cmd_abla_q+=("${WANDB_FLAG}")
        fi
        run_single "${abla_model_q}" "${backend}" "${cmd_abla_q[@]}" --backend "${backend}"
      fi

      if can_run_classical_ablation "${model}" "${backend}"; then
        abla_model_c="${model}_abla_c"
        cmd_abla_c=(
          python -u main.py
          --dataset "${DATASET}"
          --model "${abla_model_c}"
          --run_type hyperparam_search
          --hp_profile "${HP_PROFILE}"
          --big_script_name "${RUN_GROUP}"
        )
        if [[ -n "${WANDB_FLAG}" ]]; then
          cmd_abla_c+=("${WANDB_FLAG}")
        fi
        run_single "${abla_model_c}" "${backend}" "${cmd_abla_c[@]}" --backend "${backend}"
      fi
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
    IFS="|" read -r model_label backend_label exit_code log_file <<< "${failure}"
    echo " - ${model_label} (${backend_label}): exit ${exit_code}, log: tabular_data/${log_file}"
  done
  exit 1
fi
