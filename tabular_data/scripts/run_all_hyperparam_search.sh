#!/usr/bin/env bash

# Runs hyperparameter searches for all supported models/backends/datasets.
# Applies safeguards on runtime (skip after 1h) and suggests heavier search
# if searches finish suspiciously fast (<2 minutes for non-grid strategies).

set -uo pipefail

if ! command -v timeout >/dev/null 2>&1; then
  echo "ERROR: GNU timeout is required but not found in PATH." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASETS=(
  "downscaled_mnist_pca_2"
  "downscaled_mnist_pca_10"
  "downscaled_mnist_pca_20"
)

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
)  # search type derived below

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

declare -A MODEL_SEARCH_TYPE=()
eval "$(
  python - "$ROOT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cfg = root / "hyperparameters/hyperparam_search/model_search_assignment.json"
with cfg.open() as fh:
    data = json.load(fh)

def emit(models, label):
    for name in models:
        print(f'MODEL_SEARCH_TYPE["{name}"]="{label}"')

emit(data.get("halving_grid", []), "halving_grid")
emit(data.get("bayes", []), "bayes_search")
emit(data.get("grid", []), "grid_search")
PY
)"

HALVING_OR_BAYES=("halving_grid" "bayes_search")
TIMEOUT_SECONDS=$((60 * 60))   # 1 hour
FAST_THRESHOLD=$((2 * 60))     # 2 minutes

is_fast_strategy() {
  local model="$1"
  local strategy="${MODEL_SEARCH_TYPE[$model]:-}"
  if [[ -z "${strategy}" ]]; then
    return 1
  fi
  for candidate in "${HALVING_OR_BAYES[@]}"; do
    if [[ "${strategy}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

echo "====== Hyperparameter Search Sweep ======"
echo "Root directory: ${ROOT_DIR}"
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo

for dataset in "${DATASETS[@]}"; do
  echo ">>> Dataset: ${dataset}"
  for model in "${MODELS[@]}"; do
    # Use '!' test operator to safely check for missing keys
    if [[ ! -v MODEL_BACKENDS["$model"] ]]; then
      echo "    [SKIP] No backend configuration for model '${model}'."
      continue
    fi

    backends="${MODEL_BACKENDS["$model"]}"
    echo "    Using backend(s): ${backends}"

    for backend in ${backends}; do
      search_type="${MODEL_SEARCH_TYPE[$model]:-unknown}"
      echo "    -> Model: ${model}, Backend: ${backend} (search: ${search_type})"
      cmd=(python main.py --dataset "${dataset}" --model "${model}" --backend "${backend}" --run_type hyperparam_search)
      start_time=$(date +%s)
      if timeout "${TIMEOUT_SECONDS}" "${cmd[@]}"; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        printf "       Completed in %dm%02ds\n" $((elapsed / 60)) $((elapsed % 60))

        if [[ ${elapsed} -lt ${FAST_THRESHOLD} ]] && is_fast_strategy "${model}"; then
          echo "       NOTE: Search finished in under 2 minutes. Consider using GridSearchCV for a more exhaustive sweep."
        fi
      else
        exit_code=$?
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        if [[ ${exit_code} -eq 124 ]]; then
          printf "       SKIPPED: exceeded 1 hour timeout after %dm%02ds\n" $((elapsed / 60)) $((elapsed % 60))
        else
          printf "       FAILED (exit code %d) after %dm%02ds\n" "${exit_code}" $((elapsed / 60)) $((elapsed % 60))
        fi
      fi
    done
  done
  echo
done

echo "====== Sweep completed at $(timestamp) ======"
