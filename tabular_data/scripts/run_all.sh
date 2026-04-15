#!/usr/bin/env bash

# Big weekend orchestrator for tabular minimal HP sweeps across many datasets.
# Reads config from scripts/run_all_config.json (or optional first arg path).

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
RANDOM_STATE="${RANDOM_STATE:-42}"
MAX_TRAIN_TIME_SECONDS="${MAX_TRAIN_TIME_SECONDS:-1800}"

CONFIG_PATH="${1:-scripts/run_all_config.json}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: tabular_data/${CONFIG_PATH}"
  exit 1
fi

eval "$(python - "${CONFIG_PATH}" <<'PY'
import json
import shlex
import sys

config_path = sys.argv[1]
with open(config_path, "r") as f:
    cfg = json.load(f)

datasets = cfg.get("datasets")
if not isinstance(datasets, list) or len(datasets) == 0:
    raise SystemExit("Invalid config: `datasets` must be a non-empty list.")

hp_profile = cfg.get("hp_profile", "minimal")
if hp_profile not in {"minimal", "full"}:
    raise SystemExit("Invalid config: `hp_profile` must be `minimal` or `full`.")

run_ablations = "1" if bool(cfg.get("run_ablations", False)) else "0"

models = cfg.get("models", "all")
if models == "all":
    models_override = "all"
else:
    if not isinstance(models, list) or len(models) == 0:
        raise SystemExit("Invalid config: `models` must be `all` or a non-empty list.")
    models_override = ",".join(str(m) for m in models)

print(f'HP_PROFILE={shlex.quote(hp_profile)}')
print(f'RUN_ABLATIONS={shlex.quote(run_ablations)}')
print(f'MODELS_OVERRIDE={shlex.quote(models_override)}')
print("DATASETS=(" + " ".join(shlex.quote(str(d)) for d in datasets) + ")")
PY
)"

SCRIPT_NAME="$(basename "$0" .sh)"
RUN_TS="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_GROUP="${SCRIPT_NAME}/${RUN_TS}"
RESULTS_ROOT="results/${RUN_GROUP}"
LOG_DIR="${RESULTS_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/run_all.log"
mkdir -p "${LOG_DIR}"

echo "====== RUN ALL (tabular minimal hp search) ======" | tee -a "${MASTER_LOG}"
echo "Config: tabular_data/${CONFIG_PATH}" | tee -a "${MASTER_LOG}"
echo "Datasets: ${DATASETS[*]}" | tee -a "${MASTER_LOG}"
echo "Run ablations: ${RUN_ABLATIONS}" | tee -a "${MASTER_LOG}"
echo "HP profile: ${HP_PROFILE}" | tee -a "${MASTER_LOG}"
echo "Random state: ${RANDOM_STATE}" | tee -a "${MASTER_LOG}"
echo "Max single-train time (s): ${MAX_TRAIN_TIME_SECONDS}" | tee -a "${MASTER_LOG}"
echo "Models override: ${MODELS_OVERRIDE}" | tee -a "${MASTER_LOG}"
echo "Shared results root: tabular_data/${RESULTS_ROOT}" | tee -a "${MASTER_LOG}"
echo | tee -a "${MASTER_LOG}"

success_count=0
failure_count=0
failed_datasets=()

for dataset in "${DATASETS[@]}"; do
  echo "---- DATASET: ${dataset} ----" | tee -a "${MASTER_LOG}"
  RUN_GROUP_OVERRIDE="${RUN_GROUP}" \
  RUN_ABLATIONS="${RUN_ABLATIONS}" \
  HP_PROFILE="${HP_PROFILE}" \
  MODELS_OVERRIDE="${MODELS_OVERRIDE}" \
  RANDOM_STATE="${RANDOM_STATE}" \
  MAX_TRAIN_TIME_SECONDS="${MAX_TRAIN_TIME_SECONDS}" \
  ./scripts/run_all_hp_search_minimal_ablation.sh "${dataset}" 2>&1 | tee -a "${MASTER_LOG}"
  exit_code=${PIPESTATUS[0]}

  if [[ ${exit_code} -eq 0 ]]; then
    echo "[OK] ${dataset}" | tee -a "${MASTER_LOG}"
    success_count=$((success_count + 1))
  else
    echo "[FAIL] ${dataset} (exit ${exit_code})" | tee -a "${MASTER_LOG}"
    failed_datasets+=("${dataset}|${exit_code}")
    failure_count=$((failure_count + 1))
  fi
  echo | tee -a "${MASTER_LOG}"
done

echo "====== RUN ALL completed ======" | tee -a "${MASTER_LOG}"
echo "Successful datasets: ${success_count}" | tee -a "${MASTER_LOG}"
echo "Failed datasets: ${failure_count}" | tee -a "${MASTER_LOG}"

if [[ ${failure_count} -gt 0 ]]; then
  echo "Failed dataset summary:" | tee -a "${MASTER_LOG}"
  for item in "${failed_datasets[@]}"; do
    IFS="|" read -r dataset exit_code <<< "${item}"
    echo " - ${dataset}: exit ${exit_code}" | tee -a "${MASTER_LOG}"
  done
  echo "See master log: tabular_data/${MASTER_LOG}" | tee -a "${MASTER_LOG}"
  exit 1
fi

echo "All datasets completed successfully. Master log: tabular_data/${MASTER_LOG}" | tee -a "${MASTER_LOG}"
