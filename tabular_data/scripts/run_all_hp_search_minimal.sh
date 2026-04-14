#!/usr/bin/env bash

# Run minimal hyperparameter searches for all supported models/backends on one dataset,
# without running ablations.
#
# This script delegates to run_all_hp_search_minimal_ablation.sh with RUN_ABLATIONS=0.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCRIPT_NAME="$(basename "$0" .sh)"
RUN_TS="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_GROUP="${RUN_GROUP_OVERRIDE:-${SCRIPT_NAME}/${RUN_TS}}"

DATASET="${1:-${DATASET:-hidden_manifold_10_10}}"

RUN_GROUP_OVERRIDE="${RUN_GROUP}" \
RUN_ABLATIONS=0 \
./scripts/run_all_hp_search_minimal_ablation.sh "${DATASET}"
