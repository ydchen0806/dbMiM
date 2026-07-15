#!/usr/bin/env bash
set -euo pipefail

# This wrapper is intentionally validation-only. The generic runner retains its
# explicit test capability for separately approved deployment audits.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

export SAMPLES="mec8nm_08.h5 mec8nm_10.h5"
export BACKENDS="${BACKENDS:-waterz}"
export REGIONS="${REGIONS:-full seam nonseam}"
export FAIL_ON_BACKEND_ERROR="${FAIL_ON_BACKEND_ERROR:-1}"

exec bash scripts/run_mec8nm_blockwise_eval_siflow.sh
