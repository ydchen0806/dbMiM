#!/usr/bin/env bash
set -euo pipefail

cd /volume/med-train/users/dchen02/code/dbMiM
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

TOSUTIL=${TOSUTIL:-/volume/med-train/users/dchen02/bin/tosutil}
TOS_CONF=${TOS_CONF:-/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf}
TOS_PREFIX=${TOS_PREFIX:-tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_dbmim}
SIFLOW_PY=${SIFLOW_PY:-/volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python}
RESOURCE_POOL=${RESOURCE_POOL:-auto}
GPUS_PER_POD=${GPUS_PER_POD:-1}
EVAL_STAGE=${EVAL_STAGE:-eval-cremi}
SLEEP_SEC=${SLEEP_SEC:-120}
MAX_POLLS=${MAX_POLLS:-240}

mkdir -p outputs/watchers
echo "{\"event\":\"watch_start\",\"tos_prefix\":\"${TOS_PREFIX}\",\"eval_stage\":\"${EVAL_STAGE}\",\"resource_pool\":\"${RESOURCE_POOL}\",\"gpus_per_pod\":${GPUS_PER_POD}}"

probe_checkpoint() {
  local kind="$1"
  local target="${TOS_PREFIX}/finetuned_${kind}.pt"
  local probe="/tmp/dbmim_watch_${EVAL_STAGE}_${kind}_$$.pt"
  rm -f "${probe}"
  if timeout 60 "${TOSUTIL}" cp "${target}" "${probe}" -conf="${TOS_CONF}" >/dev/null 2>&1; then
    local ok=1
    if [ -s "${probe}" ]; then
      ok=0
    fi
    rm -f "${probe}"
    return "${ok}"
  fi
  rm -f "${probe}"
  return 1
}

ready=0
for idx in $(seq 1 "${MAX_POLLS}"); do
  if probe_checkpoint best; then
    ready=1
    echo "{\"event\":\"checkpoint_ready\",\"kind\":\"best\",\"poll\":${idx}}"
    break
  fi
  if probe_checkpoint latest; then
    ready=1
    echo "{\"event\":\"checkpoint_ready\",\"kind\":\"latest\",\"poll\":${idx}}"
    break
  fi
  echo "{\"event\":\"checkpoint_wait\",\"poll\":${idx}}"
  sleep "${SLEEP_SEC}"
done

if [ "${ready}" -ne 1 ]; then
  echo "{\"event\":\"checkpoint_timeout\",\"max_polls\":${MAX_POLLS}}" >&2
  exit 2
fi

"${SIFLOW_PY}" scripts/submit_siflow_dbmim.py \
  --stage "${EVAL_STAGE}" \
  --resource-pool "${RESOURCE_POOL}" \
  --gpus-per-pod "${GPUS_PER_POD}" \
  --submit
