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
CHECKPOINT_FILES=${CHECKPOINT_FILES:-"finetuned_best.pt finetuned_latest.pt"}

mkdir -p outputs/watchers
echo "{\"event\":\"watch_start\",\"tos_prefix\":\"${TOS_PREFIX}\",\"checkpoint_files\":\"${CHECKPOINT_FILES}\",\"eval_stage\":\"${EVAL_STAGE}\",\"resource_pool\":\"${RESOURCE_POOL}\",\"gpus_per_pod\":${GPUS_PER_POD}}"

emit_event() {
  printf '%s\n' "$1"
}

probe_checkpoint() {
  local filename="$1"
  local target="${TOS_PREFIX}/${filename}"
  local safe_name="${filename//[^A-Za-z0-9_.-]/_}"
  local probe="/tmp/dbmim_watch_${EVAL_STAGE}_${safe_name}_$$.pt"
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
ready_file=""
for idx in $(seq 1 "${MAX_POLLS}"); do
  for filename in ${CHECKPOINT_FILES}; do
    if probe_checkpoint "${filename}"; then
      ready=1
      ready_file="${filename}"
      emit_event "{\"event\":\"checkpoint_ready\",\"file\":\"${filename}\",\"poll\":${idx}}"
      break
    fi
  done
  if [ "${ready}" -eq 1 ]; then
    break
  fi
  emit_event "{\"event\":\"checkpoint_wait\",\"poll\":${idx}}"
  sleep "${SLEEP_SEC}"
done

if [ "${ready}" -ne 1 ]; then
  emit_event "{\"event\":\"checkpoint_timeout\",\"max_polls\":${MAX_POLLS}}"
  exit 2
fi

emit_event "{\"event\":\"submit_eval_start\",\"stage\":\"${EVAL_STAGE}\"}"
"${SIFLOW_PY}" scripts/submit_siflow_dbmim.py \
  --stage "${EVAL_STAGE}" \
  --resource-pool "${RESOURCE_POOL}" \
  --gpus-per-pod "${GPUS_PER_POD}" \
  --submit
emit_event "{\"event\":\"submit_eval_done\",\"stage\":\"${EVAL_STAGE}\"}"
