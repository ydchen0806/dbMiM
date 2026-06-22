#!/usr/bin/env bash
set -euo pipefail

ROOT=${DBMIM_ROOT:-/volume/med-train/users/dchen02/code/dbMiM}
TOSUTIL=${DBMIM_TOSUTIL:-/volume/med-train/users/dchen02/bin/tosutil}
TOS_CONF=${DBMIM_TOS_CONF:-/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf}
SIFLOW_PY=${SIFLOW_PY:-/volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python}
PRETRAIN_PREFIX=${DBMIM_R20_PRETRAIN_PREFIX:-tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_full_membrane_dbmim_r20}
CHECKPOINT_FILE=${DBMIM_R20_CHECKPOINT:-pretrained_latest.pt}
LOG_FILE=${DBMIM_R20_LOG_FILE:-train_log.jsonl}
MIN_STEP=${DBMIM_R20_MIN_STEP:-40000}
RESOURCE_POOL=${DBMIM_RESOURCE_POOL:-med-model}
GPUS=${DBMIM_GPUS_PER_POD:-2}
INTERVAL=${DBMIM_WATCH_INTERVAL_SEC:-600}
MAX_POLLS=${DBMIM_MAX_POLLS:-720}
STAGES=${DBMIM_R20_FINETUNE_STAGES:-"finetune-cremi-unetr-aniso-arch-explore-maws-mse-fullem-r20q finetune-cremi-unetr-aniso-arch-explore-maws-mse-bcar-rank-fullem-r20q"}
MARKER=${DBMIM_FINETUNE_SUBMIT_MARKER:-"$ROOT/outputs/watchers/full-em-r20-finetune.submitted"}
PARTIAL_MARKER=${DBMIM_FINETUNE_PARTIAL_MARKER:-"$ROOT/outputs/watchers/full-em-r20-finetune.partial"}

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$ROOT"
mkdir -p outputs/watchers

echo "{\"event\":\"start_full_em_finetune_watcher\",\"pretrain_prefix\":\"$PRETRAIN_PREFIX\",\"checkpoint\":\"$CHECKPOINT_FILE\",\"log_file\":\"$LOG_FILE\",\"min_step\":$MIN_STEP,\"stages\":\"$STAGES\",\"time\":\"$(date -Iseconds)\"}"

probe_checkpoint() {
  local target="$PRETRAIN_PREFIX/$CHECKPOINT_FILE"
  local probe="/tmp/dbmim_r20_finetune_probe_$$.pt"
  rm -f "$probe"
  if timeout 120 "$TOSUTIL" cp "$target" "$probe" -conf="$TOS_CONF" >/dev/null 2>&1; then
    if [[ -s "$probe" ]]; then
      rm -f "$probe"
      return 0
    fi
  fi
  rm -f "$probe"
  return 1
}

probe_min_step() {
  local target="$PRETRAIN_PREFIX/$LOG_FILE"
  local probe="/tmp/dbmim_r20_finetune_log_probe_$$.jsonl"
  rm -f "$probe"
  if ! timeout 120 "$TOSUTIL" cp "$target" "$probe" -conf="$TOS_CONF" >/dev/null 2>&1; then
    rm -f "$probe"
    return 1
  fi
  set +e
  python - "$probe" "$MIN_STEP" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_step = int(sys.argv[2])
max_step = 0
for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    max_step = max(max_step, int(row.get("step", 0)))
print(max_step)
raise SystemExit(0 if max_step >= min_step else 1)
PY
  local status=$?
  set -e
  rm -f "$probe"
  return "$status"
}

if [[ -f "$MARKER" ]]; then
  echo "{\"event\":\"already_submitted\",\"marker\":\"$MARKER\",\"time\":\"$(date -Iseconds)\"}"
  exit 0
fi

for idx in $(seq 1 "$MAX_POLLS"); do
  if probe_checkpoint && probe_min_step; then
    echo "{\"event\":\"checkpoint_ready\",\"poll\":$idx,\"min_step\":$MIN_STEP,\"time\":\"$(date -Iseconds)\"}"
    : > "$PARTIAL_MARKER"
    for stage in $STAGES; do
      echo "{\"event\":\"submit_finetune_start\",\"stage\":\"$stage\",\"resource_pool\":\"$RESOURCE_POOL\",\"gpus\":$GPUS,\"time\":\"$(date -Iseconds)\"}"
      "$SIFLOW_PY" scripts/submit_siflow_dbmim.py \
        --stage "$stage" \
        --resource-pool "$RESOURCE_POOL" \
        --gpus-per-pod "$GPUS" \
        --post-train-official-abc-eval \
        --submit
      printf '%s\t%s\n' "$(date -Iseconds)" "$stage" >> "$PARTIAL_MARKER"
      echo "{\"event\":\"submit_finetune_done\",\"stage\":\"$stage\",\"time\":\"$(date -Iseconds)\"}"
    done
    {
      printf '{"checkpoint":"%s/%s","submitted_at":"%s","stages":[' "$PRETRAIN_PREFIX" "$CHECKPOINT_FILE" "$(date -Iseconds)"
      first=1
      for stage in $STAGES; do
        if [[ "$first" -eq 0 ]]; then
          printf ','
        fi
        printf '"%s"' "$stage"
        first=0
      done
      printf ']}\n'
    } > "$MARKER"
    exit 0
  fi
  echo "{\"event\":\"checkpoint_wait\",\"poll\":$idx,\"time\":\"$(date -Iseconds)\"}"
  sleep "$INTERVAL"
done

echo "{\"event\":\"checkpoint_timeout\",\"max_polls\":$MAX_POLLS,\"time\":\"$(date -Iseconds)\"}"
exit 2
