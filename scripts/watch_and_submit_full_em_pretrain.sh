#!/usr/bin/env bash
set -euo pipefail

ROOT=${DBMIM_ROOT:-/volume/med-train/users/dchen02/code/dbMiM}
TOSUTIL=${DBMIM_TOSUTIL:-/volume/med-train/users/dchen02/bin/tosutil}
TOS_CONF=${DBMIM_TOS_CONF:-/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf}
PREFIX=${DBMIM_EM_TOS_PREFIX:-tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data}
DEFAULT_GROUPS="fafb fib25 kasthuri mitoem mb_moc"
REQUIRED_EM_GROUP_LIST=${DBMIM_REQUIRED_EM_GROUPS:-$DEFAULT_GROUPS}
INTERVAL=${DBMIM_WATCH_INTERVAL_SEC:-600}
STAGE=${DBMIM_PRETRAIN_STAGE:-pretrain-em-full-membrane-r20}
RESOURCE_POOL=${DBMIM_RESOURCE_POOL:-med-model}
GPUS=${DBMIM_GPUS_PER_POD:-8}
MARKER="$ROOT/outputs/watchers/${STAGE}.submitted"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$ROOT"
mkdir -p outputs/watchers

has_hdf5_group() {
  local group=$1
  local out
  out=$(timeout 120 "$TOSUTIL" ls "$PREFIX/$group" -s -limit=1000 -conf="$TOS_CONF" 2>&1 || true)
  printf '%s\n' "$out" > "outputs/watchers/${STAGE}.${group}.tos_ls.txt"
  printf '%s\n' "$out" | grep -Eqi '\.(h5|hdf|hdf5)([[:space:]]|$)'
}

echo "{\"event\":\"start_full_em_pretrain_watcher\",\"stage\":\"$STAGE\",\"groups\":\"$REQUIRED_EM_GROUP_LIST\",\"time\":\"$(date -Iseconds)\"}"

while true; do
  if [[ -f "$MARKER" ]]; then
    echo "{\"event\":\"already_submitted\",\"marker\":\"$MARKER\",\"time\":\"$(date -Iseconds)\"}"
    exit 0
  fi

  missing=()
  for group in $REQUIRED_EM_GROUP_LIST; do
    case "$group" in
      fafb|fib25|kasthuri|mitoem|mb_moc) ;;
      *)
        echo "{\"event\":\"invalid_group\",\"group\":\"$group\",\"allowed\":\"$DEFAULT_GROUPS\",\"time\":\"$(date -Iseconds)\"}" >&2
        exit 4
        ;;
    esac
    if has_hdf5_group "$group"; then
      echo "{\"event\":\"group_ready\",\"group\":\"$group\",\"time\":\"$(date -Iseconds)\"}"
    else
      missing+=("$group")
      echo "{\"event\":\"group_wait\",\"group\":\"$group\",\"time\":\"$(date -Iseconds)\"}"
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "{\"event\":\"submit_pretrain\",\"stage\":\"$STAGE\",\"resource_pool\":\"$RESOURCE_POOL\",\"gpus\":$GPUS,\"time\":\"$(date -Iseconds)\"}"
    /volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python \
      scripts/submit_siflow_dbmim.py \
      --stage "$STAGE" \
      --resource-pool "$RESOURCE_POOL" \
      --gpus-per-pod "$GPUS" \
      --submit
    printf '{"stage":"%s","submitted_at":"%s"}\n' "$STAGE" "$(date -Iseconds)" > "$MARKER"
    echo "{\"event\":\"submitted_pretrain\",\"stage\":\"$STAGE\",\"marker\":\"$MARKER\",\"time\":\"$(date -Iseconds)\"}"
    exit 0
  fi

  echo "{\"event\":\"sleep\",\"missing\":\"${missing[*]}\",\"interval_sec\":$INTERVAL,\"time\":\"$(date -Iseconds)\"}"
  sleep "$INTERVAL"
done
