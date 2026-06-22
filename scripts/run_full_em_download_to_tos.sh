#!/usr/bin/env bash
set -euo pipefail

ROOT=${DBMIM_ROOT:-/volume/med-train/users/dchen02/code/dbMiM}
HF_ENV=${DBMIM_HF_ENV:-/volume/med-train/users/dchen02/secrets/hf_env_dchen02.sh}
DEFAULT_GROUPS="fafb fib25 kasthuri mitoem mb_moc"
EM_GROUP_LIST=${DBMIM_EM_GROUPS:-$DEFAULT_GROUPS}
MIN_FREE_GB=${DBMIM_MIN_FREE_GB:-200}
LOG_DIR="$ROOT/outputs/full_em_download"

cd "$ROOT"
mkdir -p "$LOG_DIR"

if [[ ! -f "$HF_ENV" ]]; then
  echo "missing HF env file: $HF_ENV" >&2
  exit 2
fi
source "$HF_ENV"

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  echo "HF_TOKEN/HUGGINGFACE_HUB_TOKEN is required" >&2
  exit 2
fi

free_gb() {
  df -BG "$ROOT" | awk 'NR==2 {gsub("G","",$4); print $4}'
}

echo "{\"event\":\"start_full_em_download\",\"root\":\"$ROOT\",\"groups\":\"$EM_GROUP_LIST\",\"time\":\"$(date -Iseconds)\"}"
on_exit() {
  local code=$?
  echo "{\"event\":\"exit_full_em_download\",\"code\":$code,\"time\":\"$(date -Iseconds)\"}"
}
trap on_exit EXIT

for group in $EM_GROUP_LIST; do
  case "$group" in
    fafb|fib25|kasthuri|mitoem|mb_moc) ;;
    *)
      echo "{\"event\":\"invalid_group\",\"group\":\"$group\",\"allowed\":\"$DEFAULT_GROUPS\",\"time\":\"$(date -Iseconds)\"}" >&2
      exit 4
      ;;
  esac

  free=$(free_gb)
  if [[ "$free" -lt "$MIN_FREE_GB" ]]; then
    echo "{\"event\":\"low_disk_stop\",\"group\":\"$group\",\"free_gb\":$free,\"min_free_gb\":$MIN_FREE_GB,\"time\":\"$(date -Iseconds)\"}" >&2
    exit 3
  fi

  marker="$LOG_DIR/${group}.done"
  if [[ -f "$marker" ]]; then
    echo "{\"event\":\"skip_done_group\",\"group\":\"$group\",\"marker\":\"$marker\",\"time\":\"$(date -Iseconds)\"}"
    continue
  fi

  echo "{\"event\":\"start_group\",\"group\":\"$group\",\"free_gb\":$free,\"time\":\"$(date -Iseconds)\"}"
  python scripts/prepare_em_pretrain_data.py --group "$group" --download --extract --upload-tos
  printf '{"group":"%s","done_at":"%s"}\n' "$group" "$(date -Iseconds)" > "$marker"
  echo "{\"event\":\"done_group\",\"group\":\"$group\",\"time\":\"$(date -Iseconds)\"}"
done

python scripts/prepare_em_pretrain_data.py --group all --manifest-only
echo "{\"event\":\"done_full_em_download\",\"time\":\"$(date -Iseconds)\"}"
