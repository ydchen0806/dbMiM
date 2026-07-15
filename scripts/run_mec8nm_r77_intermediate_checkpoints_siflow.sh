#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -d "${ROOT}/external/dbMiM" ]; then
  DBMIM="${ROOT}/external/dbMiM"
else
  DBMIM="${ROOT}"
fi
cd "${ROOT}"

export PYTHONPATH="${DBMIM}:${DBMIM}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

readonly RUN_NAME="eval_mec8nm_r77_r58_intermediate_val_crop"
readonly CONFIG_NAME="finetune_mec8nm_r58_seed1789_balanced_stage1"
readonly TOS_MEC_PREFIX="tos://agi-data/users/dchen02/dbmim/assets/mec8nm_20260711/MEC_8nm"
readonly TOS_OUTPUT_PREFIX="tos://agi-data/users/dchen02/dbmim/mec8nm_runs/${RUN_NAME}"
readonly STEP8_URI="tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r58_seed1789_balanced_stage1/train/finetune_mec8nm_r58_seed1789_balanced_stage1/checkpoint_step_00008000.pt"
readonly STEP10_URI="tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r58_seed1789_balanced_stage1/train/finetune_mec8nm_r58_seed1789_balanced_stage1/checkpoint_step_00010000.pt"
readonly STEP8_SHA256="7fc758d5afc18661241e49d16c4256889be08966eecafa33d23f562456a2c73f"
readonly STEP10_SHA256="2aed3b67f565233fdce696d4341d6178a84524cbb42e181ac98ab833d9b1f130"
readonly DATA_DIR="data/MEC_8nm/hdf5"
readonly SOURCE_DIR="assets/r77_sources"
readonly RUNTIME_CONFIG="configs/runtime_${RUN_NAME}.yaml"
readonly EXPECTED_ROWS="80"
export EXPECTED_ROWS
SAMPLES=(mec8nm_08.h5 mec8nm_10.h5)
STEPS=(8000 10000)
TAGS=(step8k step10k)

mkdir -p "${DATA_DIR}" "${SOURCE_DIR}" "outputs/${RUN_NAME}" configs logs bin
if [ ! -x bin/tosutil ]; then
  echo "missing bundled tosutil" >&2
  exit 2
fi

TOS_CONF=/tmp/mec8nm_r77_tosutil.conf
: > "${TOS_CONF}"
bin/tosutil config \
  -e "${TOS_ENDPOINT:?missing TOS_ENDPOINT}" \
  -re "${TOS_REGION:?missing TOS_REGION}" \
  -i "${TOS_ACCESS_KEY_ID:?missing TOS_ACCESS_KEY_ID}" \
  -k "${TOS_SECRET_ACCESS_KEY:?missing TOS_SECRET_ACCESS_KEY}" \
  -conf="${TOS_CONF}" >/dev/null

echo "[r77] installing offline evaluation dependencies"
python -m pip install --user --no-index --find-links wheelhouse_dbmim pyyaml h5py scipy mahotas connected-components-3d || true
python -m pip install --user --no-index --find-links wheelhouse_superhuman_eval scikit-image tifffile || true
if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("waterz") else 1)
PY
then
  export CPLUS_INCLUDE_PATH="${DBMIM}/third_party/boost_1_84_0:${CPLUS_INCLUDE_PATH:-}"
  export BOOST_INCLUDEDIR="${DBMIM}/third_party/boost_1_84_0"
  python -m pip install --user --no-build-isolation --no-index \
    --find-links wheelhouse_superhuman_eval third_party/waterz
fi
python - <<'PY'
import importlib.util, json, torch
mods = ["torch", "h5py", "yaml", "scipy", "mahotas", "skimage", "waterz", "cc3d"]
probe = {name: bool(importlib.util.find_spec(name)) for name in mods}
print(json.dumps({"module_probe": probe, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()}), flush=True)
if not all(probe.values()) or not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit("R77 requires all evaluation modules and exactly eight visible GPUs")
PY

echo "[r77] downloading validation blocks only: ${SAMPLES[*]}"
for sample in "${SAMPLES[@]}"; do
  bin/tosutil cp "${TOS_MEC_PREFIX}/hdf5/${sample}" "${DATA_DIR}/${sample}" -conf="${TOS_CONF}"
done
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_manifest.json" data/MEC_8nm/mec8nm_manifest.json -conf="${TOS_CONF}" || true
bin/tosutil cp "${TOS_MEC_PREFIX}/mec8nm_splits.json" data/MEC_8nm/mec8nm_splits.json -conf="${TOS_CONF}" || true

echo "[r77] downloading audited R58 intermediate checkpoints"
bin/tosutil cp "${STEP8_URI}" "${SOURCE_DIR}/step8k.pt" -conf="${TOS_CONF}"
bin/tosutil cp "${STEP10_URI}" "${SOURCE_DIR}/step10k.pt" -conf="${TOS_CONF}"
printf '%s  %s\n' "${STEP8_SHA256}" "${SOURCE_DIR}/step8k.pt" | sha256sum -c -
printf '%s  %s\n' "${STEP10_SHA256}" "${SOURCE_DIR}/step10k.pt" | sha256sum -c -

export SOURCE_DIR STEP8_SHA256 STEP10_SHA256
python - <<'PY'
import json, os
from pathlib import Path
import torch

root = Path(os.environ["SOURCE_DIR"])
records = []
keys = None
for step, name, expected_sha in (
    (8000, "step8k.pt", os.environ["STEP8_SHA256"]),
    (10000, "step10k.pt", os.environ["STEP10_SHA256"]),
):
    path = root / name
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = payload.get("model", {})
    cfg = payload.get("config", {})
    if (
        int(payload.get("global_step", -1)) != step
        or int(cfg.get("seed", -1)) != 1789
        or cfg.get("model", {}).get("architecture") != "unetr_aniso_em"
        or len(model) != 214
        or payload.get("ema_decay") not in (None, 0, 0.0)
    ):
        raise SystemExit(f"invalid R58 intermediate checkpoint lineage: {path}")
    current_keys = sorted(model)
    if keys is not None and current_keys != keys:
        raise SystemExit("R58 intermediate checkpoint key sets differ")
    keys = current_keys
    records.append({"step": step, "path": str(path), "sha256": expected_sha, "model_keys": len(model)})
manifest = {"event": "r77_checkpoint_lineage_verified", "records": records}
Path("outputs/eval_mec8nm_r77_r58_intermediate_val_crop/checkpoint_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, sort_keys=True), flush=True)
PY

export RUN_NAME CONFIG_NAME RUNTIME_CONFIG DATA_DIR
python - <<'PY'
import json, os
from pathlib import Path
import yaml

root = Path.cwd()
config_name = os.environ["CONFIG_NAME"]
src = Path("external/dbMiM/configs") / f"{config_name}.yaml"
if not src.exists():
    src = Path("configs") / f"{config_name}.yaml"
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
samples = ["mec8nm_08.h5", "mec8nm_10.h5"]
paths = [str((root / os.environ["DATA_DIR"] / sample).resolve()) for sample in samples]
cfg["data"]["image_paths"] = paths
cfg["data"]["label_paths"] = paths
cfg["output_dir"] = str((root / "outputs" / os.environ["RUN_NAME"]).resolve())
Path(os.environ["RUNTIME_CONFIG"]).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(json.dumps({"runtime_config": os.environ["RUNTIME_CONFIG"], "samples": samples}), flush=True)
PY

for index in "${!STEPS[@]}"; do
  step="${STEPS[$index]}"
  tag="${TAGS[$index]}"
  checkpoint="${SOURCE_DIR}/${tag}.pt"
  output="outputs/${RUN_NAME}/${tag}"
  log="logs/${RUN_NAME}.${tag}.log"
  echo "[r77] evaluating step=${step} checkpoint=${checkpoint}"
  start="$(date +%s)"
  set +e
  python "${DBMIM}/scripts/evaluate_cremi_segmentation.py" \
    --config "${RUNTIME_CONFIG}" \
    --checkpoint "${checkpoint}" \
    --data-dir "${DATA_DIR}" \
    --sample-names "${SAMPLES[@]}" \
    --output-dir "${output}" \
    --crop-size 64 512 512 \
    --stride 16 80 80 \
    --backends waterz \
    --thresholds 0.16 0.18 0.20 0.22 0.24 0.30 0.40 0.50 \
    --calibration-biases 0 0 0 -0.20 -0.45 -0.45 \
    --calibration-temperatures 1.0 1.15 \
    --seed-method maxima_distance \
    --seed-distance 10 \
    --boundary-threshold 0.5 \
    --waterz-scoring hist_quantile \
    --fail-on-backend-error \
    --metric-backend skimage \
    --ignore-label 0 \
    --cremi-boundary-ignore-distance-xy 1 \
    --cremi-boundary-ignore-distance-z 0 \
    --replicate-affinity-boundary \
    --device cuda \
    2>&1 | sed -u "s/^/[r77-${tag}] /" | tee "${log}"
  code="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  echo "{\"event\":\"candidate_eval_done\",\"step\":${step},\"code\":${code},\"elapsed_sec\":$((end-start))}" | tee -a "logs/${RUN_NAME}.status.jsonl"
  if [ "${code}" -ne 0 ]; then
    bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/outputs_failed" -r -conf="${TOS_CONF}" || true
    bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs_failed" -r -conf="${TOS_CONF}" || true
    exit "${code}"
  fi

  METRICS="${output}/cremi_segmentation_metrics.csv" SUMMARY="${output}/cremi_segmentation_summary.json" STEP="${step}" python - <<'PY'
import csv, json, math, os
from pathlib import Path

metrics = Path(os.environ["METRICS"])
summary_path = Path(os.environ["SUMMARY"])
with metrics.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
samples = {row.get("sample") for row in rows}
backends = {row.get("backend") for row in rows}
finite = all(math.isfinite(float(row[field])) for row in rows for field in ("adapted_rand_error", "voi_sum"))
if (
    len(rows) != int(os.environ["EXPECTED_ROWS"])
    or samples != {"mec8nm_08.h5", "mec8nm_10.h5"}
    or backends != {"waterz"}
    or not finite
    or summary.get("failures") != []
):
    raise SystemExit(
        f"invalid R77 step={os.environ['STEP']} rows={len(rows)} samples={samples} "
        f"backends={backends} finite={finite} failures={summary.get('failures')}"
    )
print(json.dumps({"event": "metric_assertion_passed", "step": int(os.environ["STEP"]), "rows": len(rows), "samples": sorted(samples), "backends": sorted(backends)}), flush=True)
PY
done

echo "{\"event\":\"r77_candidates_complete\",\"candidates\":${#STEPS[@]},\"rows_per_candidate\":${EXPECTED_ROWS}}" | tee -a "logs/${RUN_NAME}.status.jsonl"
bin/tosutil cp "outputs/${RUN_NAME}" "${TOS_OUTPUT_PREFIX}/crop_evals" -r -conf="${TOS_CONF}"
bin/tosutil cp logs "${TOS_OUTPUT_PREFIX}/logs" -r -conf="${TOS_CONF}" || true
echo "{\"event\":\"upload_done\",\"tos_output_prefix\":\"${TOS_OUTPUT_PREFIX}\"}" | tee -a "logs/${RUN_NAME}.status.jsonl"
