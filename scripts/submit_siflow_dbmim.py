#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml


PROJECT = Path("/volume/med-train/users/dchen02/code/dbMiM")
HELPER = Path("/volume/med-train/users/dchen02/.codex/skills/yinda-public-skill/scripts/submit_tos_bootstrap_job.py")
PY = Path("/volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python")
WHEELHOUSE = PROJECT / "outputs" / "wheelhouse_dbmim"
TOSUTIL = Path("/volume/med-train/users/dchen02/bin/tosutil")
TOS_OUTPUT_PREFIX = "tos://agi-data/users/dchen02/dbmim/outputs"
CREMI_ASSET = "tos://agi-data/users/dchen02/dbmim/assets/cremi_abc_20160501.tar.gz"


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _write_yaml(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _patch_cremi_configs(bundle: Path) -> None:
    pretrain = bundle / "configs" / "pretrain_cremi_real.yaml"
    finetune = bundle / "configs" / "finetune_cremi_real.yaml"
    pre_cfg = yaml.safe_load(pretrain.read_text(encoding="utf-8"))
    pre_cfg["output_dir"] = "outputs/pretrain_cremi_real_dbmim"
    pre_cfg["data"]["train_paths"] = ["data/CREMI"]
    pre_cfg["train"]["epochs"] = max(int(pre_cfg["train"].get("epochs", 1)), 100000)
    pre_cfg["train"]["save_every"] = max(int(pre_cfg["train"].get("save_every", 1)), 50)
    _write_yaml(pretrain, pre_cfg)

    ft_cfg = yaml.safe_load(finetune.read_text(encoding="utf-8"))
    ft_cfg["output_dir"] = "outputs/finetune_cremi_real_dbmim"
    ft_cfg["pretrained"] = "outputs/pretrain_cremi_real_dbmim/pretrained_latest.pt"
    ft_cfg["data"]["image_paths"] = ["data/CREMI"]
    ft_cfg["data"]["label_paths"] = ["data/CREMI"]
    ft_cfg["train"]["epochs"] = max(int(ft_cfg["train"].get("epochs", 1)), 100000)
    ft_cfg["train"]["eval_every"] = max(int(ft_cfg["train"].get("eval_every", 1)), 20)
    ft_cfg["train"]["save_every"] = max(int(ft_cfg["train"].get("save_every", 1)), 20)
    _write_yaml(finetune, ft_cfg)


def make_bundle(entrypoint: str, stage: str) -> Path:
    out = PROJECT / "outputs" / "siflow_bundles" / f"dbmim_bundle_{stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    for name in [
        "dbmim",
        "configs",
        "train_pretrain.py",
        "train_finetune.py",
        "scripts/evaluate_cremi_segmentation.py",
        "requirements-dbMIM.txt",
    ]:
        src = PROJECT / name
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    if WHEELHOUSE.exists():
        shutil.copytree(WHEELHOUSE, out / "wheelhouse")
    if stage in {"pretrain-cremi", "finetune-cremi", "eval-cremi", "eval-cremi-sweep", "eval-cremi-gpu-probe"}:
        _patch_cremi_configs(out)
    if TOSUTIL.exists():
        (out / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(TOSUTIL.resolve(), out / "bin" / "tosutil")
        os.chmod(out / "bin" / "tosutil", 0o700)

    prelude = []
    postlude = []
    if stage in {"pretrain-cremi", "finetune-cremi", "eval-cremi", "eval-cremi-sweep", "eval-cremi-gpu-probe"}:
        prelude.extend(
            [
                "if [ -x bin/tosutil ]; then",
                "  TOS_CONF=/tmp/dbmim_tosutil.conf",
                "  : > \"$TOS_CONF\"",
                "  bin/tosutil config -e \"$TOS_ENDPOINT\" -re \"$TOS_REGION\" -i \"$TOS_ACCESS_KEY_ID\" -k \"$TOS_SECRET_ACCESS_KEY\" -conf=\"$TOS_CONF\" >/dev/null",
                "fi",
                "mkdir -p data",
                f"bin/tosutil cp {CREMI_ASSET} /tmp/cremi_abc_20160501.tar.gz -conf=\"$TOS_CONF\"",
                "tar -xzf /tmp/cremi_abc_20160501.tar.gz -C data",
                "ls -lh data/CREMI",
            ]
        )
    if stage == "finetune-cremi":
        prelude.extend(
            [
                "mkdir -p outputs/pretrain_cremi_real_dbmim",
                "bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/pretrain_cremi_real_dbmim/pretrained_latest.pt "
                "outputs/pretrain_cremi_real_dbmim/pretrained_latest.pt -conf=\"$TOS_CONF\"",
            ]
        )
    if stage in {"eval-cremi", "eval-cremi-sweep", "eval-cremi-gpu-probe"}:
        prelude.extend(
            [
                "mkdir -p outputs/finetune_cremi_real_dbmim outputs/eval_cremi_real_dbmim outputs/eval_cremi_postprocess_sweep outputs/eval_cremi_gpu_probe",
                "if bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/finetune_cremi_real_dbmim/finetuned_best.pt "
                "outputs/finetune_cremi_real_dbmim/finetuned_best.pt -conf=\"$TOS_CONF\"; then",
                "  export DBMIM_EVAL_CKPT=outputs/finetune_cremi_real_dbmim/finetuned_best.pt",
                "else",
                "  bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/finetune_cremi_real_dbmim/finetuned_latest.pt "
                "outputs/finetune_cremi_real_dbmim/finetuned_latest.pt -conf=\"$TOS_CONF\"",
                "  export DBMIM_EVAL_CKPT=outputs/finetune_cremi_real_dbmim/finetuned_latest.pt",
                "fi",
            ]
        )
    if stage == "pretrain-cremi":
        postlude.extend(
            [
                "bin/tosutil cp outputs/pretrain_cremi_real_dbmim "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "finetune-cremi":
        postlude.extend(
            [
                "bin/tosutil cp outputs/finetune_cremi_real_dbmim "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "eval-cremi":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_real_dbmim "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "eval-cremi-sweep":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_postprocess_sweep "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "eval-cremi-gpu-probe":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_gpu_probe "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )

    (out / "run.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd \"$(dirname \"$0\")\"",
                "if [ -d wheelhouse ]; then",
                "  missing_pkgs=$(python - <<'PY'",
                "import importlib.util",
                "mapping = {'yaml': 'PyYAML', 'h5py': 'h5py', 'PIL': 'Pillow', 'numpy': 'numpy', 'scipy': 'scipy', 'mahotas': 'mahotas', 'cc3d': 'connected-components-3d'}",
                "print(' '.join(pkg for mod, pkg in mapping.items() if importlib.util.find_spec(mod) is None))",
                "PY",
                "  )",
                "  if [ -n \"$missing_pkgs\" ]; then",
                "    python -m pip install --user --no-index --find-links wheelhouse $missing_pkgs",
                "  fi",
                "fi",
                "python - <<'PY'",
                "import importlib.util",
                "missing=[m for m in ['torch','yaml','h5py','PIL','numpy','scipy','mahotas','cc3d'] if importlib.util.find_spec(m) is None]",
                "print({'missing_python_modules': missing})",
                "if missing:",
                "    raise SystemExit('missing required python modules: '+','.join(missing))",
                "PY",
                *prelude,
                entrypoint,
                *postlude,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit dbMiM training to SiFlow through TOS bootstrap")
    parser.add_argument("--stage", choices=["pretrain", "finetune", "pretrain-cremi", "finetune-cremi", "eval-cremi", "eval-cremi-sweep", "eval-cremi-gpu-probe", "smoke"], default="smoke")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resource-pool", default="med-model")
    parser.add_argument("--gpus-per-pod", type=int, default=8)
    args = parser.parse_args()

    nproc = int(args.gpus_per_pod)
    if args.stage == "pretrain":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_pretrain.py --config configs/pretrain_fafb.yaml"
        prefix = "dbmim-pretrain"
    elif args.stage == "pretrain-cremi":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_pretrain.py --config configs/pretrain_cremi_real.yaml"
        prefix = "dbmim-pretrain-cremi"
    elif args.stage == "finetune-cremi":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi_real.yaml"
        prefix = "dbmim-finetune-cremi"
    elif args.stage == "eval-cremi":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_real_dbmim "
            "--crop-size 32 256 256 "
            "--stride 16 128 128 "
            "--thresholds 0.35 0.45 0.55 0.65 "
            "--min-size 32 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi"
    elif args.stage == "eval-cremi-sweep":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_postprocess_sweep "
            "--crop-size 32 256 256 "
            "--stride 16 128 128 "
            "--thresholds 0.25 0.35 0.45 0.55 0.65 0.75 0.85 "
            "--backends graph_cc cc3d_mean scipy_watershed scipy_agglomeration mahotas_watershed mahotas_agglomeration waterz "
            "--min-size 32 "
            "--seed-method maxima_distance "
            "--seed-distance 12 "
            "--boundary-threshold 0.5 "
            "--min-boundary 4 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-sweep"
    elif args.stage == "eval-cremi-gpu-probe":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_gpu_probe "
            "--crop-size 32 256 256 "
            "--stride 16 128 128 "
            "--thresholds 0.65 0.85 "
            "--backends graph_cc cc3d_mean cupy_mean cupy_graph_cc "
            "--min-size 32 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-gpu-probe"
    elif args.stage == "finetune":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi.yaml"
        prefix = "dbmim-finetune"
    else:
        entrypoint = "python train_pretrain.py --config configs/pretrain_smoke.yaml && python train_finetune.py --config configs/finetune_smoke.yaml"
        prefix = "dbmim-smoke"

    bundle = make_bundle(entrypoint, args.stage)
    cmd = [
        str(PY),
        str(HELPER),
        "--name-prefix",
        prefix,
        "--bundle-root",
        str(bundle),
        "--entrypoint",
        "bash run.sh",
        "--resource-pool",
        args.resource_pool,
        "--gpus-per-pod",
        str(args.gpus_per_pod),
        "--tos-prefix",
        "tos://agi-data/users/dchen02/dbmim/bundles",
        "--direct-network",
    ]
    if args.submit:
        cmd.append("--submit")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
