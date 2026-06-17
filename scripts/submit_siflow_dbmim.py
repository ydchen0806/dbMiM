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
CREMI_STAGES = {
    "pretrain-cremi",
    "pretrain-cremi-long",
    "finetune-cremi",
    "finetune-cremi-unetr-pretrained",
    "finetune-cremi-unetr-scratch",
    "finetune-cremi-unetr-aniso-pretrained",
    "finetune-cremi-unetr-aniso-scratch",
    "finetune-cremi-unetr-aniso-longpretrained",
    "finetune-cremi-zdice",
    "finetune-cremi-zdice-focal",
    "eval-cremi",
    "eval-cremi-unetr-pretrained",
    "eval-cremi-unetr-scratch",
    "eval-cremi-unetr-aniso-pretrained",
    "eval-cremi-unetr-aniso-scratch",
    "eval-cremi-unetr-aniso-longpretrained",
    "eval-cremi-unetr-aniso-large-pretrained",
    "eval-cremi-unetr-aniso-large-scratch",
    "eval-cremi-unetr-aniso-large-longpretrained",
    "eval-cremi-sweep",
    "eval-cremi-gpu-probe",
    "eval-cremi-rag-ablation",
    "eval-cremi-aniso-graph",
    "eval-cremi-scale64",
    "eval-cremi-zdice",
    "eval-cremi-zdice-focal",
}
CREMI_EVAL_STAGES = {
    "eval-cremi",
    "eval-cremi-unetr-pretrained",
    "eval-cremi-unetr-scratch",
    "eval-cremi-unetr-aniso-pretrained",
    "eval-cremi-unetr-aniso-scratch",
    "eval-cremi-unetr-aniso-longpretrained",
    "eval-cremi-unetr-aniso-large-pretrained",
    "eval-cremi-unetr-aniso-large-scratch",
    "eval-cremi-unetr-aniso-large-longpretrained",
    "eval-cremi-sweep",
    "eval-cremi-gpu-probe",
    "eval-cremi-rag-ablation",
    "eval-cremi-aniso-graph",
    "eval-cremi-scale64",
    "eval-cremi-zdice",
    "eval-cremi-zdice-focal",
}


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"


def _write_yaml(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _patch_cremi_configs(bundle: Path) -> None:
    pretrain = bundle / "configs" / "pretrain_cremi_real.yaml"
    pre_cfg = yaml.safe_load(pretrain.read_text(encoding="utf-8"))
    pre_cfg["output_dir"] = "outputs/pretrain_cremi_real_dbmim"
    pre_cfg["data"]["train_paths"] = ["data/CREMI"]
    pre_cfg["train"]["epochs"] = max(int(pre_cfg["train"].get("epochs", 1)), 100000)
    pre_cfg["train"]["save_every"] = max(int(pre_cfg["train"].get("save_every", 1)), 50)
    _write_yaml(pretrain, pre_cfg)

    pretrain_long = bundle / "configs" / "pretrain_cremi_real_long.yaml"
    if pretrain_long.exists():
        pre_long_cfg = yaml.safe_load(pretrain_long.read_text(encoding="utf-8"))
        pre_long_cfg["output_dir"] = "outputs/pretrain_cremi_real_long_dbmim"
        pre_long_cfg["data"]["train_paths"] = ["data/CREMI"]
        pre_long_cfg["train"]["epochs"] = max(int(pre_long_cfg["train"].get("epochs", 1)), 100000)
        pre_long_cfg["train"]["save_every"] = max(int(pre_long_cfg["train"].get("save_every", 1)), 10)
        _write_yaml(pretrain_long, pre_long_cfg)

    for name, out_dir in [
        ("finetune_cremi_real.yaml", "outputs/finetune_cremi_real_dbmim"),
        ("finetune_cremi_real_unetr_pretrained.yaml", "outputs/finetune_cremi_real_unetr_pretrained"),
        ("finetune_cremi_real_unetr_scratch.yaml", "outputs/finetune_cremi_real_unetr_scratch"),
        ("finetune_cremi_real_unetr_aniso_pretrained.yaml", "outputs/finetune_cremi_real_unetr_aniso_pretrained"),
        ("finetune_cremi_real_unetr_aniso_scratch.yaml", "outputs/finetune_cremi_real_unetr_aniso_scratch"),
        (
            "finetune_cremi_real_unetr_aniso_longpretrained.yaml",
            "outputs/finetune_cremi_real_unetr_aniso_longpretrained",
        ),
        ("finetune_cremi_real_zdice.yaml", "outputs/finetune_cremi_real_zdice"),
        ("finetune_cremi_real_zdice_focal.yaml", "outputs/finetune_cremi_real_zdice_focal"),
    ]:
        finetune = bundle / "configs" / name
        if not finetune.exists():
            continue
        ft_cfg = yaml.safe_load(finetune.read_text(encoding="utf-8"))
        ft_cfg["output_dir"] = out_dir
        if "scratch" in name:
            ft_cfg["pretrained"] = ""
        elif "longpretrained" in name:
            ft_cfg["pretrained"] = "outputs/pretrain_cremi_real_long_dbmim/pretrained_latest.pt"
        else:
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
    if stage in CREMI_STAGES:
        _patch_cremi_configs(out)
    if TOSUTIL.exists():
        (out / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(TOSUTIL.resolve(), out / "bin" / "tosutil")
        os.chmod(out / "bin" / "tosutil", 0o700)

    prelude = []
    postlude = []
    if stage in CREMI_STAGES:
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
    if stage in {
        "finetune-cremi",
        "finetune-cremi-unetr-pretrained",
        "finetune-cremi-unetr-aniso-pretrained",
        "finetune-cremi-zdice",
        "finetune-cremi-zdice-focal",
    }:
        prelude.extend(
            [
                "mkdir -p outputs/pretrain_cremi_real_dbmim",
                "bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/pretrain_cremi_real_dbmim/pretrained_latest.pt "
                "outputs/pretrain_cremi_real_dbmim/pretrained_latest.pt -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "finetune-cremi-unetr-aniso-longpretrained":
        prelude.extend(
            [
                "mkdir -p outputs/pretrain_cremi_real_long_dbmim",
                "bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/pretrain_cremi_real_long_dbmim/pretrained_latest.pt "
                "outputs/pretrain_cremi_real_long_dbmim/pretrained_latest.pt -conf=\"$TOS_CONF\"",
            ]
        )
    if stage in CREMI_EVAL_STAGES:
        prelude.extend(
            [
                "mkdir -p outputs/finetune_cremi_real_dbmim outputs/finetune_cremi_real_unetr_pretrained outputs/finetune_cremi_real_unetr_scratch outputs/finetune_cremi_real_unetr_aniso_pretrained outputs/finetune_cremi_real_unetr_aniso_scratch outputs/finetune_cremi_real_unetr_aniso_longpretrained outputs/finetune_cremi_real_zdice outputs/finetune_cremi_real_zdice_focal outputs/eval_cremi_real_dbmim outputs/eval_cremi_unetr_pretrained outputs/eval_cremi_unetr_scratch outputs/eval_cremi_unetr_aniso_pretrained outputs/eval_cremi_unetr_aniso_scratch outputs/eval_cremi_unetr_aniso_longpretrained outputs/eval_cremi_unetr_aniso_large_pretrained outputs/eval_cremi_unetr_aniso_large_scratch outputs/eval_cremi_unetr_aniso_large_longpretrained outputs/eval_cremi_postprocess_sweep outputs/eval_cremi_gpu_probe outputs/eval_cremi_rag_ablation outputs/eval_cremi_aniso_graph outputs/eval_cremi_scale64 outputs/eval_cremi_zdice outputs/eval_cremi_zdice_focal",
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
    eval_stage_map = {
        "eval-cremi-unetr-pretrained": ("finetune_cremi_real_unetr_pretrained", "DBMIM_EVAL_CKPT"),
        "eval-cremi-unetr-scratch": ("finetune_cremi_real_unetr_scratch", "DBMIM_EVAL_CKPT"),
        "eval-cremi-unetr-aniso-pretrained": ("finetune_cremi_real_unetr_aniso_pretrained", "DBMIM_EVAL_CKPT"),
        "eval-cremi-unetr-aniso-scratch": ("finetune_cremi_real_unetr_aniso_scratch", "DBMIM_EVAL_CKPT"),
        "eval-cremi-unetr-aniso-longpretrained": (
            "finetune_cremi_real_unetr_aniso_longpretrained",
            "DBMIM_EVAL_CKPT",
        ),
        "eval-cremi-unetr-aniso-large-pretrained": (
            "finetune_cremi_real_unetr_aniso_pretrained",
            "DBMIM_EVAL_CKPT",
        ),
        "eval-cremi-unetr-aniso-large-scratch": ("finetune_cremi_real_unetr_aniso_scratch", "DBMIM_EVAL_CKPT"),
        "eval-cremi-unetr-aniso-large-longpretrained": (
            "finetune_cremi_real_unetr_aniso_longpretrained",
            "DBMIM_EVAL_CKPT",
        ),
    }
    if stage in eval_stage_map:
        model_prefix, env_key = eval_stage_map[stage]
        prelude.extend(
            [
                "if bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/{model_prefix}/finetuned_best.pt "
                f"outputs/{model_prefix}/finetuned_best.pt -conf=\"$TOS_CONF\"; then",
                f"  export {env_key}=outputs/{model_prefix}/finetuned_best.pt",
                "else",
                "  bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/{model_prefix}/finetuned_latest.pt "
                f"outputs/{model_prefix}/finetuned_latest.pt -conf=\"$TOS_CONF\"",
                f"  export {env_key}=outputs/{model_prefix}/finetuned_latest.pt",
                "fi",
            ]
        )
    if stage == "eval-cremi-zdice":
        prelude.extend(
            [
                "if bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/finetune_cremi_real_zdice/finetuned_best.pt "
                "outputs/finetune_cremi_real_zdice/finetuned_best.pt -conf=\"$TOS_CONF\"; then",
                "  export DBMIM_EVAL_CKPT=outputs/finetune_cremi_real_zdice/finetuned_best.pt",
                "else",
                "  bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/finetune_cremi_real_zdice/finetuned_latest.pt "
                "outputs/finetune_cremi_real_zdice/finetuned_latest.pt -conf=\"$TOS_CONF\"",
                "  export DBMIM_EVAL_CKPT=outputs/finetune_cremi_real_zdice/finetuned_latest.pt",
                "fi",
            ]
        )
    if stage == "eval-cremi-zdice-focal":
        prelude.extend(
            [
                "if bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/finetune_cremi_real_zdice_focal/finetuned_best.pt "
                "outputs/finetune_cremi_real_zdice_focal/finetuned_best.pt -conf=\"$TOS_CONF\"; then",
                "  export DBMIM_EVAL_CKPT=outputs/finetune_cremi_real_zdice_focal/finetuned_best.pt",
                "else",
                "  bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/finetune_cremi_real_zdice_focal/finetuned_latest.pt "
                "outputs/finetune_cremi_real_zdice_focal/finetuned_latest.pt -conf=\"$TOS_CONF\"",
                "  export DBMIM_EVAL_CKPT=outputs/finetune_cremi_real_zdice_focal/finetuned_latest.pt",
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
    if stage == "pretrain-cremi-long":
        postlude.extend(
            [
                "bin/tosutil cp outputs/pretrain_cremi_real_long_dbmim "
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
    if stage == "finetune-cremi-unetr-pretrained":
        postlude.extend(
            [
                "bin/tosutil cp outputs/finetune_cremi_real_unetr_pretrained "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "finetune-cremi-unetr-scratch":
        postlude.extend(
            [
                "bin/tosutil cp outputs/finetune_cremi_real_unetr_scratch "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "finetune-cremi-unetr-aniso-pretrained":
        postlude.extend(
            [
                "bin/tosutil cp outputs/finetune_cremi_real_unetr_aniso_pretrained "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "finetune-cremi-unetr-aniso-scratch":
        postlude.extend(
            [
                "bin/tosutil cp outputs/finetune_cremi_real_unetr_aniso_scratch "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "finetune-cremi-unetr-aniso-longpretrained":
        postlude.extend(
            [
                "bin/tosutil cp outputs/finetune_cremi_real_unetr_aniso_longpretrained "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "finetune-cremi-zdice":
        postlude.extend(
            [
                "bin/tosutil cp outputs/finetune_cremi_real_zdice "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "finetune-cremi-zdice-focal":
        postlude.extend(
            [
                "bin/tosutil cp outputs/finetune_cremi_real_zdice_focal "
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
    if stage == "eval-cremi-unetr-pretrained":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_unetr_pretrained "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "eval-cremi-unetr-scratch":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_unetr_scratch "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    eval_output_dirs = {
        "eval-cremi-unetr-aniso-pretrained": "outputs/eval_cremi_unetr_aniso_pretrained",
        "eval-cremi-unetr-aniso-scratch": "outputs/eval_cremi_unetr_aniso_scratch",
        "eval-cremi-unetr-aniso-longpretrained": "outputs/eval_cremi_unetr_aniso_longpretrained",
        "eval-cremi-unetr-aniso-large-pretrained": "outputs/eval_cremi_unetr_aniso_large_pretrained",
        "eval-cremi-unetr-aniso-large-scratch": "outputs/eval_cremi_unetr_aniso_large_scratch",
        "eval-cremi-unetr-aniso-large-longpretrained": "outputs/eval_cremi_unetr_aniso_large_longpretrained",
    }
    if stage in eval_output_dirs:
        postlude.extend(
            [
                f"bin/tosutil cp {eval_output_dirs[stage]} "
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
    if stage == "eval-cremi-rag-ablation":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_rag_ablation "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "eval-cremi-aniso-graph":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_aniso_graph "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "eval-cremi-scale64":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_scale64 "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "eval-cremi-zdice":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_zdice "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if stage == "eval-cremi-zdice-focal":
        postlude.extend(
            [
                "bin/tosutil cp outputs/eval_cremi_zdice_focal "
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
    parser.add_argument(
        "--stage",
        choices=[
            "pretrain",
            "finetune",
            "pretrain-cremi",
            "pretrain-cremi-long",
            "finetune-cremi",
            "finetune-cremi-unetr-pretrained",
            "finetune-cremi-unetr-scratch",
            "finetune-cremi-unetr-aniso-pretrained",
            "finetune-cremi-unetr-aniso-scratch",
            "finetune-cremi-unetr-aniso-longpretrained",
            "finetune-cremi-zdice",
            "finetune-cremi-zdice-focal",
            "eval-cremi",
            "eval-cremi-unetr-pretrained",
            "eval-cremi-unetr-scratch",
            "eval-cremi-unetr-aniso-pretrained",
            "eval-cremi-unetr-aniso-scratch",
            "eval-cremi-unetr-aniso-longpretrained",
            "eval-cremi-unetr-aniso-large-pretrained",
            "eval-cremi-unetr-aniso-large-scratch",
            "eval-cremi-unetr-aniso-large-longpretrained",
            "eval-cremi-sweep",
            "eval-cremi-gpu-probe",
            "eval-cremi-rag-ablation",
            "eval-cremi-aniso-graph",
            "eval-cremi-scale64",
            "eval-cremi-zdice",
            "eval-cremi-zdice-focal",
            "smoke",
        ],
        default="smoke",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resource-pool", default="auto")
    parser.add_argument("--gpus-per-pod", type=int, default=8)
    args = parser.parse_args()

    nproc = int(args.gpus_per_pod)
    if args.stage == "pretrain":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_pretrain.py --config configs/pretrain_fafb.yaml"
        prefix = "dbmim-pretrain"
    elif args.stage == "pretrain-cremi":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_pretrain.py --config configs/pretrain_cremi_real.yaml"
        prefix = "dbmim-pretrain-cremi"
    elif args.stage == "pretrain-cremi-long":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_pretrain.py --config configs/pretrain_cremi_real_long.yaml"
        prefix = "dbmim-pretrain-cremi-long"
    elif args.stage == "finetune-cremi":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi_real.yaml"
        prefix = "dbmim-finetune-cremi"
    elif args.stage == "finetune-cremi-unetr-pretrained":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi_real_unetr_pretrained.yaml"
        prefix = "dbmim-finetune-cremi-unetr-pretrained"
    elif args.stage == "finetune-cremi-unetr-scratch":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi_real_unetr_scratch.yaml"
        prefix = "dbmim-finetune-cremi-unetr-scratch"
    elif args.stage == "finetune-cremi-unetr-aniso-pretrained":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi_real_unetr_aniso_pretrained.yaml"
        prefix = "dbmim-finetune-cremi-unetr-aniso-pretrained"
    elif args.stage == "finetune-cremi-unetr-aniso-scratch":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi_real_unetr_aniso_scratch.yaml"
        prefix = "dbmim-finetune-cremi-unetr-aniso-scratch"
    elif args.stage == "finetune-cremi-unetr-aniso-longpretrained":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi_real_unetr_aniso_longpretrained.yaml"
        prefix = "dbmim-finetune-cremi-unetr-aniso-longpretrained"
    elif args.stage == "finetune-cremi-zdice":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi_real_zdice.yaml"
        prefix = "dbmim-finetune-cremi-zdice"
    elif args.stage == "finetune-cremi-zdice-focal":
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/finetune_cremi_real_zdice_focal.yaml"
        prefix = "dbmim-finetune-cremi-zdice-focal"
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
    elif args.stage == "eval-cremi-unetr-pretrained":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_unetr_pretrained.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_unetr_pretrained "
            "--crop-size 32 256 256 "
            "--stride 16 128 128 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.65 0.75 0.85 0.90 "
            "--xy-thresholds 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-unetr-pretrained"
    elif args.stage == "eval-cremi-unetr-scratch":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_unetr_scratch.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_unetr_scratch "
            "--crop-size 32 256 256 "
            "--stride 16 128 128 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.65 0.75 0.85 0.90 "
            "--xy-thresholds 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-unetr-scratch"
    elif args.stage == "eval-cremi-unetr-aniso-pretrained":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_unetr_aniso_pretrained.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_unetr_aniso_pretrained "
            "--crop-size 32 320 320 "
            "--stride 16 80 80 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.45 0.55 0.65 0.75 0.85 "
            "--xy-thresholds 0.65 0.75 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-unetr-aniso-pretrained"
    elif args.stage == "eval-cremi-unetr-aniso-scratch":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_unetr_aniso_scratch.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_unetr_aniso_scratch "
            "--crop-size 32 320 320 "
            "--stride 16 80 80 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.45 0.55 0.65 0.75 0.85 "
            "--xy-thresholds 0.65 0.75 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-unetr-aniso-scratch"
    elif args.stage == "eval-cremi-unetr-aniso-longpretrained":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_unetr_aniso_longpretrained.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_unetr_aniso_longpretrained "
            "--crop-size 32 320 320 "
            "--stride 16 80 80 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.45 0.55 0.65 0.75 0.85 "
            "--xy-thresholds 0.65 0.75 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-unetr-aniso-longpretrained"
    elif args.stage == "eval-cremi-unetr-aniso-large-pretrained":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_unetr_aniso_pretrained.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_unetr_aniso_large_pretrained "
            "--crop-size 64 512 512 "
            "--stride 16 80 80 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.55 0.65 0.75 "
            "--xy-thresholds 0.75 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-unetr-aniso-large-pretrained"
    elif args.stage == "eval-cremi-unetr-aniso-large-scratch":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_unetr_aniso_scratch.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_unetr_aniso_large_scratch "
            "--crop-size 64 512 512 "
            "--stride 16 80 80 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.55 0.65 0.75 "
            "--xy-thresholds 0.75 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-unetr-aniso-large-scratch"
    elif args.stage == "eval-cremi-unetr-aniso-large-longpretrained":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_unetr_aniso_longpretrained.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_unetr_aniso_large_longpretrained "
            "--crop-size 64 512 512 "
            "--stride 16 80 80 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.55 0.65 0.75 "
            "--xy-thresholds 0.75 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-unetr-aniso-large-longpretrained"
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
    elif args.stage == "eval-cremi-rag-ablation":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_rag_ablation "
            "--crop-size 32 256 256 "
            "--stride 16 128 128 "
            "--thresholds 0.0 "
            "--backends seeded_rag scipy_agglomeration "
            "--min-size 32 "
            "--seed-method maxima_distance "
            "--seed-distance 6 10 14 "
            "--boundary-threshold 0.35 0.50 "
            "--min-boundary 4 16 "
            "--score-mode mean q25 min "
            "--rag-quantile 0.25 "
            "--z-thresholds 0.65 0.75 0.85 0.90 0.95 "
            "--xy-thresholds 0.75 0.85 0.90 0.95 0.98 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-rag-ablation"
    elif args.stage == "eval-cremi-aniso-graph":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_aniso_graph "
            "--crop-size 32 256 256 "
            "--stride 16 128 128 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.65 0.75 0.85 0.90 0.95 "
            "--xy-thresholds 0.65 0.75 0.85 0.90 0.95 0.98 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-aniso-graph"
    elif args.stage == "eval-cremi-scale64":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_scale64 "
            "--crop-size 64 512 512 "
            "--stride 16 128 128 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.75 "
            "--xy-thresholds 0.90 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-scale64"
    elif args.stage == "eval-cremi-zdice":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_zdice.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_zdice "
            "--crop-size 32 256 256 "
            "--stride 16 128 128 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.65 0.75 0.85 0.90 "
            "--xy-thresholds 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-zdice"
    elif args.stage == "eval-cremi-zdice-focal":
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            "--config configs/finetune_cremi_real_zdice_focal.yaml "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            "--output-dir outputs/eval_cremi_zdice_focal "
            "--crop-size 32 256 256 "
            "--stride 16 128 128 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            "--z-thresholds 0.65 0.75 0.85 0.90 "
            "--xy-thresholds 0.85 0.90 0.95 "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = "dbmim-eval-cremi-zdice-focal"
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
