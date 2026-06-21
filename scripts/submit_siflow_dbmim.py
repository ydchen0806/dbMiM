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
ABLATION_RUNS = {
    "pretrained-r2": {
        "config": "finetune_cremi_real_unetr_aniso_pretrained_r2.yaml",
        "output": "finetune_cremi_real_unetr_aniso_pretrained_r2",
        "eval": "eval_cremi_unetr_aniso_pretrained_r2",
        "large_eval": "eval_cremi_unetr_aniso_large_pretrained_r2",
    },
    "scratch-r2": {
        "config": "finetune_cremi_real_unetr_aniso_scratch_r2.yaml",
        "output": "finetune_cremi_real_unetr_aniso_scratch_r2",
        "eval": "eval_cremi_unetr_aniso_scratch_r2",
        "large_eval": "eval_cremi_unetr_aniso_large_scratch_r2",
    },
    "lsd-pretrained-r2": {
        "config": "finetune_cremi_real_unetr_aniso_lsd_pretrained_r2.yaml",
        "output": "finetune_cremi_real_unetr_aniso_lsd_pretrained_r2",
        "eval": "eval_cremi_unetr_aniso_lsd_pretrained_r2",
        "large_eval": "eval_cremi_unetr_aniso_large_lsd_pretrained_r2",
    },
    "lsd-scratch-r2": {
        "config": "finetune_cremi_real_unetr_aniso_lsd_scratch_r2.yaml",
        "output": "finetune_cremi_real_unetr_aniso_lsd_scratch_r2",
        "eval": "eval_cremi_unetr_aniso_lsd_scratch_r2",
        "large_eval": "eval_cremi_unetr_aniso_large_lsd_scratch_r2",
    },
    "no-dtrans": {
        "config": "finetune_cremi_real_unetr_aniso_no_dtrans.yaml",
        "output": "finetune_cremi_real_unetr_aniso_no_dtrans",
        "eval": "eval_cremi_unetr_aniso_no_dtrans",
        "large_eval": "eval_cremi_unetr_aniso_large_no_dtrans",
    },
    "dtrans2": {
        "config": "finetune_cremi_real_unetr_aniso_dtrans2.yaml",
        "output": "finetune_cremi_real_unetr_aniso_dtrans2",
        "eval": "eval_cremi_unetr_aniso_dtrans2",
        "large_eval": "eval_cremi_unetr_aniso_large_dtrans2",
    },
    "fs64": {
        "config": "finetune_cremi_real_unetr_aniso_fs64.yaml",
        "output": "finetune_cremi_real_unetr_aniso_fs64",
        "eval": "eval_cremi_unetr_aniso_fs64",
        "large_eval": "eval_cremi_unetr_aniso_large_fs64",
    },
    "boundary-loss": {
        "config": "finetune_cremi_real_unetr_aniso_boundary_loss.yaml",
        "output": "finetune_cremi_real_unetr_aniso_boundary_loss",
        "eval": "eval_cremi_unetr_aniso_boundary_loss",
        "large_eval": "eval_cremi_unetr_aniso_large_boundary_loss",
    },
    "context48": {
        "config": "finetune_cremi_real_unetr_aniso_context48.yaml",
        "output": "finetune_cremi_real_unetr_aniso_context48",
        "eval": "eval_cremi_unetr_aniso_context48",
        "large_eval": "eval_cremi_unetr_aniso_large_context48",
    },
    "neg-boundary-pretrained-r3": {
        "config": "finetune_cremi_real_unetr_aniso_neg_boundary_pretrained_r3.yaml",
        "output": "finetune_cremi_real_unetr_aniso_neg_boundary_pretrained_r3",
        "eval": "eval_cremi_unetr_aniso_neg_boundary_pretrained_r3",
        "large_eval": "eval_cremi_unetr_aniso_large_neg_boundary_pretrained_r3",
    },
    "neg-boundary-scratch-r3": {
        "config": "finetune_cremi_real_unetr_aniso_neg_boundary_scratch_r3.yaml",
        "output": "finetune_cremi_real_unetr_aniso_neg_boundary_scratch_r3",
        "eval": "eval_cremi_unetr_aniso_neg_boundary_scratch_r3",
        "large_eval": "eval_cremi_unetr_aniso_large_neg_boundary_scratch_r3",
    },
    "superhuman-pretrained-r4": {
        "config": "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r4.yaml",
        "output": "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r4",
        "eval": "eval_cremi_unetr_aniso_superhuman_pretrained_r4",
        "large_eval": "eval_cremi_unetr_aniso_large_superhuman_pretrained_r4",
        "superhuman_eval": "eval_cremi_unetr_aniso_superhuman_waterz_pretrained_r4",
    },
    "superhuman-scratch-r4": {
        "config": "finetune_cremi_real_unetr_aniso_superhuman_scratch_r4.yaml",
        "output": "finetune_cremi_real_unetr_aniso_superhuman_scratch_r4",
        "eval": "eval_cremi_unetr_aniso_superhuman_scratch_r4",
        "large_eval": "eval_cremi_unetr_aniso_large_superhuman_scratch_r4",
        "superhuman_eval": "eval_cremi_unetr_aniso_superhuman_waterz_scratch_r4",
    },
    "superhuman-pretrained-r5": {
        "config": "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5.yaml",
        "output": "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5",
        "eval": "eval_cremi_unetr_aniso_superhuman_pretrained_r5",
        "large_eval": "eval_cremi_unetr_aniso_large_superhuman_pretrained_r5",
        "superhuman_eval": "eval_cremi_unetr_aniso_superhuman_waterz_pretrained_r5",
    },
    "superhuman-scratch-r5": {
        "config": "finetune_cremi_real_unetr_aniso_superhuman_scratch_r5.yaml",
        "output": "finetune_cremi_real_unetr_aniso_superhuman_scratch_r5",
        "eval": "eval_cremi_unetr_aniso_superhuman_scratch_r5",
        "large_eval": "eval_cremi_unetr_aniso_large_superhuman_scratch_r5",
        "superhuman_eval": "eval_cremi_unetr_aniso_superhuman_waterz_scratch_r5",
    },
    "superhuman-nowiden-pretrained-r5": {
        "config": "finetune_cremi_real_unetr_aniso_superhuman_nowiden_pretrained_r5.yaml",
        "output": "finetune_cremi_real_unetr_aniso_superhuman_nowiden_pretrained_r5",
        "eval": "eval_cremi_unetr_aniso_superhuman_nowiden_pretrained_r5",
        "large_eval": "eval_cremi_unetr_aniso_large_superhuman_nowiden_pretrained_r5",
        "superhuman_eval": "eval_cremi_unetr_aniso_superhuman_waterz_nowiden_pretrained_r5",
        "calibration_eval": "eval_cremi_unetr_aniso_superhuman_calibration_nowiden_pretrained_r5",
    },
    "superhuman-bce-pretrained-r5": {
        "config": "finetune_cremi_real_unetr_aniso_superhuman_bce_pretrained_r5.yaml",
        "output": "finetune_cremi_real_unetr_aniso_superhuman_bce_pretrained_r5",
        "eval": "eval_cremi_unetr_aniso_superhuman_bce_pretrained_r5",
        "large_eval": "eval_cremi_unetr_aniso_large_superhuman_bce_pretrained_r5",
        "superhuman_eval": "eval_cremi_unetr_aniso_superhuman_waterz_bce_pretrained_r5",
        "calibration_eval": "eval_cremi_unetr_aniso_superhuman_calibration_bce_pretrained_r5",
    },
    "superhuman-bce-scratch-r5": {
        "config": "finetune_cremi_real_unetr_aniso_superhuman_bce_scratch_r5.yaml",
        "output": "finetune_cremi_real_unetr_aniso_superhuman_bce_scratch_r5",
        "eval": "eval_cremi_unetr_aniso_superhuman_bce_scratch_r5",
        "large_eval": "eval_cremi_unetr_aniso_large_superhuman_bce_scratch_r5",
        "superhuman_eval": "eval_cremi_unetr_aniso_superhuman_waterz_bce_scratch_r5",
        "calibration_eval": "eval_cremi_unetr_aniso_superhuman_calibration_bce_scratch_r5",
    },
    "superhuman-boundaryhigh-pretrained-r5": {
        "config": "finetune_cremi_real_unetr_aniso_superhuman_boundaryhigh_pretrained_r5.yaml",
        "output": "finetune_cremi_real_unetr_aniso_superhuman_boundaryhigh_pretrained_r5",
        "eval": "eval_cremi_unetr_aniso_superhuman_boundaryhigh_pretrained_r5",
        "large_eval": "eval_cremi_unetr_aniso_large_superhuman_boundaryhigh_pretrained_r5",
        "superhuman_eval": "eval_cremi_unetr_aniso_superhuman_waterz_boundaryhigh_pretrained_r5",
        "calibration_eval": "eval_cremi_unetr_aniso_superhuman_calibration_boundaryhigh_pretrained_r5",
    },
    "superhuman-encoderlr-pretrained-r5": {
        "config": "finetune_cremi_real_unetr_aniso_superhuman_encoderlr_pretrained_r5.yaml",
        "output": "finetune_cremi_real_unetr_aniso_superhuman_encoderlr_pretrained_r5",
        "eval": "eval_cremi_unetr_aniso_superhuman_encoderlr_pretrained_r5",
        "large_eval": "eval_cremi_unetr_aniso_large_superhuman_encoderlr_pretrained_r5",
        "superhuman_eval": "eval_cremi_unetr_aniso_superhuman_waterz_encoderlr_pretrained_r5",
        "calibration_eval": "eval_cremi_unetr_aniso_superhuman_calibration_encoderlr_pretrained_r5",
    },
}
ABLATION_TRAIN_STAGES = {f"finetune-cremi-unetr-aniso-{name}" for name in ABLATION_RUNS}
ABLATION_EVAL_STAGES = {f"eval-cremi-unetr-aniso-{name}" for name in ABLATION_RUNS}
ABLATION_LARGE_EVAL_STAGES = {f"eval-cremi-unetr-aniso-large-{name}" for name in ABLATION_RUNS}
ABLATION_DIAG_STAGES = {f"diagnose-cremi-unetr-aniso-{name}" for name in ABLATION_RUNS}
ABLATION_SUPERHUMAN_EVAL_STAGES = {f"eval-cremi-unetr-aniso-superhuman-{name}" for name in ABLATION_RUNS}
ABLATION_SUPERHUMAN_CALIBRATION_STAGES = {
    f"eval-cremi-unetr-aniso-superhuman-calibration-{name}"
    for name, spec in ABLATION_RUNS.items()
    if "calibration_eval" in spec
}
SUPERHUMAN_CALIBRATION_STAGES = {
    "eval-cremi-unetr-aniso-superhuman-calibration-neg-boundary-pretrained-r3",
    "eval-cremi-unetr-aniso-superhuman-calibration-neg-boundary-scratch-r3",
    "eval-cremi-unetr-aniso-superhuman-calibration-superhuman-pretrained-r4",
    "eval-cremi-unetr-aniso-superhuman-calibration-superhuman-pretrained-r5",
    "eval-cremi-unetr-aniso-superhuman-calibration-superhuman-scratch-r5",
    "eval-cremi-unetr-aniso-superhuman-calibration-all-superhuman-pretrained-r5",
    "eval-cremi-unetr-aniso-superhuman-calibration-all-superhuman-scratch-r5",
    "eval-cremi-unetr-aniso-superhuman-calibration-official-all-superhuman-pretrained-r5",
    "eval-cremi-unetr-aniso-superhuman-calibration-official-all-superhuman-scratch-r5",
    "eval-cremi-unetr-aniso-superhuman-calibration-official-superhuman-bce-pretrained-r5",
    "eval-cremi-unetr-aniso-superhuman-calibration-official-superhuman-bce-scratch-r5",
    "eval-cremi-unetr-aniso-superhuman-calibration-official-superhuman-encoderlr-pretrained-r5",
} | ABLATION_SUPERHUMAN_CALIBRATION_STAGES
SUPERHUMAN_DEP_STAGES = ABLATION_SUPERHUMAN_EVAL_STAGES | SUPERHUMAN_CALIBRATION_STAGES
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
} | ABLATION_TRAIN_STAGES | ABLATION_EVAL_STAGES | ABLATION_LARGE_EVAL_STAGES | ABLATION_DIAG_STAGES | SUPERHUMAN_DEP_STAGES
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
} | ABLATION_EVAL_STAGES | ABLATION_LARGE_EVAL_STAGES | ABLATION_DIAG_STAGES | SUPERHUMAN_DEP_STAGES


def _ablation_name_from_stage(stage: str) -> str | None:
    for prefix in [
        "finetune-cremi-unetr-aniso-",
        "eval-cremi-unetr-aniso-superhuman-calibration-",
        "eval-cremi-unetr-aniso-large-",
        "eval-cremi-unetr-aniso-superhuman-",
        "eval-cremi-unetr-aniso-",
        "diagnose-cremi-unetr-aniso-",
    ]:
        if stage.startswith(prefix):
            name = stage[len(prefix) :]
            if name in ABLATION_RUNS:
                return name
    return None


def _training_output_dir(stage: str) -> str | None:
    if stage == "pretrain-cremi":
        return "outputs/pretrain_cremi_real_dbmim"
    if stage == "pretrain-cremi-long":
        return "outputs/pretrain_cremi_real_long_dbmim"
    if stage == "finetune-cremi":
        return "outputs/finetune_cremi_real_dbmim"
    if stage == "finetune-cremi-unetr-pretrained":
        return "outputs/finetune_cremi_real_unetr_pretrained"
    if stage == "finetune-cremi-unetr-scratch":
        return "outputs/finetune_cremi_real_unetr_scratch"
    if stage == "finetune-cremi-unetr-aniso-pretrained":
        return "outputs/finetune_cremi_real_unetr_aniso_pretrained"
    if stage == "finetune-cremi-unetr-aniso-scratch":
        return "outputs/finetune_cremi_real_unetr_aniso_scratch"
    if stage == "finetune-cremi-unetr-aniso-longpretrained":
        return "outputs/finetune_cremi_real_unetr_aniso_longpretrained"
    if stage == "finetune-cremi-zdice":
        return "outputs/finetune_cremi_real_zdice"
    if stage == "finetune-cremi-zdice-focal":
        return "outputs/finetune_cremi_real_zdice_focal"
    ablation_name = _ablation_name_from_stage(stage)
    if stage in ABLATION_TRAIN_STAGES and ablation_name is not None:
        return f"outputs/{ABLATION_RUNS[ablation_name]['output']}"
    return None


def _entrypoint_lines(entrypoint: str, sync_output_dir: str | None) -> list[str]:
    if sync_output_dir is None:
        return [entrypoint]
    remote_dir = f"{TOS_OUTPUT_PREFIX}/{Path(sync_output_dir).name}"
    return [
        "DBMIM_SYNC_SEC=${DBMIM_SYNC_SEC:-60}",
        "dbmim_sync_output_loop() {",
        "  local out_dir=\"$1\"",
        "  while true; do",
        "    if [ -d \"$out_dir\" ]; then",
        "      for f in finetuned_latest.pt finetuned_best.pt pretrained_latest.pt finetune_log.jsonl pretrain_log.jsonl; do",
        "        if [ -f \"$out_dir/$f\" ]; then",
        f"          bin/tosutil cp \"$out_dir/$f\" {remote_dir}/$f -conf=\"$TOS_CONF\" >/dev/null 2>&1 || true",
        "        fi",
        "      done",
        "    fi",
        "    sleep \"$DBMIM_SYNC_SEC\"",
        "  done",
        "}",
        f"mkdir -p {sync_output_dir}",
        f"dbmim_sync_output_loop {sync_output_dir} &",
        "dbmim_sync_pid=$!",
        "set +e",
        entrypoint,
        "dbmim_status=$?",
        "set -e",
        "kill \"$dbmim_sync_pid\" >/dev/null 2>&1 || true",
        "wait \"$dbmim_sync_pid\" >/dev/null 2>&1 || true",
        f"bin/tosutil cp {sync_output_dir} {TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\" || true",
        "if [ \"$dbmim_status\" -ne 0 ]; then",
        "  exit \"$dbmim_status\"",
        "fi",
    ]


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

    ablation_configs = {spec["config"] for spec in ABLATION_RUNS.values()}
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
        *[(spec["config"], f"outputs/{spec['output']}") for spec in ABLATION_RUNS.values()],
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
        if name not in ablation_configs:
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
        "scripts/evaluate_cremi_diagnostics.py",
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
    superhuman_wheelhouse = PROJECT / "outputs" / "wheelhouse_superhuman_eval"
    if superhuman_wheelhouse.exists():
        shutil.copytree(superhuman_wheelhouse, out / "wheelhouse_superhuman_eval")
    # Keep v0.8 as the default offline source. The newer waterz tree supports
    # MeanAffinityProvider, but its pyproject metadata is rejected by the older
    # setuptools in several SiFlow images; dedicated mean-scoring sweeps should
    # patch/package it explicitly instead of changing the stable path.
    waterz_source = PROJECT.parent / "_refs" / "waterz_v08"
    if not waterz_source.exists():
        waterz_source = PROJECT.parent / "_refs" / "waterz"
    if stage in SUPERHUMAN_DEP_STAGES and waterz_source.exists():
        shutil.copytree(waterz_source, out / "third_party" / "waterz", ignore=shutil.ignore_patterns(".git"))
    boost_headers = PROJECT / "third_party" / "boost_1_84_0" / "boost"
    if stage in SUPERHUMAN_DEP_STAGES and boost_headers.exists():
        boost_dst = out / "third_party" / "boost" / "include" / "boost"
        boost_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(boost_headers, boost_dst)
    if stage in CREMI_STAGES:
        _patch_cremi_configs(out)
    if TOSUTIL.exists():
        (out / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(TOSUTIL.resolve(), out / "bin" / "tosutil")
        os.chmod(out / "bin" / "tosutil", 0o700)

    prelude = []
    postlude = []
    sync_output_dir = _training_output_dir(stage)
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
        *ABLATION_TRAIN_STAGES,
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
            ]
        )
    if stage in SUPERHUMAN_DEP_STAGES:
        prelude.extend(
            [
                "if [ -d wheelhouse_superhuman_eval ]; then",
                "  python -m pip install --user --no-index --find-links wheelhouse_superhuman_eval scikit-image",
                "fi",
                "if ! python - <<'PY'\nimport importlib.util\nraise SystemExit(0 if importlib.util.find_spec('waterz') else 1)\nPY\nthen",
                "  if [ -d third_party/waterz ]; then",
                "    if [ -d third_party/boost/include ]; then",
                "      export CPLUS_INCLUDE_PATH=\"$PWD/third_party/boost/include:${CPLUS_INCLUDE_PATH:-}\"",
                "      export BOOST_INCLUDEDIR=\"$PWD/third_party/boost/include\"",
                "    fi",
                "    if [ -d /usr/include/boost ] || [ -d third_party/boost/include/boost ] || [ -n \"${BOOST_ROOT:-}\" ] || [ -n \"${BOOST_INCLUDEDIR:-}\" ]; then",
                "      python -m pip install --user --no-build-isolation --no-index --find-links wheelhouse_superhuman_eval third_party/waterz",
                "    else",
                "      echo 'waterz build skipped: boost headers missing in this image' >&2",
                "    fi",
                "  fi",
                "fi",
            ]
        )
    if stage in {
        "eval-cremi",
        "eval-cremi-sweep",
        "eval-cremi-gpu-probe",
        "eval-cremi-rag-ablation",
        "eval-cremi-aniso-graph",
        "eval-cremi-scale64",
    }:
        prelude.extend(
            [
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
    for name, spec in ABLATION_RUNS.items():
        eval_stage_map[f"eval-cremi-unetr-aniso-{name}"] = (spec["output"], "DBMIM_EVAL_CKPT")
        eval_stage_map[f"eval-cremi-unetr-aniso-large-{name}"] = (spec["output"], "DBMIM_EVAL_CKPT")
        eval_stage_map[f"diagnose-cremi-unetr-aniso-{name}"] = (spec["output"], "DBMIM_EVAL_CKPT")
        eval_stage_map[f"eval-cremi-unetr-aniso-superhuman-{name}"] = (spec["output"], "DBMIM_EVAL_CKPT")
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-neg-boundary-pretrained-r3"] = (
        "finetune_cremi_real_unetr_aniso_neg_boundary_pretrained_r3",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-neg-boundary-scratch-r3"] = (
        "finetune_cremi_real_unetr_aniso_neg_boundary_scratch_r3",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-superhuman-pretrained-r4"] = (
        "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r4",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-superhuman-pretrained-r5"] = (
        "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-superhuman-scratch-r5"] = (
        "finetune_cremi_real_unetr_aniso_superhuman_scratch_r5",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-all-superhuman-pretrained-r5"] = (
        "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-all-superhuman-scratch-r5"] = (
        "finetune_cremi_real_unetr_aniso_superhuman_scratch_r5",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-official-all-superhuman-pretrained-r5"] = (
        "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-official-all-superhuman-scratch-r5"] = (
        "finetune_cremi_real_unetr_aniso_superhuman_scratch_r5",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-official-superhuman-bce-pretrained-r5"] = (
        "finetune_cremi_real_unetr_aniso_superhuman_bce_pretrained_r5",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map["eval-cremi-unetr-aniso-superhuman-calibration-official-superhuman-bce-scratch-r5"] = (
        "finetune_cremi_real_unetr_aniso_superhuman_bce_scratch_r5",
        "DBMIM_EVAL_CKPT",
    )
    eval_stage_map[
        "eval-cremi-unetr-aniso-superhuman-calibration-official-superhuman-encoderlr-pretrained-r5"
    ] = (
        "finetune_cremi_real_unetr_aniso_superhuman_encoderlr_pretrained_r5",
        "DBMIM_EVAL_CKPT",
    )
    for name, spec in ABLATION_RUNS.items():
        if "calibration_eval" in spec:
            eval_stage_map[f"eval-cremi-unetr-aniso-superhuman-calibration-{name}"] = (
                spec["output"],
                "DBMIM_EVAL_CKPT",
            )
    if stage in eval_stage_map:
        model_prefix, env_key = eval_stage_map[stage]
        prelude.extend(
            [
                f"mkdir -p outputs/{model_prefix}",
                "if bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/{model_prefix}/finetuned_latest.pt "
                f"outputs/{model_prefix}/finetuned_latest.pt -conf=\"$TOS_CONF\"; then",
                f"  export {env_key}=outputs/{model_prefix}/finetuned_latest.pt",
                "else",
                "  bin/tosutil cp "
                f"{TOS_OUTPUT_PREFIX}/{model_prefix}/finetuned_best.pt "
                f"outputs/{model_prefix}/finetuned_best.pt -conf=\"$TOS_CONF\"",
                f"  export {env_key}=outputs/{model_prefix}/finetuned_best.pt",
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
    ablation_name = _ablation_name_from_stage(stage)
    if sync_output_dir is None and stage == "finetune-cremi-zdice":
        postlude.extend(
            [
                "bin/tosutil cp outputs/finetune_cremi_real_zdice "
                f"{TOS_OUTPUT_PREFIX} -r -conf=\"$TOS_CONF\"",
            ]
        )
    if sync_output_dir is None and stage == "finetune-cremi-zdice-focal":
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
    for name, spec in ABLATION_RUNS.items():
        eval_output_dirs[f"eval-cremi-unetr-aniso-{name}"] = f"outputs/{spec['eval']}"
        eval_output_dirs[f"eval-cremi-unetr-aniso-large-{name}"] = f"outputs/{spec['large_eval']}"
        eval_output_dirs[f"diagnose-cremi-unetr-aniso-{name}"] = f"outputs/diagnose_cremi_unetr_aniso_{name}"
        safe_name = name.replace("-", "_")
        eval_output_dirs[f"eval-cremi-unetr-aniso-superhuman-{name}"] = (
            f"outputs/{spec.get('superhuman_eval', f'eval_cremi_unetr_aniso_superhuman_waterz_{safe_name}')}"
        )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-neg-boundary-pretrained-r3"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_neg_boundary_pretrained_r3"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-neg-boundary-scratch-r3"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_neg_boundary_scratch_r3"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-superhuman-pretrained-r4"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_superhuman_pretrained_r4"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-superhuman-pretrained-r5"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_superhuman_pretrained_r5"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-superhuman-scratch-r5"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_superhuman_scratch_r5"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-all-superhuman-pretrained-r5"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_all_superhuman_pretrained_r5"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-all-superhuman-scratch-r5"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_all_superhuman_scratch_r5"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-official-all-superhuman-pretrained-r5"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_all_superhuman_pretrained_r5"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-official-all-superhuman-scratch-r5"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_all_superhuman_scratch_r5"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-official-superhuman-bce-pretrained-r5"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_superhuman_bce_pretrained_r5"
    )
    eval_output_dirs["eval-cremi-unetr-aniso-superhuman-calibration-official-superhuman-bce-scratch-r5"] = (
        "outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_superhuman_bce_scratch_r5"
    )
    eval_output_dirs[
        "eval-cremi-unetr-aniso-superhuman-calibration-official-superhuman-encoderlr-pretrained-r5"
    ] = "outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_superhuman_encoderlr_pretrained_r5"
    for name, spec in ABLATION_RUNS.items():
        if "calibration_eval" in spec:
            eval_output_dirs[f"eval-cremi-unetr-aniso-superhuman-calibration-{name}"] = (
                f"outputs/{spec['calibration_eval']}"
            )
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
                *_entrypoint_lines(entrypoint, sync_output_dir),
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
            *sorted(ABLATION_TRAIN_STAGES),
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
            *sorted(ABLATION_EVAL_STAGES),
            *sorted(ABLATION_LARGE_EVAL_STAGES),
            *sorted(ABLATION_DIAG_STAGES),
            *sorted(SUPERHUMAN_DEP_STAGES),
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
    elif args.stage in ABLATION_TRAIN_STAGES:
        ablation_name = _ablation_name_from_stage(args.stage)
        if ablation_name is None:
            raise ValueError(f"unknown ablation stage: {args.stage}")
        spec = ABLATION_RUNS[ablation_name]
        entrypoint = f"python -m torch.distributed.run --nproc_per_node={nproc} train_finetune.py --config configs/{spec['config']}"
        prefix = f"dbmim-finetune-cremi-unetr-aniso-{ablation_name}"
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
    elif args.stage in ABLATION_EVAL_STAGES | ABLATION_LARGE_EVAL_STAGES:
        ablation_name = _ablation_name_from_stage(args.stage)
        if ablation_name is None:
            raise ValueError(f"unknown ablation stage: {args.stage}")
        spec = ABLATION_RUNS[ablation_name]
        large = args.stage in ABLATION_LARGE_EVAL_STAGES
        out_dir = spec["large_eval"] if large else spec["eval"]
        crop = "64 512 512" if large else "32 320 320"
        z_thresholds = "0.55 0.65 0.75" if large else "0.45 0.55 0.65 0.75 0.85"
        xy_thresholds = "0.75 0.85 0.90 0.95" if large else "0.65 0.75 0.85 0.90 0.95"
        entrypoint = (
            "python scripts/evaluate_cremi_segmentation.py "
            f"--config configs/{spec['config']} "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            f"--output-dir outputs/{out_dir} "
            f"--crop-size {crop} "
            "--stride 16 80 80 "
            "--thresholds 0.0 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 32 "
            f"--z-thresholds {z_thresholds} "
            f"--xy-thresholds {xy_thresholds} "
            "--max-samples 3 "
            "--device cuda"
        )
        prefix = f"dbmim-eval-cremi-unetr-aniso-{'large-' if large else ''}{ablation_name}"
    elif args.stage in ABLATION_DIAG_STAGES:
        ablation_name = _ablation_name_from_stage(args.stage)
        if ablation_name is None:
            raise ValueError(f"unknown diagnostic stage: {args.stage}")
        spec = ABLATION_RUNS[ablation_name]
        out_dir = f"diagnose_cremi_unetr_aniso_{ablation_name}"
        entrypoint = (
            "python scripts/evaluate_cremi_diagnostics.py "
            f"--config configs/{spec['config']} "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            f"--output-dir outputs/{out_dir} "
            "--crop-size 32 320 320 "
            "--stride 16 80 80 "
            "--backends graph_cc cupy_graph_cc "
            "--min-size 0 "
            "--z-thresholds 0.85 0.90 0.95 0.975 0.99 0.995 "
            "--xy-thresholds 0.90 0.95 0.975 0.99 0.995 0.999 "
            "--max-samples 3 "
            "--device cuda "
            "--include-oracle-affinity "
            "--include-inverted-affinity "
            "--diagnostics"
        )
        prefix = f"dbmim-diagnose-cremi-unetr-aniso-{ablation_name}"
    elif args.stage in SUPERHUMAN_CALIBRATION_STAGES:
        ablation_name = _ablation_name_from_stage(args.stage)
        cremi_boundary_ignore_args = ""
        if ablation_name is not None and "calibration_eval" in ABLATION_RUNS[ablation_name]:
            spec = ABLATION_RUNS[ablation_name]
            config = spec["config"]
            out_dir = spec["calibration_eval"]
            suffix = ablation_name
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "0 0 0 -0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "1"
        elif args.stage.endswith("official-all-superhuman-pretrained-r5"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_official_all_superhuman_pretrained_r5"
            suffix = "official-all-superhuman-pretrained-r5"
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "-0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "0"
            cremi_boundary_ignore_args = "--cremi-boundary-ignore-distance-xy 1 --cremi-boundary-ignore-distance-z 0 "
        elif args.stage.endswith("official-all-superhuman-scratch-r5"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_scratch_r5.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_official_all_superhuman_scratch_r5"
            suffix = "official-all-superhuman-scratch-r5"
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "-0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "0"
            cremi_boundary_ignore_args = "--cremi-boundary-ignore-distance-xy 1 --cremi-boundary-ignore-distance-z 0 "
        elif args.stage.endswith("official-superhuman-bce-pretrained-r5"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_bce_pretrained_r5.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_official_superhuman_bce_pretrained_r5"
            suffix = "official-superhuman-bce-pretrained-r5"
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "0 0 0 -0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "1"
            cremi_boundary_ignore_args = "--cremi-boundary-ignore-distance-xy 1 --cremi-boundary-ignore-distance-z 0 "
        elif args.stage.endswith("official-superhuman-bce-scratch-r5"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_bce_scratch_r5.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_official_superhuman_bce_scratch_r5"
            suffix = "official-superhuman-bce-scratch-r5"
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "0 0 0 -0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "1"
            cremi_boundary_ignore_args = "--cremi-boundary-ignore-distance-xy 1 --cremi-boundary-ignore-distance-z 0 "
        elif args.stage.endswith("official-superhuman-encoderlr-pretrained-r5"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_encoderlr_pretrained_r5.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_official_superhuman_encoderlr_pretrained_r5"
            suffix = "official-superhuman-encoderlr-pretrained-r5"
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "0 0 0 -0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "1"
            cremi_boundary_ignore_args = "--cremi-boundary-ignore-distance-xy 1 --cremi-boundary-ignore-distance-z 0 "
        elif args.stage.endswith("all-superhuman-pretrained-r5"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_all_superhuman_pretrained_r5"
            suffix = "all-superhuman-pretrained-r5"
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "-0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "0"
        elif args.stage.endswith("all-superhuman-scratch-r5"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_scratch_r5.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_all_superhuman_scratch_r5"
            suffix = "all-superhuman-scratch-r5"
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "-0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "0"
        elif args.stage.endswith("superhuman-pretrained-r5"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_superhuman_pretrained_r5"
            suffix = "superhuman-pretrained-r5"
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "0 0 0 -0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "1"
        elif args.stage.endswith("superhuman-scratch-r5"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_scratch_r5.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_superhuman_scratch_r5"
            suffix = "superhuman-scratch-r5"
            thresholds = "0.05 0.10 0.20 0.30 0.50"
            calibration_biases = "0 0 0 -0.25 -0.5 -0.5 -0.5 -1.0 -1.0"
            max_samples = "1"
        elif args.stage.endswith("superhuman-pretrained-r4"):
            config = "finetune_cremi_real_unetr_aniso_superhuman_pretrained_r4.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_superhuman_pretrained_r4"
            suffix = "superhuman-pretrained-r4"
            thresholds = "0.05 0.10 0.20 0.30"
            calibration_biases = "0 0 0 -0.5 -1.0 -1.0 -1.0 -2.0 -2.0"
            max_samples = "1"
        elif args.stage.endswith("pretrained-r3"):
            config = "finetune_cremi_real_unetr_aniso_neg_boundary_pretrained_r3.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_neg_boundary_pretrained_r3"
            suffix = "neg-boundary-pretrained-r3"
            thresholds = "0.30 0.50"
            calibration_biases = "0 0 0 -0.5 -1.0 -1.0 -1.0 -2.0 -2.0 -1.5 -3.0 -3.0"
            max_samples = "1"
        else:
            config = "finetune_cremi_real_unetr_aniso_neg_boundary_scratch_r3.yaml"
            out_dir = "eval_cremi_unetr_aniso_superhuman_calibration_neg_boundary_scratch_r3"
            suffix = "neg-boundary-scratch-r3"
            thresholds = "0.30 0.50"
            calibration_biases = "0 0 0 -0.5 -1.0 -1.0 -1.0 -2.0 -2.0 -1.5 -3.0 -3.0"
            max_samples = "1"
        entrypoint = (
            "python - <<'PY'\n"
            "import importlib.util\n"
            "missing=[m for m in ['skimage','waterz','mahotas'] if importlib.util.find_spec(m) is None]\n"
            "print({'superhuman_calibration_missing_modules': missing})\n"
            "if missing:\n"
            "    raise SystemExit('missing SuperHuman calibration modules: '+','.join(missing))\n"
            "PY\n"
            "python scripts/evaluate_cremi_segmentation.py "
            f"--config configs/{config} "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            f"--output-dir outputs/{out_dir} "
            "--crop-size 0 0 0 "
            "--stride 16 80 80 "
            f"--thresholds {thresholds} "
            "--backends waterz "
            "--min-size 0 "
            "--seed-method maxima_distance "
            "--seed-distance 10 "
            "--boundary-threshold 0.5 "
            "--waterz-scoring hist_quantile "
            "--metric-backend skimage "
            "--replicate-affinity-boundary "
            f"{cremi_boundary_ignore_args}"
            f"--calibration-biases {calibration_biases} "
            "--calibration-temperatures 1.0 "
            f"--max-samples {max_samples} "
            "--device cuda "
            "--fail-on-backend-error"
        )
        prefix = f"dbmim-eval-cremi-superhuman-calibration-{suffix}"
    elif args.stage in ABLATION_SUPERHUMAN_EVAL_STAGES:
        ablation_name = _ablation_name_from_stage(args.stage)
        if ablation_name is None:
            raise ValueError(f"unknown SuperHuman eval stage: {args.stage}")
        spec = ABLATION_RUNS[ablation_name]
        out_dir = spec.get("superhuman_eval", f"eval_cremi_unetr_aniso_superhuman_waterz_{ablation_name.replace('-', '_')}")
        entrypoint = (
            "python - <<'PY'\n"
            "import importlib.util\n"
            "missing=[m for m in ['skimage','waterz','mahotas'] if importlib.util.find_spec(m) is None]\n"
            "print({'superhuman_eval_missing_modules': missing})\n"
            "if missing:\n"
            "    raise SystemExit('missing SuperHuman eval modules: '+','.join(missing))\n"
            "PY\n"
            "python scripts/evaluate_cremi_segmentation.py "
            f"--config configs/{spec['config']} "
            "--checkpoint \"$DBMIM_EVAL_CKPT\" "
            "--data-dir data/CREMI "
            f"--output-dir outputs/{out_dir} "
            "--crop-size 0 0 0 "
            "--stride 16 80 80 "
            "--thresholds 0.30 0.40 0.50 0.60 0.70 "
            "--backends waterz "
            "--min-size 0 "
            "--seed-method maxima_distance "
            "--seed-distance 10 "
            "--boundary-threshold 0.5 "
            "--waterz-scoring hist_quantile "
            "--metric-backend skimage "
            "--replicate-affinity-boundary "
            "--max-samples 3 "
            "--device cuda "
            "--fail-on-backend-error"
        )
        prefix = f"dbmim-eval-cremi-unetr-aniso-superhuman-{ablation_name}"
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
