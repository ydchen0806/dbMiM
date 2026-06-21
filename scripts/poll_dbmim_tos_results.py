#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/volume/med-train/users/dchen02/code/dbMiM")
TOS = Path("/volume/med-train/users/dchen02/bin/tosutil")
CONF = Path("/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf")
BASE = "tos://agi-data/users/dchen02/dbmim/outputs"

RUNS = {
    "r9": [
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_allpretrained_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_allpretrained_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_encoderlr_allpretrained_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_encoderlr_allpretrained_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_freezeenc_allpretrained_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_freezeenc_allpretrained_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_ignore_allpretrained_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_ignore_allpretrained_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_ignore_scratch_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_ignore_scratch_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_allpretrained_r10", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_allpretrained_r10"),
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_scratch_r10", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_scratch_r10"),
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_ignore_allpretrained_r10", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_ignore_allpretrained_r10"),
    ],
    "r11": [
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_pure_allpretrained_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_pure_allpretrained_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_pure_scratch_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_pure_scratch_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_shwmse_mix_allpretrained_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_shwmse_mix_allpretrained_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_shwmse_mix_scratch_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_shwmse_mix_scratch_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_aug_encoderlr_allpretrained_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_aug_encoderlr_allpretrained_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_aug_scratch_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_aug_scratch_r11"),
    ],
}


def tos_cp(src: str, dst: Path, timeout: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [str(TOS), "cp", src, str(tmp), f"-conf={CONF}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


def tos_exists(uri: str, timeout: int) -> bool:
    try:
        proc = subprocess.run(
            [str(TOS), "ls", uri, f"-conf={CONF}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != 0:
        return False
    return "File number is: 0" not in proc.stdout


def read_last_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip().endswith("}"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows[-1] if rows else None


def summarize(group: str) -> int:
    root = ROOT / "outputs" / "tos_fetch" / group
    done = 0
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {group}")
    for train, eval_name in RUNS[group]:
        train_log = root / train / "finetune_log.jsonl"
        rec = read_last_jsonl(train_log)
        if rec:
            print(
                "LOG",
                train,
                "step",
                rec.get("step"),
                "loss",
                rec.get("train_loss"),
                "main",
                rec.get("train_main_loss"),
                "valid",
                rec.get("train_valid_fraction"),
            )
        summary_path = root / eval_name / "cremi_segmentation_summary.json"
        if summary_path.exists():
            done += 1
            try:
                summary = json.loads(summary_path.read_text())
                by_voi = summary.get("best_by_voi_sum", {})
                by_rand = summary.get("best_by_adapted_rand", {})
                print(
                    "SUMMARY",
                    eval_name,
                    "VOI",
                    by_voi.get("voi_sum"),
                    "ARAND@VOI",
                    by_voi.get("adapted_rand_error"),
                    "bestARAND",
                    by_rand.get("adapted_rand_error"),
                    "VOI@ARAND",
                    by_rand.get("voi_sum"),
                )
            except Exception as exc:
                print("SUMMARY_ERROR", eval_name, type(exc).__name__, str(exc)[:160])
    print(f"{group}_done_summaries={done}/{len(RUNS[group])}", flush=True)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(RUNS), required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--logs", action="store_true", help="Also download finetune logs; summaries are always checked.")
    args = parser.parse_args()

    while True:
        for train, eval_name in RUNS[args.group]:
            root = ROOT / "outputs" / "tos_fetch" / args.group
            if args.logs:
                log_uri = f"{BASE}/{train}/finetune_log.jsonl"
                if tos_exists(log_uri, args.timeout):
                    tos_cp(log_uri, root / train / "finetune_log.jsonl", args.timeout)
            summary_uri = f"{BASE}/{eval_name}/cremi_segmentation_summary.json"
            if tos_exists(summary_uri, args.timeout):
                tos_cp(summary_uri, root / eval_name / "cremi_segmentation_summary.json", args.timeout)
        done = summarize(args.group)
        if args.once or done >= len(RUNS[args.group]):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
