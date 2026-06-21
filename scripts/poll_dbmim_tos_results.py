#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/volume/med-train/users/dchen02/code/dbMiM")
TOS = Path("/volume/med-train/users/dchen02/bin/tosutil")
CONF = Path("/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf")
SIFLOW_PY = Path("/volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python")
BASE = "tos://agi-data/users/dchen02/dbmim/outputs"
GROUP_KEYS = [
    "affinity_variant",
    "calibration_bias_z",
    "calibration_bias_y",
    "calibration_bias_x",
    "calibration_temperature",
    "backend",
    "threshold",
    "seed_distance",
    "boundary_threshold",
    "min_boundary",
    "score_mode",
    "rag_quantile",
    "waterz_scoring",
    "z_threshold",
    "xy_threshold",
]
METRIC_KEYS = [
    "adapted_rand_error",
    "rand_fscore",
    "rand_precision",
    "rand_recall",
    "voi_split",
    "voi_merge",
    "voi_sum",
    "affinity_dice",
    "affinity_iou",
    "inference_sec",
    "postprocess_sec",
    "metrics_sec",
]

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
    "r12": [
        ("finetune_cremi_real_unetr_aniso_em_bce_encoderlr_allpretrained_r12", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_allpretrained_r12"),
        ("finetune_cremi_real_unetr_aniso_em_bce_encoderlr_scratch_r12", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_scratch_r12"),
    ],
    "r13": [
        ("finetune_cremi_real_unetr_aniso_em_shwmse_allpretrained_r13", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_allpretrained_r13"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_scratch_r13", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_scratch_r13"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_allpretrained_r13", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_allpretrained_r13"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_scratch_r13", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_scratch_r13"),
    ],
    "r14q": [
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_rank_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_rank_allpretrained_r14q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_calib_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_calib_allpretrained_r14q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_allpretrained_r14q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_allpretrained_r14q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws15_bcar_rank_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws15_bcar_rank_allpretrained_r14q"),
    ],
    "r14": [
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_rank_allpretrained_r14", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_rank_allpretrained_r14"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_calib_allpretrained_r14", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_calib_allpretrained_r14"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_mempretrained_r14", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_mempretrained_r14"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_mempretrained_r14", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_mempretrained_r14"),
    ],
}

SIFLOW_UUIDS = {
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_allpretrained_r9": "99ab7d58-8886-430e-86fa-92c9d4a0fcae",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_encoderlr_allpretrained_r9": "33638465-d404-4229-b150-cdbf401e8159",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_freezeenc_allpretrained_r9": "7ad21665-e4a3-4766-903e-d89088429919",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_ignore_allpretrained_r9": "37fe0514-76a0-4d74-adb5-c51e4b1d7dcc",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_ignore_scratch_r9": "885406a6-6572-46a0-8bbe-da607132b26a",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_allpretrained_r10": "6c7675a1-7661-4363-9c65-90e1f2e7129e",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_scratch_r10": "af35134e-a7ea-4f27-bfeb-4777dd48ae5b",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_ignore_allpretrained_r10": "20239e88-03f5-4269-a348-da79dd2adbb4",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_pure_allpretrained_r11": "f09b5b01-ef57-4490-8c5b-4184757cbd01",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_pure_scratch_r11": "a3eedbdf-70d0-496f-9148-1bc6f082a53d",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_shwmse_mix_allpretrained_r11": "71106220-8efa-48af-b512-c3141e76831d",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_shwmse_mix_scratch_r11": "bc6d23ed-c43d-4d9b-a61b-8ce955a1e64e",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_aug_encoderlr_allpretrained_r11": "f2e6d5da-66c6-4ae2-a66f-23cb139f3d0b",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_aug_scratch_r11": "bd71349d-b066-4cd2-90c9-5e3145ebdb1f",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_allpretrained_r12": "70c97464-7179-41b8-80f7-9ccf4f94ff25",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_scratch_r12": "bfc6ca92-13fa-4d1e-b234-fab4b8e3c14c",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_allpretrained_r13": "4be6ecf8-5053-4693-bac8-7b5f15aa2df9",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_scratch_r13": "e0a7eb92-bb68-49e6-bcc3-5e992347b98f",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_allpretrained_r13": "64c9f7f4-5f0f-4658-b4a6-73acb32e4e5e",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_scratch_r13": "28c140b6-589d-4a62-9ffb-ddea4b0eb2c7",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_rank_allpretrained_r14q": "fa076c76-f3bf-4eac-91ac-c8f4a1677062",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_calib_allpretrained_r14q": "cb95420a-2482-48a6-a3bc-9cb86c51c8d3",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_allpretrained_r14q": "38b18ca3-d4c8-4fd2-94d9-632f590d92ce",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_allpretrained_r14q": "2f63a6dc-c7a6-4e8b-97de-34ab80985b40",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws15_bcar_rank_allpretrained_r14q": "30cb7b4a-ac52-402a-9085-c97749ab5f2b",
}


def tos_cp(src: str, dst: Path, timeout: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [str(TOS), "cp", src, str(tmp), "-f", "-bt=fns", f"-conf={CONF}"],
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
            [str(TOS), "ls", uri, "-bt=fns", f"-conf={CONF}"],
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


def parse_metric_rows_from_siflow(uuid: str, region: str, cluster: str) -> list[dict]:
    code = r"""
import json
import sys
from siflow import SiFlow

uuid, region, cluster = sys.argv[1:4]
client = SiFlow(region=region, cluster=cluster)
try:
    logs = client.tasks.query_logs(uuid, limit=2000, sort_order="asc")
except Exception:
    logs = client.tasks.query_logs(uuid, limit=2000, sort_order="desc")
print(json.dumps([str(item.content) for item in logs.logs]))
"""
    proc = subprocess.run(
        [str(SIFLOW_PY), "-c", code, uuid, region, cluster],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-500:] or proc.stdout.strip()[-500:])
    rows = []
    for content in json.loads(proc.stdout):
        for line in str(content).splitlines():
            start = line.find("{'sample'")
            if start < 0 or "voi_sum" not in line or "adapted_rand_error" not in line:
                continue
            try:
                row = ast.literal_eval(line[start:].strip())
            except Exception:
                continue
            if isinstance(row, dict) and "voi_sum" in row:
                rows.append(row)
    dedup = []
    seen = set()
    for row in rows:
        key = (row.get("sample"), *(row.get(name) for name in GROUP_KEYS))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
    return dedup


def _mean_metric(rows: list[dict], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return sum(values) / len(values) if values else float("nan")


def aggregate_metric_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[object, ...], list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(name, "") for name in GROUP_KEYS), []).append(row)
    per_backend_threshold = []
    for key_tuple, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        mean_row = dict(zip(GROUP_KEYS, key_tuple))
        mean_row["n"] = len(group_rows)
        for metric in METRIC_KEYS:
            mean_row[metric] = _mean_metric(group_rows, metric)
        per_backend_threshold.append(mean_row)

    threshold_grouped: dict[float, list[dict]] = {}
    for row in rows:
        threshold_grouped.setdefault(float(row.get("threshold", 0.0)), []).append(row)
    per_threshold = []
    for threshold, group_rows in sorted(threshold_grouped.items()):
        mean_row = {"threshold": threshold, "n": len(group_rows)}
        for metric in METRIC_KEYS:
            mean_row[metric] = _mean_metric(group_rows, metric)
        per_threshold.append(mean_row)
    return per_backend_threshold, per_threshold


def write_siflow_fallback_summary(
    eval_name: str,
    path: Path,
    *,
    region: str,
    cluster: str,
) -> bool:
    uuid = SIFLOW_UUIDS.get(eval_name)
    if not uuid:
        return False
    try:
        rows = parse_metric_rows_from_siflow(uuid, region, cluster)
    except Exception as exc:
        print("SIFLOW_FALLBACK_ERROR", eval_name, type(exc).__name__, str(exc)[:160])
        return False
    if not rows:
        return False
    per_backend_threshold, per_threshold = aggregate_metric_rows(rows)
    if not per_backend_threshold:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "siflow_stdout_fallback",
        "uuid": uuid,
        "num_records": len(rows),
        "sample_names": sorted({str(row.get("sample")) for row in rows}),
        "records": rows,
        "per_backend_threshold": per_backend_threshold,
        "per_threshold": per_threshold,
        "best_by_voi_sum": min(per_backend_threshold, key=lambda row: row.get("voi_sum", float("inf"))),
        "best_by_adapted_rand": min(
            per_backend_threshold,
            key=lambda row: row.get("adapted_rand_error", float("inf")),
        ),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return True


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
    parser.add_argument(
        "--siflow-fallback",
        action="store_true",
        help="If TOS summary download is missing or slow, rebuild summaries from SiFlow stdout logs.",
    )
    parser.add_argument("--siflow-region", default="cn-shanghai")
    parser.add_argument("--siflow-cluster", default="changliu")
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
            summary_path = root / eval_name / "cremi_segmentation_summary.json"
            if args.siflow_fallback and not summary_path.exists():
                write_siflow_fallback_summary(
                    eval_name,
                    summary_path,
                    region=args.siflow_region,
                    cluster=args.siflow_cluster,
                )
        done = summarize(args.group)
        if args.once or done >= len(RUNS[args.group]):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
