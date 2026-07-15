#!/usr/bin/env python3
"""Validate and summarize a paired MEC08/MEC10 WaterZ crop sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SAMPLES = ("mec8nm_08.h5", "mec8nm_10.h5")
PRIMARY_FIELDS = ("adapted_rand_error", "voi_sum", "n_pred")


def _float(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} in metric row: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in metric row: {row}")
    return value


def _metric_key(row: dict[str, str]) -> tuple[str, str, float, float, float, float, float]:
    return (
        row["sample"],
        row["backend"],
        round(_float(row, "calibration_bias_z"), 8),
        round(_float(row, "calibration_bias_y"), 8),
        round(_float(row, "calibration_bias_x"), 8),
        round(_float(row, "calibration_temperature"), 8),
        round(_float(row, "threshold"), 8),
    )


def _load_candidates(
    path: Path,
    *,
    samples: tuple[str, ...],
    expected_rows: int,
    backend: str,
) -> tuple[dict[tuple[float, float, float, float, float], dict[str, Any]], dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} raw rows in {path}, found {len(rows)}")
    found_samples = {row.get("sample", "") for row in rows}
    if found_samples != set(samples):
        raise ValueError(f"sample set mismatch in {path}: expected {sorted(samples)}, got {sorted(found_samples)}")
    found_backends = {row.get("backend", "") for row in rows}
    if found_backends != {backend}:
        raise ValueError(f"backend mismatch in {path}: expected {backend}, got {sorted(found_backends)}")

    grouped: dict[tuple[str, str, float, float, float, float, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for field in ("adapted_rand_error", "voi_sum"):
            _float(row, field)
        grouped[_metric_key(row)].append(row)

    collapsed: list[dict[str, str]] = []
    duplicates = 0
    for key, duplicate_rows in grouped.items():
        primary_values = {
            tuple(round(_float(row, field), 12) for field in PRIMARY_FIELDS)
            for row in duplicate_rows
        }
        if len(primary_values) != 1:
            raise ValueError(f"conflicting duplicate rows for {key}: {primary_values}")
        duplicates += len(duplicate_rows) - 1
        collapsed.append(duplicate_rows[0])

    per_config: dict[tuple[float, float, float, float, float], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in collapsed:
        _, _, bias_z, bias_y, bias_x, temperature, threshold = _metric_key(row)
        config = (bias_z, bias_y, bias_x, temperature, threshold)
        per_config[config][row["sample"]] = row

    candidates: dict[tuple[float, float, float, float, float], dict[str, Any]] = {}
    for config, sample_rows in per_config.items():
        if set(sample_rows) != set(samples):
            raise ValueError(f"incomplete paired configuration {config}: {sorted(sample_rows)}")
        per_sample: dict[str, dict[str, Any]] = {}
        for sample in samples:
            row = sample_rows[sample]
            per_sample[sample] = {
                "arand": _float(row, "adapted_rand_error"),
                "voi": _float(row, "voi_sum"),
                "n_pred": int(_float(row, "n_pred")),
                "affinity_variant": row.get("affinity_variant", ""),
            }
        mean_arand = sum(per_sample[sample]["arand"] for sample in samples) / len(samples)
        mean_voi = sum(per_sample[sample]["voi"] for sample in samples) / len(samples)
        candidates[config] = {
            "bias": list(config[:3]),
            "temperature": config[3],
            "threshold": config[4],
            "mean_arand": mean_arand,
            "mean_voi": mean_voi,
            "balance": math.sqrt(mean_arand * mean_voi),
            "worst_balance": max(math.sqrt(per_sample[sample]["arand"] * per_sample[sample]["voi"]) for sample in samples),
            "samples": per_sample,
        }
    return candidates, {"raw_rows": len(rows), "collapsed_rows": len(collapsed), "duplicate_rows": duplicates}


def _pareto(candidates: dict[tuple[float, float, float, float, float], dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(candidates.values())
    frontier = []
    for candidate in values:
        dominated = any(
            other is not candidate
            and other["mean_arand"] <= candidate["mean_arand"]
            and other["mean_voi"] <= candidate["mean_voi"]
            and (other["mean_arand"] < candidate["mean_arand"] or other["mean_voi"] < candidate["mean_voi"])
            for other in values
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda value: (value["balance"], value["worst_balance"], value["mean_arand"], value["mean_voi"]))


def _with_baseline(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    primary_sample: str,
) -> dict[str, Any]:
    result = dict(candidate)
    if baseline is None:
        result["strict_crop_gate"] = None
        return result
    comparisons = {
        "mean_arand": candidate["mean_arand"] / baseline["mean_arand"] - 1.0,
        "mean_voi": candidate["mean_voi"] / baseline["mean_voi"] - 1.0,
        "balance": candidate["balance"] / baseline["balance"] - 1.0,
        "m08_arand": candidate["samples"][primary_sample]["arand"] / baseline["samples"][primary_sample]["arand"] - 1.0,
        "m08_voi": candidate["samples"][primary_sample]["voi"] / baseline["samples"][primary_sample]["voi"] - 1.0,
    }
    result["vs_baseline_pct"] = {name: 100.0 * value for name, value in comparisons.items()}
    result["strict_crop_gate"] = (
        candidate["mean_arand"] <= baseline["mean_arand"]
        and candidate["mean_voi"] <= baseline["mean_voi"]
        and candidate["samples"][primary_sample]["arand"] <= baseline["samples"][primary_sample]["arand"]
        and candidate["samples"][primary_sample]["voi"] <= baseline["samples"][primary_sample]["voi"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--expected-rows", type=int, default=80)
    parser.add_argument("--backend", default="waterz")
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--baseline-bias", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--baseline-temperature", type=float, default=1.15)
    parser.add_argument("--baseline-threshold", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    samples = tuple(str(value) for value in args.samples)
    if len(samples) != 2:
        raise ValueError("paired MEC crop analysis requires exactly two samples")
    candidates, integrity = _load_candidates(
        args.metrics,
        samples=samples,
        expected_rows=int(args.expected_rows),
        backend=str(args.backend),
    )
    baseline = None
    baseline_integrity = None
    if args.baseline_metrics is not None:
        baseline_candidates, baseline_integrity = _load_candidates(
            args.baseline_metrics,
            samples=samples,
            expected_rows=int(args.expected_rows),
            backend=str(args.backend),
        )
        baseline_key = (
            round(float(args.baseline_bias[0]), 8),
            round(float(args.baseline_bias[1]), 8),
            round(float(args.baseline_bias[2]), 8),
            round(float(args.baseline_temperature), 8),
            round(float(args.baseline_threshold), 8),
        )
        if baseline_key not in baseline_candidates:
            raise ValueError(f"baseline configuration is absent from {args.baseline_metrics}: {baseline_key}")
        baseline = baseline_candidates[baseline_key]

    result = {
        "metrics": str(args.metrics),
        "integrity": integrity,
        "baseline_metrics": None if args.baseline_metrics is None else str(args.baseline_metrics),
        "baseline_integrity": baseline_integrity,
        "baseline": None if baseline is None else _with_baseline(baseline, baseline, primary_sample=samples[0]),
        "pareto": [_with_baseline(value, baseline, primary_sample=samples[0]) for value in _pareto(candidates)],
        "best_balance": _with_baseline(
            min(candidates.values(), key=lambda value: (value["balance"], value["worst_balance"])),
            baseline,
            primary_sample=samples[0],
        ),
        "strict_crop_gate_passes": [
            _with_baseline(value, baseline, primary_sample=samples[0])
            for value in candidates.values()
            if baseline is not None and _with_baseline(value, baseline, primary_sample=samples[0])["strict_crop_gate"]
        ],
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
