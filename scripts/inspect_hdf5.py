#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def visit(handle: h5py.File) -> list[dict]:
    rows: list[dict] = []

    def cb(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append(
                {
                    "key": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                }
            )

    handle.visititems(cb)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HDF5 schema")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    report = []
    for path_s in args.paths:
        path = Path(path_s)
        with h5py.File(path, "r") as handle:
            report.append({"path": str(path), "datasets": visit(handle)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
