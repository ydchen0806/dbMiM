#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from PIL import Image, ImageSequence


PUBLIC_DATASETS = {
    "isbi2012_train": {
        "url": "https://raw.githubusercontent.com/alexklibisz/isbi-2012/master/data/train-volume.tif",
        "archive": "isbi_train-volume.tif",
        "inner": "",
        "output": "isbi2012_train_raw.h5",
        "source": "ISBI 2012 ssTEM train volume",
    },
    "isbi2012_test": {
        "url": "https://raw.githubusercontent.com/alexklibisz/isbi-2012/master/data/test-volume.tif",
        "archive": "isbi_test-volume.tif",
        "inner": "",
        "output": "isbi2012_test_raw.h5",
        "source": "ISBI 2012 ssTEM test volume",
    },
    "snemi3d_train": {
        "url": "https://zenodo.org/api/records/7142003/files/snemi.zip/content",
        "archive": "snemi.zip",
        "inner": "image/train-input.tif",
        "output": "snemi3d_train_raw.h5",
        "source": "SNEMI3D train-input volume",
    },
    "snemi3d_test": {
        "url": "https://zenodo.org/api/records/7142003/files/snemi.zip/content",
        "archive": "snemi.zip",
        "inner": "image/test-input.tif",
        "output": "snemi3d_test_raw.h5",
        "source": "SNEMI3D test-input volume",
    },
}


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
        env.pop(key, None)
    subprocess.run(cmd, check=True, env=env)


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    tmp = out.with_suffix(out.suffix + ".partial")
    run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "10",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(tmp),
            url,
        ]
    )
    tmp.replace(out)


def read_multipage_tiff(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        frames = [np.asarray(frame.convert("L"), dtype=np.uint8) for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise ValueError(f"no frames found in {path}")
    return np.stack(frames, axis=0)


def convert_tiff_to_h5(tiff_path: Path, out_path: Path, *, source: str) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    volume = read_multipage_tiff(tiff_path)
    with h5py.File(out_path, "w") as handle:
        dset = handle.create_dataset("raw", data=volume, compression="gzip", compression_opts=4, chunks=True)
        dset.attrs["source"] = source
        dset.attrs["axis_order"] = "zyx"
    return {
        "path": str(out_path),
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "bytes": out_path.stat().st_size,
        "source": source,
    }


def prepare_one(name: str, spec: dict, zip_dir: Path, out_dir: Path) -> dict:
    archive = zip_dir / str(spec["archive"])
    download(str(spec["url"]), archive)
    if spec.get("inner"):
        with tempfile.TemporaryDirectory(prefix=f"{name}_") as tmp:
            tmp_path = Path(tmp)
            with zipfile.ZipFile(archive) as zf:
                zf.extract(str(spec["inner"]), tmp_path)
            tiff_path = tmp_path / str(spec["inner"])
            return convert_tiff_to_h5(tiff_path, out_dir / str(spec["output"]), source=str(spec["source"]))
    return convert_tiff_to_h5(archive, out_dir / str(spec["output"]), source=str(spec["source"]))


def upload_to_tos(local_dir: Path, tos_prefix: str, tosutil: Path, tos_conf: Path) -> None:
    if not tosutil.exists():
        raise FileNotFoundError(f"tosutil not found: {tosutil}")
    if not tos_conf.exists():
        raise FileNotFoundError(f"TOS config not found: {tos_conf}")
    run([str(tosutil), "cp", str(local_dir), tos_prefix, "-r", f"-conf={tos_conf}"])


def upload_file_to_tos(local_file: Path, tos_prefix: str, tosutil: Path, tos_conf: Path) -> None:
    if not tosutil.exists():
        raise FileNotFoundError(f"tosutil not found: {tosutil}")
    if not tos_conf.exists():
        raise FileNotFoundError(f"TOS config not found: {tos_conf}")
    run([str(tosutil), "cp", str(local_file), tos_prefix, f"-conf={tos_conf}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public EM pretraining HDF5 assets for dbMiM.")
    parser.add_argument("--target-dir", default="/volume/med-train/users/dchen02/code/dbMiM/data/EM_pretrain_data")
    parser.add_argument("--group", default="public_em")
    parser.add_argument("--dataset", action="append", choices=sorted(PUBLIC_DATASETS), default=None)
    parser.add_argument("--upload-tos", action="store_true")
    parser.add_argument("--tos-prefix", default="tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data")
    parser.add_argument("--tosutil", default="/volume/med-train/users/dchen02/bin/tosutil")
    parser.add_argument("--tos-conf", default="/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf")
    args = parser.parse_args()

    target = Path(args.target_dir)
    zip_dir = target / "public_zips"
    out_dir = target / args.group
    selected: Iterable[str] = args.dataset or ["isbi2012_train", "isbi2012_test", "snemi3d_train", "snemi3d_test"]
    records = []
    for name in selected:
        records.append({"name": name, **prepare_one(name, PUBLIC_DATASETS[name], zip_dir, out_dir)})
    manifest = {
        "group": args.group,
        "datasets": records,
        "tos_prefix": f"{args.tos_prefix}/{args.group}",
        "status": "prepared",
    }
    manifest_path = target / f"{args.group}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.upload_tos:
        upload_to_tos(out_dir, f"{args.tos_prefix}/{args.group}", Path(args.tosutil), Path(args.tos_conf))
        upload_file_to_tos(manifest_path, f"{args.tos_prefix}/manifests/{manifest_path.name}", Path(args.tosutil), Path(args.tos_conf))
        manifest["status"] = "uploaded"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
