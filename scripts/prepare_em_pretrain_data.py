#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_REPO = "cyd0806/EM_pretrain_data"
DEFAULT_TOS_PREFIX = "tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data"
GROUP_PREFIXES = {
    "fafb": ("FAFB_crop_hdf_",),
    "fib25": ("FIB-25_hdf_",),
    "kasthuri": ("Kasthuri2015_hdf_",),
    "mitoem": ("MitoEM_",),
    "mb_moc": ("mb_moc_hdf_",),
    "all": ("FAFB_crop_hdf_", "FIB-25_hdf_", "Kasthuri2015_hdf_", "MitoEM_", "mb_moc_hdf_"),
}
FALLBACK_FILES = [
    ("FAFB_crop_hdf_1.zip", 16887800847),
    ("FAFB_crop_hdf_2.zip", 16827138593),
    ("FAFB_crop_hdf_3.zip", 16840829557),
    ("FAFB_crop_hdf_4.zip", 16868591313),
    ("FAFB_crop_hdf_5.zip", 16855894151),
    ("FAFB_crop_hdf_6.zip", 16845546924),
    ("FAFB_crop_hdf_7.zip", 16257694648),
    ("FIB-25_hdf_1.zip", 12296614433),
    ("FIB-25_hdf_2.zip", 12690080064),
    ("FIB-25_hdf_3.zip", 11926458495),
    ("FIB-25_hdf_4.zip", 12670005980),
    ("FIB-25_hdf_5.zip", 12536349785),
    ("FIB-25_hdf_6.zip", 12099256009),
    ("FIB-25_hdf_7.zip", 12675509173),
    ("FIB-25_hdf_8.zip", 13004429622),
    ("FIB-25_hdf_9.zip", 12842923472),
    ("FIB-25_hdf_10.zip", 1374844970),
    ("Kasthuri2015_hdf_1.zip", 14254692015),
    ("Kasthuri2015_hdf_2.zip", 13267091354),
    ("Kasthuri2015_hdf_3.zip", 11967441398),
    ("Kasthuri2015_hdf_4.zip", 13048414846),
    ("Kasthuri2015_hdf_5.zip", 12279518708),
    ("Kasthuri2015_hdf_6.zip", 12909935603),
    ("Kasthuri2015_hdf_7.zip", 12559968953),
    ("Kasthuri2015_hdf_8.zip", 13263431964),
    ("Kasthuri2015_hdf_9.zip", 5810708935),
    ("MitoEM_human_hdf_1.zip", 8208187067),
    ("MitoEM_rat_hdf_1.zip", 8450193355),
    ("mb_moc_hdf_1.zip", 21263553957),
    ("mb_moc_hdf_2.zip", 21225116725),
    ("mb_moc_hdf_3.zip", 21222686110),
    ("mb_moc_hdf_4.zip", 21206915418),
    ("mb_moc_hdf_5.zip", 21221284794),
    ("mb_moc_hdf_6.zip", 21224882846),
    ("mb_moc_hdf_7.zip", 959308184),
]


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    clean_env = os.environ.copy()
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
        clean_env.pop(key, None)
    if env:
        clean_env.update(env)
    subprocess.run(cmd, check=True, env=clean_env)


def hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""


def list_hf_repo(repo: str, token: str = "") -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=1&expand=1"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fallback_hf_files() -> list[dict]:
    return [{"path": path, "type": "file", "size": size, "source": "fallback"} for path, size in FALLBACK_FILES]


def select_files(files: list[dict], group: str) -> list[dict]:
    prefixes = GROUP_PREFIXES[group]
    selected = [
        item
        for item in files
        if item.get("type") == "file"
        and str(item.get("path", "")).endswith(".zip")
        and str(item.get("path", "")).startswith(prefixes)
    ]
    return sorted(selected, key=lambda item: item["path"])


def resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def download_file(repo: str, item: dict, zip_dir: Path, token: str) -> Path:
    zip_dir.mkdir(parents=True, exist_ok=True)
    path = str(item["path"])
    out = zip_dir / path
    expected = int(item.get("size") or 0)
    if out.exists() and expected > 0 and out.stat().st_size == expected:
        return out
    tmp = out.with_suffix(out.suffix + ".partial")
    header = f"Authorization: Bearer {token}"
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "20",
        "--retry-delay",
        "10",
        "--continue-at",
        "-",
        "-H",
        header,
        "--output",
        str(tmp),
        resolve_url(repo, path),
    ]
    run(cmd)
    if expected > 0 and tmp.stat().st_size != expected:
        raise RuntimeError(f"downloaded size mismatch for {path}: {tmp.stat().st_size} != {expected}")
    tmp.replace(out)
    return out


def extract_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / f".extracted_{zip_path.name}.json"
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        existing = [extract_dir / name for name in payload.get("members", [])]
        if existing and all(path.exists() for path in existing):
            return existing
    before = {p.resolve() for p in extract_dir.rglob("*") if p.is_file()}
    with zipfile.ZipFile(zip_path) as handle:
        handle.extractall(extract_dir)
        members = [name for name in handle.namelist() if not name.endswith("/")]
    after = [p for p in extract_dir.rglob("*") if p.is_file() and p.resolve() not in before]
    if not after:
        after = [extract_dir / name for name in members if (extract_dir / name).exists()]
    marker.write_text(
        json.dumps(
            {
                "zip": str(zip_path),
                "members": [str(path.relative_to(extract_dir)) for path in after],
                "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return after


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def upload_to_tos(local_dir: Path, tos_prefix: str, tosutil: Path, tos_conf: Path, group: str) -> None:
    if not tosutil.exists():
        raise FileNotFoundError(f"tosutil not found: {tosutil}")
    if not tos_conf.exists():
        raise FileNotFoundError(f"tosutil config not found: {tos_conf}")
    run(
        [
            str(tosutil),
            "cp",
            str(local_dir),
            f"{tos_prefix}/{group}",
            "-r",
            f"-conf={tos_conf}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare gated HF EM pretraining data for offline dbMiM runs.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--group", choices=sorted(GROUP_PREFIXES), default="fafb")
    parser.add_argument("--target-dir", default="/volume/med-train/users/dchen02/code/dbMiM/data/EM_pretrain_data")
    parser.add_argument("--tos-prefix", default=DEFAULT_TOS_PREFIX)
    parser.add_argument("--tosutil", default="/volume/med-train/users/dchen02/bin/tosutil")
    parser.add_argument("--tos-conf", default="/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--upload-tos", action="store_true")
    parser.add_argument("--keep-zips", action="store_true")
    args = parser.parse_args()

    token = hf_token()
    try:
        files = list_hf_repo(args.repo, token=token)
        manifest_source = "hf_api"
    except Exception as exc:
        files = fallback_hf_files()
        manifest_source = f"fallback:{type(exc).__name__}"
    selected = select_files(files, args.group)
    total_size = sum(int(item.get("size") or 0) for item in selected)
    target = Path(args.target_dir)
    group_dir = target / args.group
    zip_dir = target / "_zips" / args.group
    manifest = {
        "repo": args.repo,
        "group": args.group,
        "selected_files": selected,
        "manifest_source": manifest_source,
        "total_bytes": total_size,
        "requires_hf_token": True,
        "has_hf_token": bool(token),
        "target_dir": str(group_dir),
        "tos_prefix": f"{args.tos_prefix}/{args.group}",
        "status": "manifest",
    }
    write_manifest(target / f"{args.group}_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["repo", "group", "total_bytes", "has_hf_token", "target_dir", "tos_prefix"]}, indent=2))

    if args.manifest_only or not (args.download or args.extract or args.upload_tos):
        return
    if not token and args.download:
        raise SystemExit("HF_TOKEN/HUGGINGFACE_HUB_TOKEN is required to download gated EM pretraining data.")

    downloaded: list[str] = []
    extracted: list[str] = []
    if args.download:
        for item in selected:
            zip_path = download_file(args.repo, item, zip_dir, token)
            downloaded.append(str(zip_path))
            if args.extract:
                extracted.extend(str(path) for path in extract_zip(zip_path, group_dir))
                if not args.keep_zips:
                    zip_path.unlink(missing_ok=True)
        if not args.keep_zips and zip_dir.exists() and not any(zip_dir.iterdir()):
            shutil.rmtree(zip_dir)
    elif args.extract:
        for zip_path in sorted(zip_dir.glob("*.zip")):
            extracted.extend(str(path) for path in extract_zip(zip_path, group_dir))

    manifest.update(
        {
            "status": "prepared",
            "downloaded": downloaded,
            "extracted_count": len(extracted),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    write_manifest(target / f"{args.group}_manifest.json", manifest)
    if args.upload_tos:
        upload_to_tos(group_dir, args.tos_prefix, Path(args.tosutil), Path(args.tos_conf), args.group)
        upload_to_tos(
            target / f"{args.group}_manifest.json",
            args.tos_prefix,
            Path(args.tosutil),
            Path(args.tos_conf),
            args.group,
        )


if __name__ == "__main__":
    main()
