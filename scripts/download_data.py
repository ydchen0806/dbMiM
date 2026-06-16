from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


CREMI_URLS = {
    "sample_A_20160501.hdf": "https://cremi.org/static/data/sample_A_20160501.hdf",
    "sample_B_20160501.hdf": "https://cremi.org/static/data/sample_B_20160501.hdf",
    "sample_C_20160501.hdf": "https://cremi.org/static/data/sample_C_20160501.hdf",
}


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
        env.pop(key, None)
    subprocess.run(cmd, check=True, env=env)


def download_cremi(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, url in CREMI_URLS.items():
        out = target / name
        if not out.exists():
            run(["curl", "-L", "--fail", "--continue-at", "-", "--output", str(out), url])
        entries.append({"name": name, "url": url, "path": str(out), "bytes": out.stat().st_size})
    (target / "cremi_manifest.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(entries, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or snapshot dbMiM data assets")
    parser.add_argument("--target-dir", default="/volume/med-train/users/dchen02/code/dbMiM/data")
    parser.add_argument("--hf-repo", default="cyd0806/EM_pretrain_data")
    parser.add_argument("--hf-pattern", action="append", default=None,
                        help="Hugging Face allow-pattern. Can be repeated.")
    parser.add_argument("--hf-subdir", default="FAFB_hdf",
                        help="Legacy subdir pattern; used only when --hf-pattern is omitted.")
    parser.add_argument("--cremi", action="store_true", help="Download CREMI sample A/B/C cropped HDF5 files.")
    parser.add_argument("--allow-large-download", action="store_true")
    args = parser.parse_args()

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)
    if args.cremi:
        download_cremi(target / "CREMI")
        return

    patterns = args.hf_pattern or [f"{args.hf_subdir}/**"]
    manifest = {
        "hf_repo": args.hf_repo,
        "hf_patterns": patterns,
        "target_dir": str(target),
        "status": "not_started",
        "notes": [
            "The FAFB pretraining data is large and may require gated Hugging Face access.",
            "Set HF_TOKEN or login with huggingface-cli before running the full download.",
        ],
    }
    if not args.allow_large_download:
        manifest["status"] = "dry_run"
        (target / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN/HUGGINGFACE_HUB_TOKEN is required for the gated/large HF download.")

    from huggingface_hub import snapshot_download

    local_dir = target / "EM_pretrain_data"
    snapshot_download(
        repo_id=args.hf_repo,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    manifest["status"] = "downloaded"
    manifest["local_dir"] = str(local_dir)
    (target / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
