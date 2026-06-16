from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or snapshot dbMiM data assets")
    parser.add_argument("--target-dir", default="/volume/med-train/users/dchen02/code/dbMiM/data")
    parser.add_argument("--hf-repo", default="cyd0806/EM_pretrain_data")
    parser.add_argument("--hf-subdir", default="FAFB_hdf")
    parser.add_argument("--allow-large-download", action="store_true")
    args = parser.parse_args()

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "hf_repo": args.hf_repo,
        "hf_subdir": args.hf_subdir,
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
        allow_patterns=[f"{args.hf_subdir}/**"],
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
