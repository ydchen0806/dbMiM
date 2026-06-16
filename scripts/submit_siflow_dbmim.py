#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path


PROJECT = Path("/volume/med-train/users/dchen02/code/dbMiM")
HELPER = Path("/volume/med-train/users/dchen02/.codex/skills/yinda-public-skill/scripts/submit_tos_bootstrap_job.py")
PY = Path("/volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python")


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def make_bundle(entrypoint: str) -> Path:
    out = PROJECT / "outputs" / "siflow_bundles" / f"dbmim_bundle_{stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["dbmim", "configs", "train_pretrain.py", "train_finetune.py", "requirements-dbMIM.txt"]:
        src = PROJECT / name
        dst = out / name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    (out / "run.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd \"$(dirname \"$0\")\"",
                "python - <<'PY'",
                "import importlib.util",
                "missing=[m for m in ['torch','yaml','h5py','PIL','numpy'] if importlib.util.find_spec(m) is None]",
                "print({'missing_python_modules': missing})",
                "if missing:",
                "    raise SystemExit('missing required python modules: '+','.join(missing))",
                "PY",
                entrypoint,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit dbMiM training to SiFlow through TOS bootstrap")
    parser.add_argument("--stage", choices=["pretrain", "finetune", "pretrain-cremi", "finetune-cremi", "smoke"], default="smoke")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resource-pool", default="med-model")
    parser.add_argument("--gpus-per-pod", type=int, default=8)
    args = parser.parse_args()

    if args.stage == "pretrain":
        entrypoint = "python -m torch.distributed.run --nproc_per_node=8 train_pretrain.py --config configs/pretrain_fafb.yaml"
        prefix = "dbmim-pretrain"
    elif args.stage == "pretrain-cremi":
        entrypoint = "python -m torch.distributed.run --nproc_per_node=8 train_pretrain.py --config configs/pretrain_cremi_real.yaml"
        prefix = "dbmim-pretrain-cremi"
    elif args.stage == "finetune-cremi":
        entrypoint = "python -m torch.distributed.run --nproc_per_node=8 train_finetune.py --config configs/finetune_cremi_real.yaml"
        prefix = "dbmim-finetune-cremi"
    elif args.stage == "finetune":
        entrypoint = "python -m torch.distributed.run --nproc_per_node=8 train_finetune.py --config configs/finetune_cremi.yaml"
        prefix = "dbmim-finetune"
    else:
        entrypoint = "python train_pretrain.py --config configs/pretrain_smoke.yaml && python train_finetune.py --config configs/finetune_smoke.yaml"
        prefix = "dbmim-smoke"

    bundle = make_bundle(entrypoint)
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
    ]
    if args.submit:
        cmd.append("--submit")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
