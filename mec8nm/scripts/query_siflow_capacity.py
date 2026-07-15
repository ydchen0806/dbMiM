#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HELPER = Path(
    "/volume/med-train/users/dchen02/.codex/skills/yinda-public-skill/scripts/submit_tos_bootstrap_job.py"
)


def load_helper():
    spec = importlib.util.spec_from_file_location("yinda_submit_tos_bootstrap", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load yinda-public-skill helper: {HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_rows(client, helper, *, status: str, timeout: float) -> list[dict]:
    try:
        response = client.tasks.list(count=100, status=status, owners="dchen02", timeout=timeout)
    except TypeError:
        response = client.tasks.list(count=100, status=status, owners="dchen02")
    rows = response if isinstance(response, list) else helper.rows_from_response(response)
    return [helper.object_to_dict(row) for row in rows]


def quota_rows(client, helper, *, timeout: float) -> list[dict]:
    try:
        response = client.instances.list_quotas(
            region="cn-shanghai",
            cluster="changliu",
            timeout=timeout,
        )
    except TypeError:
        response = client.instances.list_quotas(region="cn-shanghai", cluster="changliu")
    return helper.rows_from_response(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitized MEC SiFlow quota and active-task snapshot")
    parser.add_argument("--task-prefix", default="mec8nm")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    helper = load_helper()
    helper.clear_proxy_env()
    helper.load_env_file(helper.SIFLOW_ENV)

    from siflow import SiFlow

    client = SiFlow(region="cn-shanghai", cluster="changliu", timeout=args.timeout)
    pools = ["med-model", helper.SHANGHAI_SHARED_POOL]
    rows = quota_rows(client, helper, timeout=args.timeout)
    quotas = {}
    for pool in pools:
        match = next(
            (
                row
                for row in rows
                if str(row.get("resourcePoolName", "")) == pool
                and str(row.get("instanceName", "")) == "sci.g21-3"
            ),
            None,
        )
        quotas[pool] = (
            {
                "quota_seen": True,
                "avail": int(match.get("avail", -1)),
                "quota": int(match.get("quota", 0)),
                "used": int(match.get("used", 0)),
                "max_count_per_node": int(match.get("maxCountPerNode", 0)),
                "resource_pool_type": str(match.get("resourcePoolType", "")),
                "resource_pool_isolation": str(match.get("resourcePoolIsolation", "")),
            }
            if match is not None
            else {"quota_seen": False, "avail": -1, "quota": 0, "used": 0}
        )

    active = []
    for status in ("Running", "Queueing", "Pending"):
        for row in task_rows(client, helper, status=status, timeout=args.timeout):
            name = str(row.get("name", row.get("taskName", row.get("task_name", ""))))
            if args.task_prefix and args.task_prefix.lower() not in name.lower():
                continue
            active.append(
                {
                    "uuid": str(row.get("uuid", row.get("id", ""))),
                    "name": name,
                    "status": str(row.get("status", status)),
                    "resource_pool": str(
                        row.get("resourcePool", row.get("resource_pool", row.get("resourcePoolName", "")))
                    ),
                    "status_msg": str(row.get("statusMsg", row.get("status_msg", ""))),
                }
            )

    print(
        json.dumps(
            {
                "region": "cn-shanghai",
                "cluster": "changliu",
                "instance": "sci.g21-3",
                "quotas": quotas,
                "active_matching_tasks": active,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
