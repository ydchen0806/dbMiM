# SiFlow and TOS Reference

Use this with the global `yinda-public-skill` instructions. This file records
dbMiM-specific operational pitfalls.

## Default Execution Pattern

Shanghai changliu pods may not see local repo paths and may not have public
internet. Use TOS bootstrap:

1. Package the minimal runnable repo bundle.
2. Upload the bundle to TOS.
3. Presign the bundle.
4. Submit a SiFlow command that downloads/extracts the bundle inside the pod.
5. Upload outputs back to TOS.

Project launcher:

```bash
/volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python \
  scripts/submit_siflow_dbmim.py \
  --stage <stage> \
  --resource-pool <pool-or-auto> \
  --gpus-per-pod 8 \
  --submit
```

Do not print secret env files or TOS config contents. Load them only through the
existing secret files.

## Resource Pool Pitfall

`--resource-pool auto` may choose
`cn-shanghai-changliu-skyinfer-reserved-shared` instead of `med-model`.

Before reporting which pool was used, inspect the saved JSON under:

```text
/volume/med-train/users/dchen02/siflow_submissions/yinda_public_skill/
```

Fields to check:

- `resource_pool`
- `resource_selection.choice`
- `resource_selection.evaluated`
- `uuid`
- `gpus_per_pod`
- `instance_name`

If the user wants a specific pool, pass it explicitly rather than using auto.

## Watchers

Evaluation watchers poll TOS for checkpoints and submit 1-GPU eval jobs once a
checkpoint appears.

Typical logs:

```text
outputs/watchers/eval_aniso_pretrained_*.log
outputs/watchers/eval_ablation_*_*.log
```

Interpretation:

- `checkpoint_wait`: no checkpoint found yet.
- `checkpoint_ready`: watcher saw a checkpoint and is about to submit eval.
- "wrote ...json": eval submission record exists; inspect it for UUID/pool.

Watcher processes do not use GPUs. They are login-node polling processes.

## Log Access

SiFlow SDK status/log APIs can timeout. If they do:

- do not keep retrying indefinitely;
- state that SDK log/status query timed out;
- fall back to watcher logs and TOS checkpoint probes;
- never invent loss values.

Useful status checks:

```bash
cd /volume/med-train/users/dchen02/code/dbMiM
date -u
for f in outputs/watchers/*.log; do
  printf '%s\t' "$(basename "$f")"
  tail -n 3 "$f" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g'
  printf '\n'
done
```

Use `timeout` around TOS probes because `tosutil stat/ls` can hang:

```bash
timeout 45 /volume/med-train/users/dchen02/bin/tosutil stat \
  tos://agi-data/users/dchen02/dbmim/outputs/<prefix>/finetuned_latest.pt \
  -conf=/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf
```

## Output Prefixes

Key TOS locations:

```text
tos://agi-data/users/dchen02/dbmim/assets/cremi_abc_20160501.tar.gz
tos://agi-data/users/dchen02/dbmim/bundles/
tos://agi-data/users/dchen02/dbmim/outputs/
```

Main checkpoint names:

- pretrain: `pretrained_latest.pt`
- finetune: `finetuned_best.pt`, `finetuned_latest.pt`
- eval: `cremi_segmentation_summary.json`,
  `cremi_segmentation_records.json`, `cremi_segmentation_metrics.csv`

## Better Heartbeats

For long jobs, prefer adding a lightweight rank-0 heartbeat or periodic upload
of `train_log.jsonl` / `finetune_log.jsonl` to TOS. Waiting until checkpoint
save makes it impossible to inspect early loss when SDK logs are slow.

As of the R2 restart, `scripts/submit_siflow_dbmim.py` wraps training stages
with a shell sync loop. While the foreground training command runs, the loop
periodically uploads:

- `finetuned_latest.pt`
- `finetuned_best.pt`
- `pretrained_latest.pt`
- `finetune_log.jsonl`
- `pretrain_log.jsonl`

The full output directory is uploaded once more after the training command
exits. Watchers should therefore see checkpoints during training instead of
waiting for process exit. If watchers still show only `checkpoint_wait`, check
whether the job started, whether the output directory name matches the TOS
prefix, and whether the training process reached its first `save_every` epoch.

SiFlow SDK task stop/status calls can hang in network/proxy setup. For
submission commands, use the project submitter or `submit_tos_bootstrap_job.py`
with `--direct-network`; for ad hoc SDK scripts, unset proxy variables before
creating the client.

## GPU Accounting

When asked how many GPUs are occupied, deduplicate by SiFlow UUID. Querying
`tasks.list` separately for `med-model`, `cpt-train`, and shared pools can
return the same running task multiple times.

Use this pattern:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
source /volume/med-train/users/dchen02/secrets/siflow_env_dchen02.sh
/volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python - <<'PY'
from siflow import SiFlow
client = SiFlow(region="cn-shanghai", cluster="changliu", timeout=120)
seen = {}
for status in ["Running", "Queueing", "Pending"]:
    tasks = client.tasks.list(count=200, status=status, owners="dchen02", timeout=120)
    for task in tasks:
        d = task if isinstance(task, dict) else getattr(task, "__dict__", {})
        uuid = str(d.get("uuid") or getattr(task, "uuid", ""))
        if not uuid or uuid in seen:
            continue
        name = str(d.get("name") or getattr(task, "name", ""))
        instances = d.get("instances") or getattr(task, "instances", []) or []
        gpus = 0
        if instances:
            first = instances[0] if isinstance(instances[0], dict) else getattr(instances[0], "__dict__", {})
            gpus = first.get("count_per_pod") or first.get("count") or 0
        seen[uuid] = (status, name, int(gpus or 0))
print("total_gpus", sum(row[2] for row in seen.values()))
print("dbmim_gpus", sum(row[2] for row in seen.values() if "dbmim" in row[1].lower()))
for uuid, row in seen.items():
    print(uuid, row)
PY
```

As of 2026-06-21 15:20 UTC, dchen02 had 36 unique running GPUs: 12 for dbMiM
full R14 finetuning and 24 for unrelated STEM data-transfer packs.

## EM Pretraining Data Pitfall

`data/EM_pretrain_data/*_manifest.json` and
`reports/em_pretrain_data_manifests/*.json` are manifests only. They do not
prove that all-EM HDF5 volumes are present.

Before claiming all-EM pretraining, verify the TOS prefix contains actual
`.h5/.hdf/.hdf5` files:

```bash
/volume/med-train/users/dchen02/bin/tosutil ls \
  tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data/all \
  -conf=/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf
```

The 2026-06-21 MA-dbMiM pretrain initially failed because the `all` prefix
listed successfully but contained no HDF5 files. The submitter now falls back
to CREMI-only and prints
`em_pretrain_data_status=missing_offline_tos_fallback_to_cremi_only`. Any
checkpoint from that fallback must be described as CREMI-only MA-dbMiM, not
all-EM pretraining.

## Polling Results

`scripts/poll_dbmim_tos_results.py` has SiFlow stdout fallback support. It must
load the SiFlow env in the subprocess and clear proxy variables; otherwise the
fallback may fail even when direct interactive `query_logs` works.

When a summary is reconstructed from stdout, inspect:

- `source`
- `num_records`
- `sample_names`
- `best_by_voi_sum.n`
- `best_by_adapted_rand.n`

Use TOS-native `cremi_segmentation_summary.json` when available, but stdout
fallback is useful while a long waterz eval is still streaming rows.
