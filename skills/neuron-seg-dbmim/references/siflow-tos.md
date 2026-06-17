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
