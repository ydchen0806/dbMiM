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

When a watcher probes a numeric threshold with a Python helper under
`set -euo pipefail`, wrap the helper in `set +e` / `set -e` and capture its
status explicitly. Otherwise a normal "checkpoint exists but max_step is still
below threshold" return code of 1 can terminate the whole watcher before it
writes `checkpoint_wait` or sleeps. This happened while adapting the R20
finetune watcher for R21 decoder-aware pretraining on 2026-06-23.

If a watcher needs a second independent submission wave, make marker paths
configurable and set unique values. Reusing the R20 default marker
`outputs/watchers/full-em-r20-finetune.submitted` will make a later R21 watcher
exit as `already_submitted`.

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

The same proxy issue affects `tosutil`. If TOS `cp/ls/stat` hangs for an object
that should exist, retry with all proxy variables removed:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  timeout 120 /volume/med-train/users/dchen02/bin/tosutil cp \
  tos://agi-data/users/dchen02/dbmim/outputs/<prefix>/<file> \
  /tmp/<file> -f -bt=fns \
  -conf=/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf
```

On 2026-06-22 this was required to verify
`pretrain_public_em_membrane_dbmim_r16/pretrained_latest.pt`; with proxies set,
both watcher and manual `cp` timed out, while no-proxy `cp` downloaded the
40.20 MB checkpoint successfully.

The same rule applies to GitHub. This login environment can expose
`HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` values pointing at
`192.168.32.28:18000`; with those set, `git push` to GitHub may hang until a
manual `timeout` kills it. Use a no-proxy wrapper and avoid printing or
embedding tokens:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  GIT_TERMINAL_PROMPT=0 \
  git -c http.version=HTTP/1.1 push origin main
```

Observed behavior on 2026-06-23:

- Earlier no-proxy pushes succeeded for commits through `bb8ee57`.
- Later pushes of the local learned-RAG commit `92576d6` timed out with
  `rc=124` and no stderr even after clearing proxies.
- No-proxy `git ls-remote --heads origin main` returned immediately, and
  `curl -I https://github.com/ydchen0806/dbMiM.git` reached GitHub. The remote
  `main` was still `bb8ee57`, while local `main` had two unpushed commits:
  `92576d6` and `0d3a1b4`.
- Disabling credentials/askpass made push fail immediately with
  `fatal: could not read Username for 'https://github.com': terminal prompts
  disabled`. The shell had `GIT_ASKPASS` pointing at a VS Code askpass script,
  but lacked the required `VSCODE_GIT_ASKPASS_NODE`,
  `VSCODE_GIT_ASKPASS_MAIN`, and `VSCODE_GIT_IPC_HANDLE` environment. There
  was no SSH private key, no configured credential helper, no `.netrc`, and no
  `.git-credentials`.
- Therefore this specific push failure is best explained as non-interactive
  GitHub credentials/askpass being unavailable or hanging, not a bad commit or
  a remote rejection. If it happens, report the unpushed local commit hash,
  keep SiFlow submissions based on the TOS bundle, and push later from an
  authenticated shell or a safe non-logged credential path.
- Do not add the user's GitHub token to `.git/config`, the remote URL, shell
  history, scripts, or skill files.

The yinda public submit helper truncates `name_prefix` to 35 characters before
calling SiFlow. If the first 35 characters end with `-`, SiFlow rejects the
task. `scripts/submit_siflow_dbmim.py` now shortens such arch-explore prefixes
locally; keep this behavior when adding long stage names.

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

## R15 Architecture-Explore Operations

The R15 architecture-explore wave uses 2-GPU `med-model` jobs so several
method variants can run concurrently on H200s without waiting for a full
8-GPU slot.

Active/important 2026-06-22 UUIDs:

| purpose | UUID | GPUs | pool |
|---|---|---:|---|
| long-range affinity + MA-dbMiM | `73a40fd4-287b-4bfc-98b6-c57aa6a38c1a` | 2 | med-model |
| long-range affinity + LSD auxiliary | `08fad37f-4257-4f4f-9e9b-ad86c4b7f93f` | 2 | med-model |
| long-range affinity + stronger BCAR | `cb58f241-4fac-490f-b958-1ca6376bfb14` | 2 | med-model |
| standalone postprocess sweep, first invalid attempt | `bcac3b16-9896-4114-84bd-a70f854e2a8e` | 2 | med-model |
| standalone postprocess sweep, dependency-fixed rerun | `8ffeb749-2da8-4236-a46d-b60e9443598e` | 2 | med-model |

Submit command pattern:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
source /volume/med-train/users/dchen02/secrets/siflow_env_dchen02.sh
/volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python \
  scripts/submit_siflow_dbmim.py \
  --stage finetune-cremi-unetr-aniso-arch-explore-longaff-mempretrained-r15q \
  --resource-pool med-model \
  --gpus-per-pod 2 \
  --post-train-arch-bench \
  --submit
```

Change only the stage name for the LSD and BCAR2 variants.

The first standalone postprocess sweep exposed an important packaging pitfall:
`eval-cremi-arch-explore-postprocess-r15q` used `--metric-backend skimage` but
was not included in the SuperHuman dependency-install path, so graph/RAG metric
rows failed with missing `skimage`. The submitter now treats that stage as a
SuperHuman-style eval dependency user and runs architecture sweeps with
`--fail-on-backend-error`. The invalid first attempt was stopped and the fixed
rerun is `8ffeb749-2da8-4236-a46d-b60e9443598e`; its startup logs showed
`waterz=True` and `cupy=True` in the package probe. If a future log reports
missing `skimage`, `waterz`, or `mahotas`, fix the bundle/install path and
resubmit; do not summarize the failed rows as performance.

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

As of 2026-06-22, the public non-gated bridge dataset is available at:

```text
tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data/public_em/
```

It contains converted HDF5 raw volumes from ISBI 2012 ssTEM and SNEMI3D. The
stage `pretrain-public-em-membrane-r16` copies only this `public_em` group and
must print `em_pretrain_data_status='available_offline_tos'` before training.
The active 8-GPU run is UUID `5a10fe9e-2d34-4568-8009-5902c73cc592`; output
prefix is:

```text
tos://agi-data/users/dchen02/dbmim/outputs/pretrain_public_em_membrane_dbmim_r16/
```

That run has now succeeded. `pretrained_latest.pt` was downloaded and inspected
on 2026-06-22: `global_step=160000`, `epoch=44`, and `max_steps=160000`.

The large HF dataset is accessible with the user-provided token, but the token
must stay outside git. It is stored only in:

```text
/volume/med-train/users/dchen02/secrets/hf_env_dchen02.sh
```

Load it with `source .../hf_env_dchen02.sh` before gated HF download commands.
Do not print the token, copy it into configs, or write it into a skill. The HF
access probe on 2026-06-22 succeeded for `cyd0806/EM_pretrain_data`: 37 files,
35 zip files, total about 485.84 GB.

As of 2026-06-22, the gated HF download, extraction, and TOS upload completed
for all five groups: `fafb`, `fib25`, `kasthuri`, `mitoem`, and `mb_moc`.
TOS contains actual HDF5 files for each group under:

```text
tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data/<group>/
```

The upload layout is nested (`<group>/<group>/...`) because `tosutil cp` copied
local group directories to same-named group prefixes. This is valid for the
current pretrain bundle because the pod-side gate searches `*/<group>/*` and the
dataset code recurses through HDF5 files.

Recommended gated download flow:

```bash
source /volume/med-train/users/dchen02/secrets/hf_env_dchen02.sh
python scripts/prepare_em_pretrain_data.py --group <fafb|fib25|kasthuri|mitoem|mb_moc|all> \
  --download --extract --upload-tos
```

This can download hundreds of GB; prefer one group at a time, verify free disk,
and upload to TOS before launching all-EM pretraining.

The 2026-06-22 gated download flow was automated by detached `screen`
sessions:

```bash
screen -ls
tail -f outputs/watchers/full_em_download_20260622T032208Z.log
tail -f outputs/watchers/full_em_pretrain_watch_20260622T032734Z.log
```

- `scripts/run_full_em_download_to_tos.sh` downloads/extracts/uploads the five
  gated groups serially. It intentionally uses `EM_GROUP_LIST`, not bash
  `GROUPS`; `GROUPS` is a readonly/special array in bash and expands to the
  current group id (`0` here), which previously caused a silent invalid group
  run.
- `scripts/watch_and_submit_full_em_pretrain.sh` polls TOS every 600 seconds
  and submits `pretrain-em-full-membrane-r20` on 8 `med-model` GPUs only after
  `fafb`, `fib25`, `kasthuri`, `mitoem`, and `mb_moc` each contain HDF5 files
  on TOS.
- `scripts/submit_siflow_dbmim.py` has a pod-side hard gate for
  `pretrain-em-full-membrane-r20`; it exits status 21 if any required gated
  group is missing after TOS copy. This prevents a fallback CREMI/public-only
  checkpoint from being mislabeled as full-EM pretraining.
- Ordinary `nohup ... &` background jobs have been observed to disappear in
  this execution environment. Use `screen -dmS ...` for multi-hour local
  download/watch tasks.

R20 full-EM pretraining has a specific runtime staging pitfall. Do not copy the
full HF corpus into the bundle temporary directory under `/tmp`: it is hundreds
of GB and can fill local node storage. The maintained submitter now stages it
under:

```text
/volume/med-train/users/dchen02/code/dbMiM_runtime/em_pretrain_data/full_r20/all
```

Latest failed R20 pretrain task:

- UUID: `ab50e050-a1c9-42ee-b40d-e4e43e109212`
- Pool/instance: `cn-shanghai/changliu`, `med-model`, `sci.g21-3`
- GPUs: 4
- Start time: `2026-06-22T09:02:49Z`
- End time: `2026-06-22T10:23:27Z`
- Failure mode: all five full-EM groups copied successfully from TOS, then the
  launch script exited before training because `set -o pipefail` propagated a
  SIGPIPE from the diagnostic `find ... | head -20` listing. The submitter now
  appends `|| true` to that diagnostic pipeline.

Do not count stopped R20 UUIDs as active: `1488e82a-301a-4b8a-961c-615657dd3491`
(old bundle), `1d0916f7-0c98-4d2f-8c02-3ec7cd47a46b` (8-GPU quota short), and
`a34bbacf-b483-4bb2-85cd-852adf9e8e16` (7-GPU fragmentation) were intentionally
stopped before meaningful training. Also do not count
`16b28f27-7e75-40d9-a5ec-04f3067b4001`; it was a post-patch 8-GPU retry that
was stopped immediately because `med-model` still reported only 7 available
instances for an 8-GPU request.

Current active R20 pretrain task:

- UUID: `7be2f62b-c1f3-482a-85a7-74cd63c63c35`
- Pool/instance: `cn-shanghai/changliu`, `med-model`, `sci.g21-3`
- GPUs: 4
- Start time: `2026-06-22T13:46:02Z`
- State at 2026-06-22 22:17 China time: Running and training. All five
  full-EM groups copied from TOS; `available_offline_tos` printed at
  `2026-06-22T14:13:52Z`; `dataset_size=131694592`, `batches=16461824`, and
  `world_size=4` printed at `2026-06-22T14:13:57Z`.
- TOS sync state at 2026-06-22 22:17 China time: `train_log.jsonl` had reached
  step 4240 and `pretrained_latest.pt` was present on TOS, about 41 MB.
- The sync loop uploads `pretrained_latest.pt` and `train_log.jsonl` every
  `DBMIM_SYNC_SEC=60`, so the downstream watcher can trigger mid-training once
  the TOS log reaches `DBMIM_R20_MIN_STEP=40000`.

Once R20 is running, the downstream watcher can be launched immediately. It
will wait for both `pretrained_latest.pt` and `train_log.jsonl` to reach the
default `DBMIM_R20_MIN_STEP=40000` before submitting finetune jobs. This avoids
running downstream on a very early 2000-step checkpoint.

```bash
screen -dmS dbmim_r20_finetune_watch bash -lc \
  'cd /volume/med-train/users/dchen02/code/dbMiM && \
   ./scripts/watch_and_submit_full_em_finetune.sh \
   > outputs/watchers/full_em_finetune_watch_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1'
```

It submits:

- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-fullem-r20q`
- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-bcar-rank-fullem-r20q`

with `--post-train-official-abc-eval`. Summarize with:

```bash
python scripts/poll_dbmim_tos_results.py --group r20q --once --logs --siflow-fallback
```

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

For official A/B/C stages, the poller prints `PARTIAL` until all three CREMI
samples appear in `sample_names`. This matters for `r17q_fine`: early stdout
fallback with only sample A can have a very low VOI, but it is not an A/B/C
result.

Active R19 submissions on 2026-06-22:

| purpose | UUID | GPUs | pool |
|---|---|---:|---|
| context48 publicEM | `e9e01802-c98e-466b-b3cf-f5cf1b0edbbd` | 2 | med-model |
| context48 scratch | `6655b114-66b7-4a66-8efc-d55ca6d2dfcc` | 2 | med-model |
| fs48 publicEM | `77d049b8-76a1-4369-84c7-d02bc361851e` | 2 | med-model |
| fs48 scratch | `2ade7cb2-667c-4410-b25d-cba312fc112e` | 2 | med-model |

Poll command:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r19q --once --logs --siflow-fallback
```

## R24 PublicEM dbMiM++ Operations

R24 is the current hard test of dbMiM-specific gain over plain MAE. Use the
project submitter and watcher, not ad hoc SiFlow commands:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  /volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python \
  scripts/submit_siflow_dbmim.py \
  --stage pretrain-public-em-decoderaware-r24 \
  --resource-pool med-model \
  --gpus-per-pod 4 \
  --submit
```

The submitted R24 pretrain UUID is
`bd588c91-6328-455e-a1bc-fc1a3316bbdd`. A login-node watcher was started with:

```bash
nohup scripts/watch_and_submit_public_em_decoderaware_r24_finetune.sh \
  > outputs/watchers/public_em_decoderaware_r24_watcher.log 2>&1 &
```

It waits for:

- `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_public_em_decoderaware_dbmim_r24/checkpoint_step_00160000.pt`
- `train_log.jsonl` max step at least 160000

Then it submits the 2-GPU downstream stage
`finetune-cremi-unetr-aniso-arch-explore-maws-mse-publicem-decoderaware-r24q`
with post-train official A/B/C waterz evaluation.

Do not count the fullEM plain-MAE shared retry as productive running work while
it reports `实例配额不足 | 需求:4, 实际可用(实例配额):0`. It is a queued future
control, not a completed or running baseline.
