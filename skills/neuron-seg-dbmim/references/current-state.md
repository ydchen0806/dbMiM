# Current State Reference

Use this file to orient a new agent before reporting status or launching more
jobs. Re-check local files because job status and outputs change.

## Repository

- Repo path: `/volume/med-train/users/dchen02/code/dbMiM`
- Remote: `https://github.com/ydchen0806/dbMiM.git`
- Main maintained docs: `DBMIM_WORKFLOW.md` and `EXPERIMENTS.md`
- Maintained model path: `dbmim/models.py::UNETRAnisotropicAffinityNet`
- Maintained training entrypoints:
  - `train_pretrain.py`
  - `train_finetune.py`
  - `scripts/evaluate_cremi_segmentation.py`
  - `scripts/submit_siflow_dbmim.py`

## Current Paper-Aligned Experiment

The active scientific question is whether dbMiM pretraining improves CREMI
neuron segmentation with a paper-aligned anisotropic UNETR backbone and
VOI/ARAND evaluation.

## 2026-06-22 R16 Public-EM Pretraining Wave

The public-EM pretraining path is now reproducible and TOS-backed:

- Pretraining config: `configs/pretrain_public_em_membrane_r16.yaml`
- Data prep: `scripts/prepare_public_em_pretrain_data.py`
- Data prefix:
  `tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data/public_em/`
- Output prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_public_em_membrane_dbmim_r16/`
- Pretrain UUID:
  `5a10fe9e-2d34-4568-8009-5902c73cc592`
- Status: succeeded on 2026-06-21/22, `med-model`, 8 GPUs.
- Dataset seen in pod: public ISBI2012 + SNEMI3D HDF5 volumes, plus CREMI.
- Final available checkpoint:
  `pretrain_public_em_membrane_dbmim_r16/pretrained_latest.pt`, 40.20 MB.
- The checkpoint was explicitly re-downloaded and inspected on 2026-06-22:
  `global_step=160000`, `epoch=44`, `max_steps=160000`, and 82 model keys.
  Do not describe it as an early or undertrained pretraining checkpoint.
- Final observed pretrain loss near the end was about `0.0486`; the exact
  rank-0 logs are in SiFlow stdout and `pretrain_log.jsonl` on TOS.
- The offline public-EM data is only about 195 MB and contains ISBI 2012 plus
  SNEMI3D raw HDF5 volumes. The larger HF `cyd0806/EM_pretrain_data` manifest
  totals about 486 GB. As of 2026-06-22 11:22 China time, the user-provided HF
  token is stored only in `/volume/med-train/users/dchen02/secrets/hf_env_dchen02.sh`
  and the gated full-EM data preparation is actively running. Do not print the
  token or store it in git. Do not claim gated all-EM pretraining has started
  until the five groups are present on TOS and the R20 pretraining submission
  marker/log exists.

Important operational pitfall: local `tosutil cp` and watcher probes can hang
when proxy variables point at `192.168.32.28:18000`. Always unset
`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and lowercase variants before TOS or
SiFlow SDK probes. After unsetting proxies, `pretrained_latest.pt` downloaded
successfully in under a second.

## 2026-06-22 R20 Gated Full-EM Pretraining

Full gated HF data preparation finished on 2026-06-22. The download/upload log
is:

```bash
tail -n 80 /volume/med-train/users/dchen02/code/dbMiM/outputs/watchers/full_em_download_20260622T033243Z.log
```

It ended with `done_group mb_moc`, `done_full_em_download`, and
`exit_full_em_download code=0`. There are no remaining screen sessions for the
download/watcher path.

The source dataset was `cyd0806/EM_pretrain_data`. The user-provided HF token is
stored only in `/volume/med-train/users/dchen02/secrets/hf_env_dchen02.sh`.
Never print the token or commit it.

Compressed bytes from the HF manifest:

| group | compressed size |
|---|---:|
| `fafb` | about 117.38 GB |
| `fib25` | about 116.10 GB |
| `kasthuri` | about 115.37 GB |
| `mitoem` | about 16.66 GB |
| `mb_moc` | about 137.32 GB |

Extracted local HDF5 counts/sizes after completion:

| group | HDF5 count | local size |
|---|---:|---:|
| `fafb` | 1421 | about 139 GB |
| `fib25` | 1857 | about 182 GB |
| `kasthuri` | 1732 | about 170 GB |
| `mitoem` | 81 | about 8.0 GB |
| `mb_moc` | 2944 | about 121 GB |

TOS contains actual HDF5 files for all five groups under:

```text
tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data/<group>/
```

Because the upload command copied each group directory to a same-named TOS
prefix, object paths are nested like `<group>/<group>/...`. This is expected;
the pod-side gate checks `*/<group>/*` and the dataset loader recurses.

The full-data pretraining config/stage is:

- Config: `configs/pretrain_em_full_membrane_r20.yaml`
- Stage: `pretrain-em-full-membrane-r20`
- Output: `outputs/pretrain_em_full_membrane_dbmim_r20`
- Output TOS prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_full_membrane_dbmim_r20/`

Important SiFlow scheduling state:

- Initial 8-GPU task `1488e82a-301a-4b8a-961c-615657dd3491` was stopped before
  running because the first bundle would have copied hundreds of GB of full-EM
  data into the bundle/runtime temporary path.
- Fixed 8-GPU task `1d0916f7-0c98-4d2f-8c02-3ec7cd47a46b` was stopped because
  `med-model` reported `实例配额不足 | 需求:8, 实际可用(实例配额):7`.
- Fixed 7-GPU task `a34bbacf-b483-4bb2-85cd-852adf9e8e16` was stopped because
  SiFlow reported resource fragmentation.
- Failed task: `ab50e050-a1c9-42ee-b40d-e4e43e109212`, `med-model`,
  `sci.g21-3`, 4 GPUs. It started running at `2026-06-22T09:02:49Z` and
  failed at `2026-06-22T10:23:27Z`, after all five full-EM groups had been
  copied from TOS but before `train_pretrain.py` launched.
- Retry 8-GPU task `16b28f27-7e75-40d9-a5ec-04f3067b4001` was stopped
  immediately because `med-model` again reported
  `实例配额不足 | 需求:8, 实际可用(实例配额):7`.
- Active retry task: `7be2f62b-c1f3-482a-85a7-74cd63c63c35`, `med-model`,
  `sci.g21-3`, 4 GPUs. It started running at `2026-06-22T13:46:02Z`.

The active R20 bundle stages full-EM data to a mounted-volume path instead of
the bundle temporary directory:

```text
/volume/med-train/users/dchen02/code/dbMiM_runtime/em_pretrain_data/full_r20/all
```

The `ab50e050-a1c9-42ee-b40d-e4e43e109212` logs showed successful staging for
all five full-EM groups, then the shell exited before training because
`set -euo pipefail` treated `find ... | head -20` as a failed pipeline when
`head` closed early and `find` received SIGPIPE. The maintained submitter was
patched on 2026-06-22 to append `|| true` to this diagnostic listing. If a new
R20 run fails before training, inspect the generated `run.sh` and stdout before
assuming a data or model issue.

At `2026-06-22T13:47Z`, active retry
`7be2f62b-c1f3-482a-85a7-74cd63c63c35` had downloaded the bundle, installed
offline wheels, downloaded/extracted CREMI, and started the full-EM TOS copy.
Expect no training loss until all five groups finish staging.

At `2026-06-22T14:13:52Z`, the same retry completed all five full-EM TOS copy
steps and printed `em_pretrain_data_status='available_offline_tos'`. The
patched diagnostic `find ... | head -20 || true` passed, so the previous
pipefail/SIGPIPE failure is fixed. Training launched at `2026-06-22T14:13:57Z`
with:

- `dataset_size=131694592`
- `batches=16461824`
- `world_size=4`
- `model_trainable_params=3254848`
- `decision_trainable_params=247555`

Early loss was healthy enough for a real run: step 20 loss about `0.134`, step
160 loss about `0.058`, and the TOS-synced `train_log.jsonl` reached step 4240
by `2026-06-22T14:17Z`. `pretrained_latest.pt` also appeared on TOS by then
with size about 41 MB. The downstream watcher still waits for
`DBMIM_R20_MIN_STEP=40000` before submitting finetune/eval jobs.

A finetune watcher is available:

```bash
scripts/watch_and_submit_full_em_finetune.sh
```

It polls
`tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_full_membrane_dbmim_r20/pretrained_latest.pt`
and `train_log.jsonl`. By default it waits for at least step 40000 before
submitting downstream arms, matching the R20 decision-module freeze point and
avoiding very early 2000-step checkpoints. Override with `DBMIM_R20_MIN_STEP`
only for smoke tests. It then submits two 2-GPU downstream arms with post-train
official A/B/C waterz evaluation:

- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-fullem-r20q`
- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-bcar-rank-fullem-r20q`

These are matched to the existing R17 scratch controls in
`scripts/poll_dbmim_tos_results.py --group r20q`.

Matched R16 downstream controls are running on `med-model`, 2 GPUs each, all
with post-train A/B/C architecture benchmark (`graph_cc`, `cupy_graph_cc`,
`seeded_rag`, `waterz`):

| arm | UUID | expected load |
|---|---|---|
| long-affinity + publicEM dbMiM | `11df9593-273a-4456-8da4-6f844b1d8292` | loaded 77 pretrained keys |
| long-affinity scratch | `198e39cc-c2aa-4ab1-82f3-edcc15e6917f` | no pretrained keys |
| MAWS+BCAR rank + publicEM dbMiM | `b8e5f3dc-9047-48a2-9b90-dd439f0265c7` | loaded 77 pretrained keys |
| MAWS+BCAR rank scratch | `997ba3ec-77f6-41f0-851b-83f770427cfd` | no pretrained keys |

The submitter records live under
`/volume/med-train/users/dchen02/siflow_submissions/yinda_public_skill/`.
Use `scripts/poll_dbmim_tos_results.py --group r16q --once --logs
--siflow-fallback` after post-processing finishes to summarize VOI/ARAND even
if TOS summary downloads are slow.

Because the arch-bench evaluates slow/low-value graph/RAG paths before waterz,
four standalone 1-GPU official A/B/C waterz-only evals were also launched on
2026-06-22. They use `--backends waterz`, `--max-samples 0`, and CREMI
boundary-ignore `xy=1,z=0`:

| arm | UUID |
|---|---|
| long-affinity + publicEM waterz A/B/C | `d820080c-2c13-4278-980d-155add949017` |
| long-affinity scratch waterz A/B/C | `418b6d25-06df-4820-8594-53e7a55e9b9c` |
| MAWS+BCAR rank + publicEM waterz A/B/C | `a9456991-13c5-41cc-af72-f3f4427b3f26` |
| MAWS+BCAR rank scratch waterz A/B/C | `b51f6415-2bae-4e34-8b95-50994f092496` |

Use `scripts/poll_dbmim_tos_results.py --group r16q_waterz --once --logs
--siflow-fallback` for these standalone waterz evals.

As of 2026-06-22 08:16 China time, the standalone R16 waterz-only evals are
still running and have only sample-A metric rows in SiFlow stdout. Current
sample-A best rows under the official boundary-ignore mask are:

| arm | best VOI | ARAND at best VOI | note |
|---|---:|---:|---|
| long-affinity + publicEM | `0.237032` | `0.028895` | bias `z=-0.25,y=-0.50,x=-0.50`, threshold `0.50` |
| long-affinity scratch | `0.235447` | `0.040266` | threshold `0.50` |
| MAWS+BCAR rank + publicEM | `0.234412` | `0.029636` | threshold `0.50` |
| MAWS+BCAR rank scratch | `0.226193` | `0.033905` | threshold `0.50`; best ARAND row is `0.028309` with bias |

Do not report these as A/B/C aggregates until `sample_B_20160501.hdf` and
`sample_C_20160501.hdf` appear in `sample_names`.

## 2026-06-22 Best Current Downstream Signal

The most reliable positive signal so far is the R17 MSE+MAWS publicEM-vs-scratch
comparison under official A/B/C full-volume waterz evaluation:

| arm | VOI sum | best ARAND | note |
|---|---:|---:|---|
| MSE+MAWS + publicEM dbMiM | 1.002919 | 0.188832 | best current publicEM arm |
| MSE+MAWS scratch | 1.095164 | 0.210442 | matched architecture/loss |
| MSE+MAWS+BCAR + publicEM dbMiM | 1.063910 | 0.193443 | positive but weaker |
| MSE+MAWS+BCAR scratch | 1.087732 | 0.196075 | matched BCAR control |

This gives roughly `-0.092` VOI and `-0.0216` best-ARAND for publicEM
pretraining on the strongest MSE+MAWS arm. R19 context48/fs48 also shows
positive pretraining deltas but does not beat R17 MSE+MAWS. Therefore the R20
full-EM downstream watcher follows the R17 MSE+MAWS and MSE+MAWS+BCAR-rank
recipes rather than prioritizing context48/fs48.

## 2026-06-22 R17 MSE/MAWS Wave

R17 directly tests the user's MSE hypothesis under the strongest current
structure: UNETR-aniso-EM + MAWS, with and without BCAR. It uses the same
publicEM dbMiM checkpoint as R16 and matched scratch controls. All four jobs
were submitted on `cn-shanghai/changliu`, pool `med-model`, `sci.g21-3`, 2
GPUs each, with post-train official A/B/C waterz-only eval in the same pod.

| arm | UUID | initial status |
|---|---|---|
| MAWS + MSE + publicEM dbMiM | `68013c0c-b712-4c93-9b46-984c69f812ac` | Running; step 960 loss `0.057741`, loaded 77 keys |
| MAWS + MSE scratch | `1d218802-03e8-4121-8101-09637a775089` | Running; step 700 loss `0.102678` |
| MAWS + MSE + BCAR + publicEM dbMiM | `8f9e4a5e-729f-42b5-8516-d0c0784fb8cb` | Running; step 360 loss `0.143978`, main `0.142010`, loaded 77 keys |
| MAWS + MSE + BCAR scratch | `d208dea6-9b2c-4dc3-ae94-a5104ad38d39` | Running; step 160 loss `0.159586`, main `0.145523` |

Configs:

- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_bcar_rank_publicem_r17q.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_bcar_rank_scratch_r17q.yaml`

Poll with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r17q --once --logs --siflow-fallback
```

The first training logs already show that publicEM initialization starts with a
lower MSE loss than scratch, but this is not enough to claim segmentation
improvement. Wait for waterz A/B/C VOI/ARAND.

As of 2026-06-20, the earlier R3/R4 supervised finetuning results are
invalidated as scientific evidence because the dataset applied random geometric
flips to the image but not to the instance label. This broke image/label
alignment during supervised affinity training. Keep their logs only as
debugging history.

Current controlled arms:

- `aniso UNETR pretrained`: existing dbMiM pretrained ViT encoder, loaded into
  anisotropic UNETR with interpolated positional embeddings.
- `aniso UNETR scratch`: same architecture, random initialization.
- `aniso UNETR long-pretrained`: long pretraining at `32x160x160`, then same
  finetune.

Open structural/loss/context ablations:

- `no-dtrans`: remove anisotropic z compression.
- `dtrans2`: z stride 2 instead of paper-style stride 4.
- `fs64`: decoder feature size 64.
- `boundary-loss`: stronger z/xy loss weighting.
- `context48`: larger crop `48x192x192`.

## Submitted Jobs Snapshot

All jobs below were submitted on 2026-06-17 via Shanghai changliu with
`sci.g21-3`, 8 GPUs per training pod. Saved submission JSONs show
`resource_pool: cn-shanghai-changliu-skyinfer-reserved-shared`, not
`med-model`.

| stage | UUID | TOS output prefix |
|---|---|---|
| long dbMiM pretrain | `56d6c8f0-184f-4dcd-98c2-01060b3230a0` | `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_cremi_real_long_dbmim/` |
| aniso pretrained finetune | `90e2ca36-6c50-4346-8fa3-d2320b914459` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_pretrained/` |
| aniso scratch finetune | `adb18e9f-e4a5-4935-a00e-485388c9545b` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_scratch/` |
| no-dtrans | `1d9a076c-7bea-4224-8067-94bda7b39c95` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_no_dtrans/` |
| dtrans2 | `7a5631c4-63bc-4ae6-ad83-feca3fc78221` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_dtrans2/` |
| fs64 | `53424b91-e37c-4ecd-9eaf-311042355c5c` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_fs64/` |
| boundary-loss | `d015653e-fc71-434c-b6c8-96a51cc04e4a` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_boundary_loss/` |
| context48 | `c4982264-be93-4f50-8aaf-b09cc3feb655` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_context48/` |

Watcher logs live in `outputs/watchers/`. If their tail only says
`checkpoint_wait`, no checkpoint has been observed in TOS yet.

## R5 Root-Cause Fix Snapshot

On 2026-06-20 the supervised dataset and loss were fixed to align with the
SuperHuman/Kisuk affinity-training style:

- `dbmim/datasets.py` now uses synchronized image/label flips through
  `augment_image_and_label`.
- `data.widen_border: true` performs 2D in-plane instance-border invalidation
  before affinity generation.
- `train_finetune.py` supports `loss.loss_type: weighted_mse`, a
  SuperHuman-style per-sample/channel binary ratio weighting over sigmoid
  affinities.
- R5 configs are:
  - `configs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5.yaml`
  - `configs/finetune_cremi_real_unetr_aniso_superhuman_scratch_r5.yaml`

Current R5 SiFlow jobs:

| arm | UUID | status at last update | notes |
|---|---|---|---|
| pretrained R5 4-GPU train | `f9f74d70-4798-400f-a189-afad4c1d7e4e` | Succeeded | med-model, global batch 8, loaded 77 pretrained keys, 30k final checkpoint uploaded |
| pretrained R5 full-volume sample-A eval, mid-training latest | `9cabcc88-d7ed-4936-89ca-e2891904b648` | Stopped after useful evidence | 1 GPU med-model, waterz/skimage, sample A full volume |
| pretrained R5 full-volume sample-A eval, final 30k | `68115eb6-ecd3-4f6b-af50-98c26c35faee` | Succeeded | 1 GPU med-model, waterz/skimage, sample A full volume |
| scratch R5 4-GPU train | `162150c0-9350-4254-bdce-c599b7fdcfa4` | Succeeded | med-model, global batch 8, no pretrained, 30k final checkpoint uploaded |
| scratch R5 full-volume sample-A eval, final 30k | `a4afc427-3b32-4439-b9b5-cf6f5aec37c7` | Succeeded | 1 GPU med-model, waterz/skimage, sample A full volume |
| no-widen pretrained R5 ablation | `31935cc0-64f0-4e55-bacd-14d043741acb` | Succeeded | cpt-train, 4 GPU, disables label border widening |
| BCE pretrained R5 ablation | `41fe0043-f7e5-4e10-8067-4c5c5618d926` | Succeeded | cpt-train, 4 GPU, uses BCE instead of weighted MSE |
| boundary-high pretrained R5 ablation | `e28c3264-aaa5-491a-a79f-f307bbe5ef43` | Succeeded | cpt-train, 4 GPU, boundary Dice weight 0.55 |
| no-widen pretrained R5 sample-A eval | `28295ae2-2b2a-4c00-a8b7-5e4c9ba3da16` | Running | cpt-train, 1 GPU, waterz/skimage |
| BCE pretrained R5 sample-A eval | `adb48dd4-e833-4744-addf-78c3defc2c5e` | Running | cpt-train, 1 GPU, waterz/skimage |
| boundary-high pretrained R5 sample-A eval | `c605d26c-163b-4b3b-8fa4-512fbe7029c5` | Running | cpt-train, 1 GPU, waterz/skimage |
| pretrained R5 A/B/C full-volume eval | `812df662-d464-49d3-8622-93da3e1709cc` | Submitted | med-model, 1 GPU, all CREMI samples |
| scratch R5 A/B/C full-volume eval | `98bc4983-7ca2-4f6e-9fe4-31f4a2791ba0` | Submitted | med-model, 1 GPU, all CREMI samples |

Stopped R5 attempts due insufficient actual `sci.g21-3` availability:

- `26b6b0c5-5c9c-497f-80b7-4daef7a305ce`: pretrained 8-GPU med-model.
- `8e5119d9-681d-41f7-afd4-4d318e18e0e6`: scratch 8-GPU med-model.
- `452f68a8-dc34-4a60-a3a6-85e9d9cf86c7`: pretrained 8-GPU shared pool.

R5 pretrained training signs are healthier than R4:

- step 600: `train_loss` about `0.321`, boundary Dice loss about `0.552`.
- step 10k-12k: common `train_loss` about `0.10-0.19`, boundary Dice loss
  about `0.14-0.27`.
- TOS checkpoint prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5/`.

The R5 eval confirmed true full-volume sample A scope:
`raw_shape [125,1250,1250]`, crop `[[0,125],[0,1250],[0,1250]]`, checkpoint
`outputs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5/finetuned_latest.pt`.

First raw `pred` waterz/skimage rows are in the normal CREMI scale:

| threshold | VOI sum | VOI split | VOI merge | ARAND | n_pred | n_gt |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.742093 | 0.449843 | 0.292251 | 0.126184 | 27535 | 37366 |
| 0.10 | 0.759984 | 0.430677 | 0.329307 | 0.161975 | 23599 | 37366 |
| 0.20 | 0.747036 | 0.410954 | 0.336082 | 0.162093 | 19172 | 37366 |
| 0.30 | 0.735735 | 0.396551 | 0.339184 | 0.161714 | 16231 | 37366 |
| 0.50 | 0.717879 | 0.371610 | 0.346268 | 0.161406 | 12209 | 37366 |

This strongly supports the root-cause diagnosis: the previous VOI 8-10 range
was caused mainly by broken supervised augmentation/target training, not by the
VOI implementation itself. Continue the pretrained R5 run to 30k and start/keep
the scratch R5 control for the pretraining-effect comparison.

Final pretrained R5 30k sample-A eval has improved further. On the final
checkpoint, uncalibrated waterz `hist_quantile` gives:

| threshold | VOI sum | VOI split | VOI merge | ARAND | n_pred | n_gt |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.645302 | 0.367920 | 0.277382 | 0.111310 | 21346 | 37366 |
| 0.10 | 0.637800 | 0.359174 | 0.278626 | 0.111051 | 18972 | 37366 |
| 0.20 | 0.628222 | 0.347504 | 0.280718 | 0.110802 | 16394 | 37366 |
| 0.30 | 0.620097 | 0.336619 | 0.283478 | 0.110683 | 14557 | 37366 |
| 0.50 | 0.604249 | 0.315553 | 0.288696 | 0.109885 | 11402 | 37366 |

The final calibration sweep improved both VOI and ARAND. Current best confirmed
pretrained R5 row is `VOI_sum=0.563006`, `ARAND=0.073386`, waterz threshold
`0.50`, calibration bias `z=-0.50,y=-1.00,x=-1.00`.

The scratch R5 train finished 30k with final `train_loss=0.122041`, affinity
Dice loss `0.032573`, and boundary Dice loss `0.161410`. Its full-volume
sample-A calibration eval finished as UUID
`a4afc427-3b32-4439-b9b5-cf6f5aec37c7`.

Scratch sample-A is currently better than pretrained sample-A:

- scratch best-by-VOI: `VOI_sum=0.546834`, `ARAND=0.066024`,
  bias `z=-0.25,y=-0.50,x=-0.50`, threshold `0.50`.
- scratch best-by-ARAND: `ARAND=0.058093`, `VOI_sum=0.548672`,
  bias `z=-0.50,y=-1.00,x=-1.00`, threshold `0.50`.
- pretrained best: `VOI_sum=0.563006`, `ARAND=0.073386`,
  bias `z=-0.50,y=-1.00,x=-1.00`, threshold `0.50`.

Interpretation: R5 fixed the segmentation method, but sample-A currently does
not support a positive dbMiM-pretraining effect. Wait for the A/B/C full-volume
evals before making a broader claim.

## 2026-06-20 Current R5 Addendum

Sample-A ablations completed with the raw-label `ignore_label=0` metric path:

- BCE pretrained R5 is the best current sample-A VOI row:
  `VOI_sum=0.535448`, `ARAND=0.080269`, bias `z=-0.50,y=-1.00,x=-1.00`,
  waterz threshold `0.50`.
- BCE's best ARAND row is `ARAND=0.067376`, `VOI_sum=0.543432`, threshold
  `0.20`, still weaker than scratch R5's best ARAND `0.058093`.
- Boundary-high pretrained R5 does not improve the main R5 stack:
  `VOI_sum=0.559225`, `ARAND=0.073207`.
- No-widen pretrained R5 is catastrophic:
  `VOI_sum=4.516494`, `ARAND=0.961479`. Treat 2D instance-border invalidation
  as a required part of the method.

Metric-mask clarification:

- CREMI train HDF5 `volumes/labels/neuron_ids` has no original zero labels on
  samples A/B/C, so raw-label `ignore_label=0` is stricter than the CREMI
  challenge boundary-ignore convention.
- `scripts/evaluate_cremi_segmentation.py` now supports
  `--cremi-boundary-ignore-distance-xy` and
  `--cremi-boundary-ignore-distance-z`.
- The official-style A/B/C stages use `xy=1,z=0`, matching the "same section"
  object-boundary wording. Do not mix these official-style numbers with
  raw-label numbers.
- Official-style sample-A rows observed so far:
  - pretrained R5 raw pred threshold `0.50`: `VOI_sum=0.246585`,
    `ARAND=0.077635`, effective `n_gt=10929`, ignored voxels `11.69%`.
  - scratch R5 raw pred threshold `0.50`: `VOI_sum=0.204558`,
    `ARAND=0.048634`, effective `n_gt=10929`, ignored voxels `11.69%`.
  - scratch R5 raw pred threshold `0.10`: best current ARAND
    `0.046854`, `VOI_sum=0.223857`.

Raw-label A/B/C follow-up finished on 2026-06-20:

- pretrained R5 strict raw-label A/B/C: best VOI `1.556559`, best ARAND
  `0.356289`.
- scratch R5 strict raw-label A/B/C: best VOI `1.441593`, best ARAND
  `0.312191`.

Interpretation: strict raw-label A/B/C confirms the sample-A trend. Scratch is
currently ahead of dbMiM-pretrained R5; do not claim a pretraining gain from
the current 30k setup.

Updated follow-up jobs:

| job | UUID | status note |
|---|---|---|
| raw-label pretrained R5 A/B/C | `812df662-d464-49d3-8622-93da3e1709cc` | succeeded, strict diagnostic |
| raw-label scratch R5 A/B/C | `98bc4983-7ca2-4f6e-9fe4-31f4a2791ba0` | succeeded, strict diagnostic |
| official-style pretrained R5 A/B/C | `44efdd27-baf7-418b-aa80-3b74207dc70b` | running, CREMI boundary-ignore xy=1 |
| official-style scratch R5 A/B/C | `5aad9226-1c61-40a4-b330-93bdf87abc31` | running, CREMI boundary-ignore xy=1 |
| corrected encoder-LR pretrained R5 | `c5f2a38a-d6a7-46d2-b46d-73b57a0f000b` | succeeded, encoder lr `2e-5`, decoder lr `8e-5` |
| encoder-LR official-A eval, failed packaging attempt | `1b8cf096-294e-46d5-a3f6-029c219dda08` | failed before eval due newer waterz pyproject/setuptools incompatibility |
| encoder-LR official-A eval, stable waterz_v08 bundle | `8400bb84-5126-4c96-9266-fbe54b5499a5` | running/submitted |
| BCE scratch R5 train | `9f2bbe23-abfa-41c0-b00c-63f76fd76868` | running, cpt-train 4 GPU |
| BCE pretrained official-A eval | `47ddaea5-1a31-4c7c-8518-326d4c46b869` | running, official boundary-ignore sample A |

The first low-encoder run at
`finetune_cremi_real_unetr_aniso_superhuman_lowencoder_pretrained_r5` is
invalid because DDP parameter names prevented encoder/decoder optimizer-group
splitting. It was stopped; use the corrected `encoderlr` prefix instead.

Official-style sample-A calibrated rows observed so far:

- pretrained R5 best observed: `VOI_sum=0.198980`, `ARAND=0.038487`, bias
  `z=-0.50,y=-1.00,x=-1.00`, threshold `0.50`.
- scratch R5 best observed: `VOI_sum=0.177157`, `ARAND=0.021736`, bias
  `z=-0.50,y=-1.00,x=-1.00`, threshold `0.50`.

This is the right CREMI-style direction: boundary-ignore brings VOI well below
the raw-label strict numbers. It still does not show a pretraining gain.

Final-checkpoint watcher logs:

- old scratch watcher, superseded by manual eval:
  `outputs/watchers/eval_scratch_r5_20260620T091340Z_final30k.log`
- no-widen watcher attempt:
  `outputs/watchers/eval_nowiden_pretrained_r5_20260620T093118Z_final30k_nohup.log`
- BCE watcher attempt:
  `outputs/watchers/eval_bce_pretrained_r5_20260620T093118Z_final30k_nohup.log`
- boundary-high watcher attempt:
  `outputs/watchers/eval_boundaryhigh_pretrained_r5_20260620T093118Z_final30k_nohup.log`

Those watcher attempts intentionally waited only for `checkpoint_step_00030000.pt`,
not `finetuned_latest.pt`, but local background processes were not reliable in
this environment. Prefer explicit SiFlow/TOS polling and manual eval
submission once the final checkpoint appears.

## 2026-06-21 Current Snapshot

Official-style A/B/C evaluations with the CREMI boundary-ignore mask
(`xy=1,z=0`) completed:

- pretrained R5 official A/B/C, UUID `44efdd27-baf7-418b-aa80-3b74207dc70b`,
  best VOI `1.250565`, best ARAND `0.332747`.
- scratch R5 official A/B/C, UUID `5aad9226-1c61-40a4-b330-93bdf87abc31`,
  best VOI `1.121139`, best ARAND `0.287647`.

The boundary-ignore mask effective GT counts were sample A `10929`, sample B
`1092`, and sample C `1878`. The ignored voxel fractions were `11.6916%`,
`8.4857%`, and `10.5215%`, respectively.

Conclusion: the VOI implementation and post-processing are now in the expected
CREMI-scale regime, but the current standard dbMiM-pretrained R5 arm does not
beat scratch. Treat scratch R5 as the current aggregate baseline.

Official sample-A ablations now available:

- BCE pretrained R5, UUID `47ddaea5-1a31-4c7c-8518-326d4c46b869`:
  best VOI/ARAND `VOI_sum=0.185597`, `ARAND=0.032785`, bias
  `z=-0.50,y=-1.00,x=-1.00`, threshold `0.20`.
- encoder-LR pretrained R5, UUID `8400bb84-5126-4c96-9266-fbe54b5499a5`:
  best VOI `0.210299`, best ARAND `0.036421`.

BCE pretrained is a credible sample-A improvement over the standard pretrained
arm but is not proven to be a pretraining effect. The paired BCE scratch train
finished as UUID `9f2bbe23-abfa-41c0-b00c-63f76fd76868`, final step 30000
`train_loss=0.147030`, `train_main_loss=0.091217`, `train_dice_loss=0.028496`,
and `train_boundary_dice_loss=0.155396`.

Paired BCE scratch evaluations submitted on 2026-06-21:

| job | UUID | status note |
|---|---|---|
| official-style BCE scratch sample A | `df6582ca-c87c-4843-a8ca-23e41030c3b5` | succeeded |
| raw-label BCE scratch sample A | `4ce29ab6-4775-41f5-ac70-c2b62d94b9c4` | succeeded |

Paired BCE scratch results:

- official-style sample A: best VOI `0.169502`, best ARAND `0.024062`.
- raw-label sample A: best VOI `0.517256`, best ARAND `0.058831`.

Both beat the paired BCE-pretrained rows (`0.185597`/`0.032785` official and
`0.535448`/`0.067376` raw-label). Report BCE as a useful loss/training-stack
choice, not as evidence for a positive dbMiM-pretraining effect. Expand BCE
scratch/pretrained to A/B/C only if a full aggregate comparison is needed.

## R2 Restart Snapshot

On 2026-06-18 the old local watchers were killed because no intermediate
checkpoint was visible. The root cause was the bootstrap script uploading
training output directories only after the training process exited. A second
attempt to stop the old remote SiFlow training UUIDs through the SDK hung in
network/proxy connection setup and was interrupted locally; do not assume the
remote tasks stopped unless a fresh SiFlow status query confirms it.

R2 training jobs were submitted explicitly to Shanghai changliu `med-model`,
`sci.g21-3`, 8 GPUs each:

| stage | UUID | TOS output prefix |
|---|---|---|
| pretrained-r2 | `9291405a-046f-4165-bbac-d0fdf71fb3eb` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_pretrained_r2/` |
| scratch-r2 | `151ff158-429d-486e-9ffd-59bc71dbe458` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_scratch_r2/` |
| lsd-pretrained-r2 | `2bbd5c24-2124-4d0e-89a2-326932ecd866` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_lsd_pretrained_r2/` |
| lsd-scratch-r2 | `9cc450dd-ba4a-4c43-98ea-765aa363d768` | `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_lsd_scratch_r2/` |

R2 watcher logs:

- `outputs/watchers/eval_pretrained-r2_20260618T154324_setsid.log`
- `outputs/watchers/eval_scratch-r2_20260618T154324_setsid.log`
- `outputs/watchers/eval_lsd-pretrained-r2_20260618T154324_setsid.log`
- `outputs/watchers/eval_lsd-scratch-r2_20260618T154324_setsid.log`

The R2 bootstrap now uploads `finetuned_latest.pt`, `finetuned_best.pt`, and
JSONL logs periodically while training runs. Evaluation still waits for a
checkpoint before submitting 1-GPU VOI/ARAND jobs.

## Historical Results To Treat Carefully

- The simplified `UNETRAffinityNet` pretrained-vs-scratch run is a negative
  diagnostic, not a paper-aligned conclusion.
- Its crop evaluation produced identical ARAND for pretrained and scratch and
  VOI sum around 4.05 on small center crops. Do not use it to claim dbMiM does
  or does not help.
- MAE-head and graph-CC baselines are useful engineering references but do not
  replace the anisotropic UNETR comparison.

## Status Reporting Checklist

When asked for current status:

1. Check `date`, `outputs/watchers/*.log`, and relevant TOS prefixes.
2. Check submission JSONs under
   `/volume/med-train/users/dchen02/siflow_submissions/yinda_public_skill/`.
3. Report resource pool from the saved JSON, not from memory.
4. Separate "submitted", "running/queued", "checkpoint observed", "eval
   submitted", and "metrics available".
5. If SiFlow SDK log/status calls timeout, say so and fall back to watcher/TOS
   evidence. Do not invent loss values.

## 2026-06-21 R14 Active State

Latest pushed repo commit: `6c1bd4b Submit full membrane-pretrained R14
experiments`.

R14 MA-dbMiM pretraining completed to 120k steps. Checkpoint prefix:

```text
tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_membrane_dbmim_r14/
```

It includes `pretrained_latest.pt`, `checkpoint_step_00120000.pt`, and
2k-step intermediate checkpoints. This run used CREMI-only fallback because the
all-EM TOS data prefix had no HDF5 volumes.

Quick R14q complete A/B/C results:

| run | best VOI | ARAND at best VOI | UUID |
|---|---:|---:|---|
| `bcar-rank-allpretrained-r14q` | 1.0818 | 0.1965 | `fa076c76-f3bf-4eac-91ac-c8f4a1677062` |
| `maws-allpretrained-r14q` | 1.0745 | 0.2011 | `2f63a6dc-c7a6-4e8b-97de-34ab80985b40` |
| `maws-bcar-rank-allpretrained-r14q` | 1.0407 | 0.1929 | `38b18ca3-d4c8-4fd2-94d9-632f590d92ce` |
| `maws15-bcar-rank-allpretrained-r14q` | 1.0441 | 0.1973 | `30cb7b4a-ac52-402a-9085-c97749ab5f2b` |

Do not treat `bcar-calib-allpretrained-r14q` as an A/B/C conclusion unless a
fresh summary has `n=3`; the first fallback row only had sample A.

Active full R14 jobs submitted on 2026-06-21:

| run | UUID | GPUs | note |
|---|---|---:|---|
| `em-shwmse-mempretrained-r14` | `f4c00499-19a5-4fb2-99a8-99adf54bad4d` | 4 | Running |
| `em-shwmse-bcar-mempretrained-r14` | `d23472e9-567f-4bd3-9e04-24f450dbab85` | 4 | Running |
| `em-shwmse-maws-bcar-rank-mempretrained-r14` | `9494b4fa-f6e6-410c-90ba-052bb8e70d01` | 4 | Running, strongest current method |

Current dbMiM GPU occupancy is 12 H200/GPU slots for these three full R14
finetune jobs. Account-wide occupancy may be higher because unrelated STEM
packs can run under the same owner.

## 2026-06-22 R15 Active State

Latest pushed repo commit before the R15 skill update:
`1450886 Add postprocess architecture exploration experiments`.

The active R15 wave is testing whether EM-specific UNETR structure and
post-processing changes can improve the R14/R14q MA-dbMiM result while staying
inside the dbMiM pretraining framework.

Active 2-GPU `med-model` architecture jobs:

| run | UUID | setting | last observed note |
|---|---|---|---|
| `longaff-mempretrained-r15q` | `73a40fd4-287b-4bfc-98b6-c57aa6a38c1a` | 6-channel nearest+long-range affinity | training around step 5640/12000 on 2026-06-22 |
| `longaff-lsd-mempretrained-r15q` | `08fad37f-4257-4f4f-9e9b-ad86c4b7f93f` | 6 affinity + 4 LSD descriptor channels | training around step 5120/12000 on 2026-06-22 |
| `longaff-bcar2-mempretrained-r15q` | `cb58f241-4fac-490f-b958-1ca6376bfb14` | stronger BCAR rank term | training around step 5160/12000 on 2026-06-22 |

Common setting:

- data: CREMI A/B/C, crop `32x160x160`, synchronized augmentation, widened
  labels;
- model: `unetr_aniso_em`, patch `[4,16,16]`, embed dim `192`, depth `6`,
  heads `6`, feature size `32`, EM refinement depth `2`;
- pretrained: R14 MA-dbMiM membrane-pretrained checkpoint from
  `pretrain_em_membrane_dbmim_r14`;
- optimization: per-GPU batch `2`, 2 GPUs, `max_steps: 12000`, AMP,
  `lr: 8e-5`, `encoder_lr: 1e-5`;
- evaluation: post-train full-volume A/B/C architecture benchmark with CREMI
  boundary-ignore `xy=1,z=0`, `graph_cc`, `cupy_graph_cc`, `seeded_rag`, and
  `waterz`.

Standalone post-processing benchmark:

| run | UUID | status note |
|---|---|---|
| `eval-cremi-arch-explore-postprocess-r15q` first attempt | `bcac3b16-9896-4114-84bd-a70f854e2a8e` | stopped after missing `skimage` invalidated graph/RAG rows |
| `eval-cremi-arch-explore-postprocess-r15q` fixed rerun | `8ffeb749-2da8-4236-a46d-b60e9443598e` | running on 2 GPUs; startup probe showed `waterz=True`, `cupy=True` |

R15 reporting rule: separate training convergence from post-processing runtime.
The 12k-step training jobs can finish quickly on H200s, but full-volume
A/B/C architecture sweeps are dominated by CPU-heavy watershed/RAG/waterz and
can continue for hours after training reaches the final step.

## 2026-06-22 Public-EM R16 Pretraining

The gated HF dataset `cyd0806/EM_pretrain_data` still requires an authorized
HF token; without it only manifests are available. To avoid another
CREMI-only pretraining fallback, a public-EM bridge dataset was prepared and
uploaded to TOS:

```text
tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data/public_em/
```

Prepared HDF5 raw volumes:

| dataset | local/TOS file | shape |
|---|---|---|
| ISBI 2012 train ssTEM | `isbi2012_train_raw.h5` | `30x512x512` |
| ISBI 2012 test ssTEM | `isbi2012_test_raw.h5` | `30x512x512` |
| SNEMI3D train-input | `snemi3d_train_raw.h5` | `100x1024x1024` |
| SNEMI3D test-input | `snemi3d_test_raw.h5` | `100x1024x1024` |

Preparation script:

```bash
python scripts/prepare_public_em_pretrain_data.py --group public_em --upload-tos
```

Active R16 public-EM pretraining:

| run | UUID | GPUs | note |
|---|---:|---:|---|
| `pretrain-public-em-membrane-r16` | `5a10fe9e-2d34-4568-8009-5902c73cc592` | 8 | Succeeded on `med-model`, output `pretrain_public_em_membrane_dbmim_r16`, final `global_step=160000` |

Startup logs confirmed:

- `em_pretrain_data_status='available_offline_tos'`;
- four public-EM HDF5 files were present inside the pod;
- `dataset_size=57344`, `world_size=8`;
- training reached at least step `7760` with loss around `0.083`.

Downstream config prepared but should be submitted only after
`pretrained_latest.pt` is confirmed readable from TOS:

```text
configs/finetune_cremi_real_unetr_aniso_em_shwmse_longaff_publicem_r16q.yaml
```

Submit stage:

```text
finetune-cremi-unetr-aniso-arch-explore-longaff-publicem-r16q
```

## 2026-06-22 Latest Result Snapshot

R16 and R17 official A/B/C waterz-only evaluations have completed. All rows use
full CREMI A/B/C, `crop-size 0 0 0`, waterz `hist_quantile`, skimage
VOI/ARAND, and CREMI boundary-ignore `xy=1,z=0`. The poller summaries were
rebuilt from SiFlow stdout fallback, with `records=60` and sample names
`A/B/C` for every arm.

| arm | best VOI | ARAND at best VOI | best ARAND | interpretation |
|---|---:|---:|---:|---|
| R16 longaff + SHW-MSE + publicEM | `1.012079` | `0.181445` | `0.181445` | best ARAND so far |
| R16 longaff + SHW-MSE scratch | `1.036742` | `0.200020` | `0.193729` | publicEM helps |
| R16 MAWS+BCAR + SHW-MSE + publicEM | `1.026490` | `0.187259` | `0.186044` | publicEM helps |
| R16 MAWS+BCAR + SHW-MSE scratch | `1.062392` | `0.192804` | `0.192804` | weaker control |
| R17 MAWS + MSE + publicEM | `1.002919` | `0.188832` | `0.188832` | best VOI so far |
| R17 MAWS + MSE scratch | `1.095164` | `0.213401` | `0.210442` | publicEM strongly helps |
| R17 MAWS + MSE+BCAR + publicEM | `1.063910` | `0.197817` | `0.193443` | BCAR hurts here |
| R17 MAWS + MSE+BCAR scratch | `1.087732` | `0.200894` | `0.196075` | not better than pure MSE |

Key current conclusion: publicEM dbMiM pretraining now gives a stable
A/B/C improvement under matched controls, and the strongest downstream stack is
`UNETR-aniso-EM + MAWS + pure MSE + publicEM`. BCAR is not universally helpful;
it helped some SHW-MSE settings but hurt the best R17 MSE setting.

R18 was submitted to cross the best components:

| arm | UUID | current decision |
|---|---|---|
| long-affinity + MSE + publicEM | `353e6a63-4071-473e-9a67-e8aa1485be2d` | stopped on 2026-06-22 after A/B partial was clearly worse than scratch |
| long-affinity + MSE scratch | `acc136c7-b5d6-4e51-b500-d0f391247daa` | stopped after publicEM arm was killed; no longer a useful paired comparison |
| long-affinity + MSE+BCAR + publicEM | `deac4066-3261-4739-8887-f3d3c4629aae` | keep running |
| long-affinity + MSE+BCAR scratch | `dffb90bf-1526-4a8a-8a65-ee7d36065824` | keep running as matched control |

R18 watcher:

```text
/volume/med-train/users/dchen02/code/dbMiM/outputs/watchers/poll_r18q_20260622T095756.log
```

A fine waterz calibration sweep was also submitted for the current best R17
checkpoint:

| eval | UUID | note |
|---|---|---|
| `r17q_fine` MAWS+MSE publicEM | `4503d96c-9b52-4974-8e5e-7ee08bc21362` | 9 thresholds x 6 biases x A/B/C |

As of 2026-06-22 10:13 China time, `r17q_fine` was still running and only
sample A had appeared in stdout fallback. The poller now prints this as
`PARTIAL` and does not count it as done until samples A/B/C all appear. Do not
report sample-A-only fine rows as A/B/C.

Fine watcher:

```text
/volume/med-train/users/dchen02/code/dbMiM/outputs/watchers/poll_r17q_fine_20260622T100119.log
```

Poll commands:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r18q --once --logs --siflow-fallback

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r17q_fine --once --logs --siflow-fallback
```

R19 was submitted after R18 to test structure capacity without mixing in
long-affinity targets. It keeps the current best R17 recipe
(`UNETR-aniso-EM + MAWS + pure MSE`) and changes only context size or decoder
width, each with a matched publicEM/scratch pair. All are 2-GPU `med-model`
jobs with post-train official A/B/C waterz eval:

| arm | UUID | config |
|---|---|---|
| context `48x192x192` + publicEM | `e9e01802-c98e-466b-b3cf-f5cf1b0edbbd` | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_context48_publicem_r19q.yaml` |
| context `48x192x192` scratch | `6655b114-66b7-4a66-8efc-d55ca6d2dfcc` | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_context48_scratch_r19q.yaml` |
| decoder `feature_size=48` + publicEM | `77d049b8-76a1-4369-84c7-d02bc361851e` | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_fs48_publicem_r19q.yaml` |
| decoder `feature_size=48` scratch | `2ade7cb2-667c-4410-b25d-cba312fc112e` | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_fs48_scratch_r19q.yaml` |

R19 uses per-GPU batch 1 and 12k steps to fit the larger crop/decoder on
2-GPU H200 jobs. Poll with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r19q --once --logs --siflow-fallback
```

## 2026-06-22 Cleanup Snapshot

After the R16/R17 waterz-only A/B/C results and early R18/R19 partial rows,
stale jobs were stopped to free GPUs. These should not be resumed unless a
specific missing artifact is needed:

| reason | UUIDs |
|---|---|
| R15 mempretrained variants superseded by publicEM R16/R17/R18/R19 | `73a40fd4-287b-4bfc-98b6-c57aa6a38c1a`, `08fad37f-4257-4f4f-9e9b-ad86c4b7f93f`, `cb58f241-4fac-490f-b958-1ca6376bfb14` |
| R15 standalone postprocess benchmark superseded by waterz-only official A/B/C paths | `8ffeb749-2da8-4236-a46d-b60e9443598e` |
| R16 arch-bench jobs superseded by standalone R16 waterz-only complete summaries | `11df9593-273a-4456-8da4-6f844b1d8292`, `198e39cc-c2aa-4ab1-82f3-edcc15e6917f`, `b8e5f3dc-9047-48a2-9b90-dd439f0265c7`, `997ba3ec-77f6-41f0-851b-83f770427cfd` |
| R18 pure long-affinity + MSE pair stopped after bad publicEM A/B partial | `353e6a63-4071-473e-9a67-e8aa1485be2d`, `acc136c7-b5d6-4e51-b500-d0f391247daa` |

Remaining active dbMiM tasks immediately after cleanup: seven tasks / 13 GPUs.
They are R18 BCAR publicEM/scratch, R19 four structure-capacity arms, and R17
fine calibration. Before launching new jobs, query live SiFlow state again
because stopped tasks can linger briefly as `Stopping`.

## 2026-06-22 Late R20/R21 End-to-End Postprocess Snapshot

At 2026-06-22 23:12 China time, live SiFlow state had changed substantially:
most older dbMiM work had completed or been stopped. Four dbMiM jobs were
running on `med-model`: two R20 downstream official-eval pods, one R20+DPP pod,
and one R21 decoder-aware pretrain pod. Always re-query live state before
reporting GPU usage.

R20 full-EM pretraining checkpoint status:

- UUID `7be2f62b-c1f3-482a-85a7-74cd63c63c35`, `med-model`, 4 GPUs.
- Final task status is `Failed`, not `Succeeded`.
- Failure happened after useful training: TOS contains
  `checkpoint_step_00060000.pt`, `pretrained_latest.pt`, and `train_log.jsonl`
  under
  `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_full_membrane_dbmim_r20/`.
- Treat R20 as a 60k-step full-EM encoder-only checkpoint, not a completed
  240k-step pretraining run. It is still valid for fast downstream signal, but
  do not overstate convergence.

R20 downstream status:

| arm | UUID | state at 23:10 CST | note |
|---|---|---|---|
| MSE+MAWS + R20 full-EM | `0e29a6b1-26bb-45c2-813d-db8efb266d21` | Running | finetune reached step 12000, loss `0.021585`; post-train official A/B/C waterz not uploaded yet |
| MSE+MAWS+BCAR + R20 full-EM | `eb647c53-f408-4a5f-99c9-ead3d7b1f2df` | Running | finetune reached step 12000, loss `0.025563`; post-train official A/B/C waterz not uploaded yet |

Poll command:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r20q --once --logs --siflow-fallback
```

The poller showed the matched R17 scratch summaries are complete A/B/C:
`MSE+MAWS scratch VOI=1.095164`, `bestARAND=0.210442`; and
`MSE+MAWS+BCAR scratch VOI=1.087732`, `bestARAND=0.196075`. Do not claim R20
pretraining gain until the two R20 official A/B/C summaries appear.

New end-to-end postprocess and decoder-aware jobs:

| purpose | stage/config | UUID | GPUs | status at submit |
|---|---|---:|---:|---|
| learned differentiable postprocess on R20 full-EM | `finetune-cremi-unetr-aniso-arch-explore-maws-mse-dpp-fullem-r20q` / `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_dpp_fullem_r20q.yaml` | `e52d773c-abae-481f-9da7-c3b34cab38a8` | 2 | Running; started `2026-06-22T15:09:22Z` |
| decoder-aware full-EM dbMiM pretraining | `pretrain-em-full-decoderaware-r21` / `configs/pretrain_em_full_decoderaware_r21.yaml` | `388347d2-3a33-4cd3-ab65-78b6d6ab2949` | 4 | Running; started `2026-06-22T15:11:13Z` |

R21 pretraining uses the same five gated HF groups (`fafb`, `fib25`,
`kasthuri`, `mitoem`, `mb_moc`) and the same mounted staging path:

```text
/volume/med-train/users/dchen02/code/dbMiM_runtime/em_pretrain_data/full_r20/all
```

The submitter now skips TOS copy for a group if HDF5 files are already staged
under that mounted path. This avoids re-copying hundreds of GB for R21 after
R20 staged the same data. The run still has a hard gate: if any required group
is absent after the copy/skip step, it exits 21 and does not silently fall back
to CREMI-only.

## 2026-06-23 00:12 China R21 Watcher Snapshot

At `2026-06-23T00:04-00:12+08:00`, the reliable live state was:

- dchen02 had 5 unique running/queued tasks using 18 GPUs total.
- dbMiM used 10 GPUs total, all on `med-model`:
  - R21 decoder-aware pretrain `388347d2-3a33-4cd3-ab65-78b6d6ab2949`, 4 GPUs.
  - R20 MSE full-EM downstream `0e29a6b1-26bb-45c2-813d-db8efb266d21`, 2 GPUs.
  - R20 BCAR full-EM downstream `eb647c53-f408-4a5f-99c9-ead3d7b1f2df`, 2 GPUs.
  - R20 DPP full-EM downstream `e52d773c-abae-481f-9da7-c3b34cab38a8`, 2 GPUs.
- R21 `train_log.jsonl` had reached step `42100` by the first stable watcher
  poll. Earlier direct probes showed step `36120` loss `0.1070` and step
  `40620` soon after; the run was progressing normally.
- R20 DPP finetune had completed supervised step `12000`, last logged
  `train_loss=0.024395`, `train_main_loss=0.022607`. Its official A/B/C
  evaluation prefix was still empty, so no VOI/ARAND conclusion was available.
- R20 MSE and R20 BCAR had both completed supervised step `12000`, but their
  official eval prefixes were also empty. The current bottleneck remained
  CPU-heavy waterz/post-processing inside the post-train evaluation pods.

A stable R21 downstream watcher is running in screen:

```bash
screen -ls
tail -n 40 "$(cat outputs/watchers/r21_decoderaware_finetune_screen.latest_log)"
```

The watcher uses:

```bash
scripts/watch_and_submit_r21_decoderaware_finetune.sh
```

It waits for R21 `train_log.jsonl` to reach `DBMIM_R20_MIN_STEP=80000` before
submitting four 2-GPU downstream jobs with post-train official A/B/C waterz
evaluation:

- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-scratch-r21q`
- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-decoderaware-r21q`
- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-dpp-scratch-r21q`
- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-decoderaware-dpp-r21q`

This gives a matched `scratch/pretrained x DPP/no-DPP` comparison after full
decoder-aware pretraining, rather than judging an early 40k/60k checkpoint.

At `2026-06-23T00:17+08:00`, R20 official A/B/C results began landing.
At `2026-06-23T00:34+08:00`, the DPP summary completed and corrected the
earlier partial-read impression:

| arm | source | records | VOI sum | best ARAND | note |
|---|---|---:|---:|---:|---|
| R20 full-EM MSE+MAWS | TOS summary/stdout | 60 | `1.085331` | `0.195722` | small gain vs R17 scratch VOI `1.095164`, ARAND gain larger |
| R20 full-EM MSE+MAWS+BCAR | TOS summary/stdout | 60 | `1.179617` | `0.236163` | worse than matched scratch; do not pursue BCAR here |
| R20 full-EM MSE+MAWS+DPP | TOS summary/stdout | 60 | `1.123336` | `0.210163` | final complete result is worse than no-DPP and not better than scratch |

Important pitfall: an intermediate stdout fallback with only 45 records showed
`VOI=0.798560` and `ARAND=0.108556`, but that was a partial sweep and reversed
after all 60 records appeared. Do not use partial waterz stdout rows for method
claims unless the record count and sample coverage match the intended grid.

The useful current hypothesis is therefore `full-EM pretraining + pure
MSE/MAWS` with no BCAR and no current DPP claim. The R21 watcher still tests
decoder-aware pretraining and the DPP controls under a matched design, but DPP
should be treated as exploratory rather than a current positive result.

At `2026-06-23T00:34+08:00`, only one dbMiM SiFlow task remained active:

- R21 decoder-aware pretrain `388347d2-3a33-4cd3-ab65-78b6d6ab2949`, 4 GPUs,
  `med-model`.
- Latest TOS `train_log.jsonl`: step `56940/80000`, loss `0.131452`,
  affinity loss `0.011736`, elapsed `4911s`.
- The screen watcher had reached poll 8 and was waiting for step `80000` before
  submitting the four R21 downstream arms.

At `2026-06-23T00:40+08:00`, the old R21 watcher was stopped and restarted with
only two downstream stages:

- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-scratch-r21q`
- `finetune-cremi-unetr-aniso-arch-explore-maws-mse-decoderaware-r21q`

The DPP R21 arms were deliberately removed from the automatic watcher because
the completed R20 DPP result was not positive. The current watcher log is still
tracked by:

```bash
cat outputs/watchers/r21_decoderaware_finetune_screen.latest_log
```

At `2026-06-23T00:48+08:00`, a replacement learned-postprocess experiment was
submitted:

- Stage: `eval-cremi-learned-rag-r20q`
- UUID: `57808e2d-e0e8-43cb-88d2-105ab58641ab`
- Pool/GPUs: `med-model`, 2 GPUs.
- Status at submit check: `Pending`.

This is not the old DPP affinity calibrator. It trains
`LearnedRAGMergeScorer`, a lightweight MLP over watershed fragment-pair
boundary features, and then merges fragment pairs once by learned probability.
The intended comparison is:

- quality: learned-RAG VOI/ARAND vs R20 waterz `VOI=1.085331`,
  `bestARAND=0.195722`;
- speed: learned-RAG `postprocess_sec` vs waterz `postprocess_sec`;
- generalization: A/B train-like rows vs C holdout rows in
  `learned_rag_summary.json`.

Treat learned-RAG as the current "可学习后处理" branch. It can plausibly speed
up inference because it replaces waterz agglomeration/sweep with one learned
edge-score pass plus union-find, although 2D watershed fragment generation is
still CPU-bound in this first version.

## 2026-06-23 01:20 China R21 and Learned-RAG Update

R21 decoder-aware pretraining completed its 80k-step trigger point and the
watcher submitted the two retained downstream arms. The old DPP downstream arms
were intentionally removed before this submission because the completed R20 DPP
result was negative.

R21 pretraining:

- UUID: `388347d2-3a33-4cd3-ab65-78b6d6ab2949`
- Status: `Succeeded`
- Output prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_full_decoderaware_dbmim_r21/`
- Final TOS `train_log.jsonl`: `step=80000`, `loss=0.130470`,
  `pixel_loss=0.106881`, `structure_loss=0.033845`,
  `affinity_loss=0.052893`, elapsed about `6884s`.

The R21 watcher log is:

```text
outputs/watchers/r21_decoderaware_nodpp_watch_20260622T163956Z.log
```

It submitted:

| arm | UUID | status at 2026-06-23 01:20 CST | GPUs | pool |
|---|---|---|---:|---|
| R21 scratch no-DPP | `dae6bec4-3dc8-46a0-b7af-6a2f39619be2` | Running | 2 | `med-model` |
| R21 decoder-aware pretrained no-DPP | `89a29ac5-03d5-4c85-8e86-55ef8ad19f52` | Running | 2 | `med-model` |

At that time these two R21 finetune/eval jobs were the only active dbMiM
SiFlow jobs found by `tasks.list` for running/pending/queueing states, so dbMiM
occupied 4 GPUs. Re-query before reporting because jobs may have moved into
post-train CPU-heavy waterz evaluation or finished.

Poll after they finish with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r21q --once --logs --siflow-fallback
```

The fixed learned-RAG rerun also completed:

- Failed first attempt: `57808e2d-e0e8-43cb-88d2-105ab58641ab`; import path
  bug, `ModuleNotFoundError: No module named 'scripts.evaluate_cremi_segmentation'`.
- Fixed attempt: `c6416133-74d8-421b-936d-676a87894dbc`; status `Succeeded`.
- Output prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_learned_rag_r20q/`
- Artifacts: `learned_rag_summary.json`, `learned_rag_records.json`,
  `learned_rag_scorer.pt`.

Complete learned-RAG result:

| split | best threshold | VOI sum | ARAND | postprocess sec | note |
|---|---:|---:|---:|---:|---|
| A/B/C all records (`n=3`) | `0.7` | `7.025362` | `0.963295` | `9.40` | far worse than R20 waterz |
| C holdout (`n=1`) | `0.7` | `6.590155` | `0.932134` | `9.77` | does not generalize |

Training set pair stats were large enough for a real probe:
`train_pairs=3164125`, positive fraction about `0.668`. The failure is
therefore not a missing-output or no-training issue. The first learned-RAG
scorer over simple boundary features is too aggressive/poorly calibrated for
CREMI instance segmentation, merging many true boundaries. Keep this branch as
a negative baseline and as scaffolding for a more principled learnable
postprocess, but do not present it as a positive method.

Current method ranking from complete official-style results:

| method | complete result | conclusion |
|---|---|---|
| R17 publicEM MSE+MAWS | VOI `1.002919`, ARAND `0.188832` | best positive pretraining signal so far |
| R20 full-EM MSE+MAWS | VOI `1.085331`, ARAND `0.195722` | small gain over R17 scratch but worse than R17 publicEM |
| R17 scratch MSE+MAWS | VOI `1.095164`, ARAND `0.210442` | main scratch reference |
| R20 full-EM DPP | VOI `1.123336`, ARAND `0.210163` | negative; stop automatic DPP expansion |
| R20 full-EM BCAR | VOI `1.179617`, ARAND `0.236163` | negative |
| R20 learned-RAG one-pass merge | VOI `7.025362`, ARAND `0.963295` | strong negative |

The next useful direction is not another simple edge MLP over frozen fragment
statistics. Better candidates are: improve affinity prediction and calibration
inside the UNETR path; keep waterz/elf as the quality reference; and only
replace post-processing with a learned method if it preserves monotonic merge
constraints or optimizes a segmentation-aware loss on held-out volumes.

## 2026-06-23 01:50 China R22 and Learnable Calibration Wave

The user requested continued full-data pretraining and learnable post-processing
with a hard cap of 16 total GPUs. Live SiFlow state before submission was 4
dbMiM GPUs: the two R21 downstream jobs, each 2 GPUs. The new wave deliberately
uses 10 more GPUs, for 14 total dbMiM GPUs:

| purpose | UUID | GPUs | pool | status at submit check |
|---|---|---:|---|---|
| R22 full-EM decoder-aware pretrain continuation | `f54b2ac9-81e2-407c-b39b-85b22deab40b` | 8 | `med-model` | Running |
| R20 learned affinity calibration postprocess | `db054938-053d-4d25-9849-d616d66ff57e` | 2 | `med-model` | Pending, then startup logs appeared |
| R21 scratch downstream | `dae6bec4-3dc8-46a0-b7af-6a2f39619be2` | 2 | `med-model` | Running |
| R21 decoder-aware downstream | `89a29ac5-03d5-4c85-8e86-55ef8ad19f52` | 2 | `med-model` | Running |

R22 details:

- Stage: `pretrain-em-full-decoderaware-r22`
- Config: `configs/pretrain_em_full_decoderaware_r22.yaml`
- Output prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_full_decoderaware_dbmim_r22/`
- It resumes from:
  `tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_full_decoderaware_dbmim_r21/pretrained_latest.pt`
- R21 checkpoint size was verified on TOS before submission: about `213.04MB`.
- R22 target `max_steps=160000`, LR `1e-4`, starting from R21 step 80000.
- Startup logs showed all five full-EM groups already staged under the mounted
  path and `em_pretrain_data_status='available_offline_tos'`. The R21
  checkpoint downloaded successfully, and `torchrun` launched. Wait for
  `train_log.jsonl` or `pretrained_latest.pt` under the R22 output prefix
  before reporting loss.

Learnable postprocess details:

- Stage: `eval-cremi-learned-affinity-calibration-r20q`
- Script: `scripts/train_learned_affinity_calibration.py`
- Output prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_learned_affinity_calibration_r20q/`
- It uses the R20 full-EM MSE+MAWS finetuned checkpoint and learns only a
  differentiable global z/y/x affinity scale+bias on CREMI A/B affinity targets.
  Final instance segmentation still uses waterz on A/B/C. This is intentionally
  more conservative than learned-RAG: it cannot arbitrarily merge fragment
  pairs, so it tests whether learnable calibration can improve waterz sweeps
  without destroying topology.
- Expected artifacts: `learned_affinity_calibration.json`,
  `cremi_segmentation_summary.json`, and `cremi_segmentation_records.json`.
- Poll with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r20q_learned_calib --once --logs --siflow-fallback
```

Do not submit more dbMiM GPU work while these four tasks are active unless a
task finishes or is stopped; the user cap is 16 GPUs and the current planned
occupancy is 14.

## 2026-06-23 02:30 China Scale-aware Blockwise/Sparse-edge Wave

The user asked to improve for MICRONS/CAVE-scale inference and add 4 GPUs for
experiments. Implemented and committed:

- `scripts/evaluate_cremi_blockwise_scale.py`: chunked+halo inference on CREMI
  crops, full/seam/nonseam VOI/ARAND, throughput, and RAG edge-density stats.
- `scripts/train_sparse_edge_postprocess.py`: sparse learned RAG edge scorer
  with deterministic affinity-score baselines and held-out sample-C reporting.
- `scripts/submit_siflow_dbmim.py` stage:
  `eval-cremi-blockwise-scale-r17q`.

Submission:

| purpose | UUID | GPUs | pool | status at submit check |
|---|---|---:|---|---|
| R17 blockwise scale + sparse-edge pair | `68c11da3-6abc-4244-b50e-8ece35d5aa2f` | 4 | `med-model` | Queueing |

The 4-GPU pod runs four single-GPU processes:

| GPU | arm |
|---:|---|
| 0 | R17 publicEM MSE+MAWS blockwise/seam eval |
| 1 | R17 scratch MSE+MAWS blockwise/seam eval |
| 2 | R17 publicEM MSE+MAWS sparse learned-edge postprocess |
| 3 | R17 scratch MSE+MAWS sparse learned-edge postprocess |

Output prefixes after success:

```text
tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_blockwise_scale_r17q/
tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_sparse_edge_r17q/
```

The stage intentionally avoids waterz to keep the experiment focused on
scale-aware blockwise inference and sparse post-processing. It still installs
`scikit-image` and `mahotas` from offline wheelhouses, but does not build
waterz.

At the status check immediately after submission, active/queued dbMiM GPU
accounting was: 10 actually running GPUs (`R22` 8 + R21 decoder-aware 2) and
4 queued GPUs for the new scale-aware job. The previously active R20 learned
affinity calibration and R21 scratch downstream jobs had succeeded.

The R20 learned affinity calibration result is a negative/no-gain result:

| selection | variant | VOI | ARAND | note |
|---|---|---:|---:|---|
| best A/B/C by VOI | raw `pred` | `1.108290` | `0.202786` | worse than R20 raw baseline `1.085331` |
| best A/B/C by ARAND | raw `pred` | `1.144105` | `0.198023` | learned calibrated variant not selected |
| held-out C by VOI | raw `pred` | `1.750963` | `0.413981` | no held-out gain |

Learned scale/bias were approximately:

```text
scale = [0.7962, 0.8303, 0.8072]
bias  = [0.0600, 0.5843, 0.6307]
```

Conclusion: simple learnable z/y/x calibration is safer than learned-RAG, but
it did not improve the current R20 full-EM checkpoint. The next useful learned
postprocess evidence is the sparse-edge R17 publicEM vs scratch wave above.

## 2026-06-23 R23 Plain-MAE Control Baseline

The user correctly challenged that the current `R17 publicEM dbMiM` gain is
small and does not yet prove a dbMiM-specific contribution. Before R23, there
was no clean completed `plain MAE pretrain -> same UNETR finetune -> same
official A/B/C waterz` baseline. Existing evidence only showed:

- `publicEM dbMiM` vs scratch, not dbMiM vs ordinary MAE;
- several dbMiM objective/postprocess variants, most of which were negative or
  weak;
- R22 full-EM decoder-aware pretraining checkpoint, but not yet a matched
  plain-MAE control.

R23 adds the required control. Plain MAE is intentionally conservative:

- same EM/CREMI data and same `32x160x160` crop as dbMiM pretraining;
- `DBMIM3DMAE` encoder-only architecture with random 75% patch mask;
- masked voxel reconstruction only;
- `structure_weight: 0.0`;
- `membrane_weight: 0.0`;
- `decision.enabled: false`;
- no affinity proxy, no membrane weighting, no decision-mask policy, no UNETR
  decoder/head pretraining.

This isolates whether the current gain is just generic MAE initialization.
The decision rule is:

| outcome | interpretation |
|---|---|
| plain MAE ≈ or better than dbMiM | dbMiM-specific claim is weak; use "MAE-style EM pretraining helps" instead |
| dbMiM > plain MAE > scratch | membrane/structure/dbMiM objective has a real contribution |
| dbMiM > scratch but plain MAE ≈ scratch | dbMiM objective is the main source of transfer |
| all pretrained arms ≈ scratch | current gain is not stable enough for a method claim |

Configs and stages:

| purpose | config | stage/output |
|---|---|---|
| publicEM plain MAE pretrain | `configs/pretrain_public_em_plain_mae_r23.yaml` | `pretrain-public-em-plain-mae-r23` / `pretrain_public_em_plain_mae_r23` |
| publicEM plain MAE finetune | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_plainmae_r23q.yaml` | `finetune-cremi-unetr-aniso-arch-explore-maws-mse-publicem-plainmae-r23q` |
| fullEM plain MAE pretrain | `configs/pretrain_em_full_plain_mae_r23.yaml` | `pretrain-em-full-plain-mae-r23` / `pretrain_em_full_plain_mae_r23` |
| fullEM plain MAE finetune | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_plainmae_r23q.yaml` | `finetune-cremi-unetr-aniso-arch-explore-maws-mse-fullem-plainmae-r23q` |

Submission state and updates:

| purpose | UUID | GPUs | pool | state |
|---|---|---:|---|---|
| publicEM plain MAE pretrain | `7e2aa43b-48b8-40ad-bba7-b04b48fe2f60` | 4 | `med-model` | Succeeded; 160k checkpoint and `pretrained_latest.pt` uploaded |
| publicEM plain MAE downstream | `2c848e2e-7e75-425c-8bd0-215f979f70c2` | 2 | `med-model` | Running at 2026-06-23 10:40 CST; `finetune_log.jsonl` and `finetuned_latest.pt` visible on TOS |
| fullEM plain MAE pretrain, first try | `f428bd1e-83bf-4d7e-9038-5fe497eb5847` | 4 | `med-model` | Stopped; actual quota was 3/4 |
| fullEM plain MAE pretrain, shared retry | `9658bbd2-cf61-4a54-a392-a014ee488114` | 4 | `cn-shanghai-changliu-skyinfer-reserved-shared` | Queueing at 2026-06-23 10:40 CST; actual quota still 0/4 |

The publicEM plain-MAE logs confirmed it is a true plain-MAE control:

```text
affinity_loss = 0.0
membrane_weight_mean = 1.0
policy_loss = 0.0
mask_ratio = 0.75
loss = pixel_loss
```

Use the fair 160k-step checkpoint for the main comparison, because the R16
publicEM dbMiM checkpoint also ran to 160k. Do not compare a 40k early plain
MAE checkpoint against the 160k dbMiM result except as an explicitly labeled
early-signal table.

Watchers:

```bash
tail -n 50 outputs/watchers/plain_mae_r23_watcher.log
tail -n 50 outputs/watchers/plain_mae_full_r23_watcher.log
```

They wait for `pretrained_latest.pt` and `train_log.jsonl` with max step at
least `160000`, then submit the corresponding 2-GPU finetune job with
post-train official A/B/C waterz evaluation. Poll final publicEM plain-MAE
comparison with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r23_plainmae --once --logs --siflow-fallback
```

Poll the fullEM plain-MAE comparison with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r23_plainmae_full --once --logs --siflow-fallback
```

## 2026-06-23 R24 dbMiM++ vs Plain MAE Target

The user set the hard target: dbMiM must show a gain over ordinary MAE, not
only over scratch. R24 is the current publicEM experiment for that exact
claim.

Method definition:

- same publicEM+CREMI pretraining data as R23 plain MAE;
- same crop `32x160x160`, mask ratio `0.75`, and 160k-step target;
- `DecoderAwareDBMIM3DMAE`, so pretraining initializes encoder, UNETR-EM
  decoder, and affinity head;
- masked reconstruction + anisotropic gradient structure loss;
- membrane-weighted reconstruction;
- membrane-weighted pseudo-affinity decoder/head loss via
  `affinity_membrane_weight`;
- decision masking enabled and frozen after 40k steps.

Configs and stages:

| purpose | config | stage/output |
|---|---|---|
| R24 publicEM dbMiM++ pretrain | `configs/pretrain_public_em_decoderaware_r24.yaml` | `pretrain-public-em-decoderaware-r24` / `pretrain_public_em_decoderaware_dbmim_r24` |
| R24 downstream | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_decoderaware_r24q.yaml` | `finetune-cremi-unetr-aniso-arch-explore-maws-mse-publicem-decoderaware-r24q` |

Validation before submit:

- `py_compile` passed for `dbmim/models.py`, `train_pretrain.py`,
  `train_finetune.py`, `scripts/submit_siflow_dbmim.py`, and
  `scripts/poll_dbmim_tos_results.py`;
- local R24 forward on `1x1x32x160x160` produced finite pixel, structure, and
  affinity losses;
- simulated loading into downstream `UNETR-aniso-EM` loaded 214 compatible
  keys, including encoder blocks, decoder modules, and head keys.

Active state at 2026-06-23 10:40 CST:

| purpose | UUID/process | GPUs | pool | state |
|---|---|---:|---|---|
| R24 publicEM dbMiM++ pretrain | `bd588c91-6328-455e-a1bc-fc1a3316bbdd` | 4 | `med-model` | Running |
| R24 watcher | PID from `outputs/watchers/public_em_decoderaware_r24_watcher.pid` | 0 | login node | Waiting for `checkpoint_step_00160000.pt` and max step 160000 |

Watcher log:

```bash
tail -n 50 outputs/watchers/public_em_decoderaware_r24_watcher.log
```

Final comparison:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r24_dbmim_vs_mae --once --logs --siflow-fallback
```

Update at 2026-06-23 15:40 CST:

- R23 publicEM plain-MAE downstream completed. Its official A/B/C waterz
  summary has 60 records and `n=3` for best rows.
- R16 publicEM dbMiM is slightly better than R23 plain MAE, but the margin is
  small:

| arm | VOI sum | ARAND at best VOI | best ARAND | note |
|---|---:|---:|---:|---|
| R16 publicEM dbMiM + MSE+MAWS | `1.002919` | `0.188832` | `0.188832` | current dbMiM baseline |
| R23 publicEM plain MAE + MSE+MAWS | `1.027073` | `0.192763` | `0.189247` | matched plain-MAE control |
| R17 scratch + MSE+MAWS | `1.095164` | `0.213401` | `0.210442` | matched scratch |

Current defensible claim: publicEM dbMiM has a real but weak gain over plain
MAE: about `-0.0242` VOI (`2.35%` relative) and `-0.0004` best ARAND. This is
not yet a strong paper-level method gain.

R24 publicEM dbMiM++ pretrain failed at step `74780/160000`, after uploading
`checkpoint_step_00074000.pt`. The failure was not a main-loss explosion; SiFlow
logs show finite loss near failure and then `DecisionModule` produced NaN
categorical logits after the policy freeze point. The code was patched to:

- sanitize `DecisionModule` hidden state, logits, and value with
  `torch.nan_to_num` and logit/value clipping;
- stop using the frozen policy for mask sampling by default after
  `freeze_after_steps` (`use_frozen_policy_after_freeze: false`);
- add resume stage `pretrain-public-em-decoderaware-r24-resume74`.

Resume task:

| purpose | UUID/process | GPUs | pool | state |
|---|---|---:|---|---|
| R24 resume from 74k | `363306f7-e09d-4f07-8ccb-e24735d0fcd5` | 4 | `med-model` | Submitted at 2026-06-23 15:39 CST |
| R24 watcher restart | PID from `outputs/watchers/public_em_decoderaware_r24_watcher.pid` | 0 | login node | Waiting for `checkpoint_step_00160000.pt` |

Do not report R24 as a method result until this resume completes and the
watcher-submitted downstream official A/B/C summary appears.

Update at 2026-06-23 22:50 CST:

- R24 resume `363306f7-e09d-4f07-8ccb-e24735d0fcd5` completed the 160k-step
  publicEM dbMiM++ pretraining. TOS now contains
  `checkpoint_step_00160000.pt`, `pretrained_latest.pt`, and `train_log.jsonl`
  under `pretrain_public_em_decoderaware_dbmim_r24`.
- Final R24 pretrain log row near step 160000 was finite:
  `loss=0.0520`, `pixel_loss=0.0352`, `structure_loss=0.0840`,
  `affinity_loss=9.25e-05`, `mask_ratio=0.75`.
- The old watcher process had exited after one `checkpoint_wait`, so the R24
  downstream was submitted manually with post-train official A/B/C waterz eval.
  UUID: `628faa9d-4e5a-4b19-98bd-555bef604302`, 2 GPUs, `med-model`.
- R24 downstream loaded `214` pretrained keys from
  `outputs/pretrain_public_em_decoderaware_dbmim_r24/pretrained_latest.pt`.
  At about 22:47 CST it was running normally around step `5800/12000`, with
  recent MSE+MAWS train losses mostly in the `0.03-0.06` range.

R25 was added to test whether dbMiM's gain over MAE is larger in a
label-efficient / early-finetune regime. It keeps the same UNETR-aniso-EM,
MSE+MAWS loss, augmentation, and official A/B/C waterz evaluation, but stops
finetuning at 3000 steps. All four arms use seed `250` to reduce augmentation
noise:

| arm | config | UUID | GPUs/pool |
|---|---|---|---:|
| R24 dbMiM++ early3k | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_decoderaware_r24q.yaml` | `191db4dd-9949-4e08-8b79-838b34c19755` | 2, `med-model` |
| R23 plain MAE early3k | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_plainmae_r23q.yaml` | `d650fd4d-4447-44dc-b25e-64c3cb6b65d0` | 2, `med-model` |
| R16 dbMiM early3k | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_r17q.yaml` | `2b6dda20-de13-4ca4-af18-05af44a9988f` | 2, `med-model` |
| scratch early3k | `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_scratch_r17q.yaml` | `ffb9c6a4-935a-4dc4-aee9-87cf62d36c89` | 2, `med-model` |

Poll R25 with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r25_early3k --once --logs --siflow-fallback
```

GPU accounting at submission time: R24 full downstream plus four R25 early3k
arms were running on `med-model`, total 10 GPUs. FullEM plain-MAE pretrain
`9658bbd2-cf61-4a54-a392-a014ee488114` was still queueing in the shared pool
with `实例配额不足 | 需求:4, 实际可用(实例配额):0`, so the active plus queued
dbMiM work was within the user's 16-GPU cap.

Update at 2026-06-23 23:25 CST:

R24/R25 were still in CPU-bound official waterz post-processing on sample A;
the complete A/B/C summaries had not appeared. To diagnose whether R24's
decoder/head transfer is hurting downstream finetuning, R26 adds an
encoder-only load path for the same R24 checkpoint:

- code: `load_pretrained_backbone(..., include_prefixes=..., exclude_prefixes=...)`;
- full config:
  `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_decoderaware_encoderonly_r26q.yaml`;
- early3k config:
  `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_decoderaware_encoderonly_r26q.yaml`;
- full stage:
  `finetune-cremi-unetr-aniso-arch-explore-maws-mse-publicem-decoderaware-encoderonly-r26q`;
- poll groups: `r26_encoderonly` and `r26_encoderonly_early3k`.

The R26 configs load only `pos_embed`, `patch_embed`, `encoder_blocks`, and
`norm` from `pretrain_public_em_decoderaware_dbmim_r24/pretrained_latest.pt`.
They deliberately do not load decoder or affinity-head weights. This isolates
the R24 dbMiM++ encoder representation from pseudo-affinity decoder/head
initialization. A local smoke check with a synthetic checkpoint confirmed that
`head.*` keys are skipped when the include-prefix filter is set.

R26 full 12k was submitted on 2026-06-23 23:22 CST:

| purpose | UUID | GPUs | pool | state at submit |
|---|---|---:|---|---|
| R26 encoder-only full | `faba607c-b53c-4e45-9de2-798eb8462612` | 2 | `med-model` | Pending |

At that time unique dbMiM GPU accounting was 16 GPUs including this R26 task:
R24 full 2, R25 early3k four arms 8, R26 full 2, and fullEM plain-MAE queue 4.
Do not submit another GPU job until one of these completes or is stopped.

Update at 2026-06-23 23:50 CST:

The poller was patched again so official A/B/C summaries are not counted as
complete merely because sample C has appeared once. For `official_abc` evals,
`_is_complete_summary` now requires all three samples and at least 60 records.
This avoids mislabeling partial stdout fallback rows such as 41/60 or 54/60 as
complete.

Latest partial signals:

- R24 full-init downstream has completed training (`step=12000`,
  `loss=0.02568`) and is in C-sample waterz post-processing. Fallback has
  47/60 rows across A/B/C, but it is still `PARTIAL`. Current partial best VOI
  is about `1.158`, worse than both R16 dbMiM (`1.003`) and R23 plain MAE
  (`1.027`). Wait for 60/60 before making the final call, but the trend does
  not support transferring the R24 decoder/head.
- R25 early3k remains partial. At 41-54/60 rows, R24 full-init early3k is much
  worse than the other early3k arms; R16 dbMiM early3k and scratch early3k are
  close, and plain MAE early3k is also weak. Do not use this as final
  label-efficient evidence until all four arms reach 60/60.
- R26 encoder-only full has trained normally and started official A/B/C eval.
  It loaded exactly 77 keys with prefix counts
  `{'encoder_blocks': 72, 'norm': 2, 'patch_embed': 2, 'pos_embed': 1}` and
  no decoder/head keys. Its current sample-A-only fallback is strong
  (`VOI≈0.245`, `ARAND≈0.040` at 7 records), but this is only sample A and
  must not be compared to A/B/C aggregates.

Current best hypothesis: R24's pseudo-affinity decoder/head transfer is likely
harmful; the cleaner test is R26 encoder-only. If R26 A/B/C beats R23 plain MAE
and R16 dbMiM, use encoder-only R24 as the method path. If R26 also fails, stay
with R16 membrane dbMiM and focus on full-data pretraining or a better
pretext-objective rather than transferring the decoder/head.

Update at 2026-06-24 00:05 CST:

R24 full-init completed and is a negative result:

| arm | records | VOI | ARAND at best VOI | best ARAND |
|---|---:|---:|---:|---:|
| R24 dbMiM++ full-init | 60 | `1.482749` | `0.281327` | `0.275869` |
| R23 plain MAE | 60 | `1.027073` | `0.192763` | `0.189247` |
| R16 dbMiM | 60 | `1.002919` | `0.188832` | `0.188832` |
| scratch | 60 | `1.095164` | `0.213401` | `0.210442` |

This confirms that transferring the R24 pseudo-affinity decoder/head is harmful
under the maintained UNETR-aniso-EM + MSE+MAWS downstream recipe.

R25 early3k has completed the two main pretrained controls:

| early3k arm | records | VOI | ARAND at best VOI | best ARAND |
|---|---:|---:|---:|---:|
| R23 plain MAE early3k | 60 | `1.566102` | `0.389115` | `0.374662` |
| R16 dbMiM early3k | 60 | `1.550130` | `0.338732` | `0.338732` |

This is a small VOI gain over plain MAE and a clearer ARAND gain, but the
absolute 3k performance is still weak. R24 full-init early3k and scratch
early3k were still partial at this timestamp; R24 full-init early3k was
already clearly poor.

R26 encoder-only full was still running/evaluating. Its sample-A-only fallback
was strong (`VOI≈0.224`, `ARAND≈0.036` at 20 records), but only full A/B/C
should be used for conclusions.

Because active/queued dbMiM usage dropped to 10 GPUs, R26 encoder-only early3k
was submitted to directly test whether the encoder-only R24 checkpoint gives a
stronger label-efficient gain over plain MAE:

| purpose | UUID | GPUs | pool |
|---|---|---:|---|
| R26 encoder-only early3k | `992d7ef4-dcca-43d5-9a01-1ba7ca9477cd` | 2 | `med-model` |

After this submission, expected active/queued dbMiM usage is 12 GPUs: R26 full
2, R26 early3k 2, R25 R24 full-init early3k 2, R25 scratch early3k 2, and
fullEM plain-MAE queue 4.

Update at 2026-06-24 00:12 CST:

R25 early3k completed enough to resolve the label-efficient question for the
non-R26 arms:

| early3k arm | records | VOI | ARAND at best VOI | best ARAND |
|---|---:|---:|---:|---:|
| R24 dbMiM++ full-init early3k | 60 | `2.913022` | `0.574107` | `0.565924` |
| R23 plain MAE early3k | 60 | `1.566102` | `0.389115` | `0.374662` |
| R16 dbMiM early3k | 60 | `1.550130` | `0.338732` | `0.338732` |
| scratch early3k | 60 | `1.563107` | `0.309977` | `0.309226` |

Interpretation:

- R24 full-init is decisively negative at both 12k and 3k; do not spend more
  GPU on transferring its decoder/head.
- R16 dbMiM still beats plain MAE slightly on VOI at 3k and more clearly on
  ARAND, but the gain is not large enough for a strong method claim by itself.
- Scratch early3k is competitive, especially by ARAND, so early3k is a noisy
  regime and should mainly be used to diagnose harmful transfer rather than as
  the paper headline.
- The open question is now R26: same R24 checkpoint, encoder-only loading.

R26 encoder-only full is evaluating and has only sample A/B partial rows so
far. R26 encoder-only early3k is running as SiFlow UUID
`992d7ef4-dcca-43d5-9a01-1ba7ca9477cd`. Active/queued dbMiM usage at this
timestamp is 8 GPUs: R26 full 2, R26 early3k 2, and fullEM plain-MAE queue 4.

Update at 2026-06-24 00:26 CST:

The learnable post-processing goal was narrowed to "fast and robust, with
waterz-comparable quality" rather than a standalone absolute-performance
advance. R27 implements this as a tiny per-axis affinity logit calibrator
followed by deterministic post-processing backends:

- script: `scripts/train_learned_affinity_calibration.py`;
- stage: `eval-cremi-fast-learned-postprocess-r17q`;
- outputs:
  `outputs/eval_cremi_fast_learned_postprocess_r17q/{publicem,scratch}`;
- valid final SiFlow UUID: `b86d2af4-09ca-414e-a493-42e1d9c039e1`;
- resources: 2 GPUs on `med-model`, one process for R17 publicEM and one for
  R17 scratch;
- setting: full CREMI A/B/C (`--crop-size 0 0 0`), stride `16 80 80`, train
  calibrator on A/B affinity targets, evaluate A/B/C with `graph_cc`,
  `cupy_graph_cc`, `seeded_rag`, and `waterz`, CREMI boundary ignore
  `xy=1,z=0`, waterz `hist_quantile`.

The R27 submitter deliberately does not use `--fail-on-backend-error`: optional
fast backends may fail on a pod, but waterz and the remaining backends should
still run and write failure rows. Two earlier short UUIDs are not method
evidence: `a4c95d62-8434-4bcf-be8a-8750db6a92ab` omitted waterz packaging, and
`e5d3341b-dedb-4e24-a6e8-f1b3efe607be` was stopped to remove fail-fast
behavior. Poll R27 with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r27_fast_learned_postprocess --once --logs --siflow-fallback
```

After R27 submission, active/queued dbMiM usage is about 10 GPUs: R26 full 2,
R26 early3k 2, R27 fast learned postprocess 2, and fullEM plain-MAE queue 4.

Update at 2026-06-24 09:08 CST:

R26 encoder-only completed both full 12k and early3k A/B/C official evals.
The conclusion is negative for the R24 decoder-aware dbMiM++ path:

| arm | records | VOI | ARAND at best VOI | best ARAND |
|---|---:|---:|---:|---:|
| R26 encoder-only full | 60 | `1.050369` | `0.194521` | `0.189692` |
| R24 full-init | 60 | `1.482749` | `0.281327` | `0.275869` |
| R23 plain MAE | 60 | `1.027073` | `0.192763` | `0.189247` |
| R16 dbMiM | 60 | `1.002919` | `0.188832` | `0.188832` |
| scratch | 60 | `1.095164` | `0.213401` | `0.210442` |

R26 encoder-only is much better than R24 full-init, confirming that the R24
pseudo-affinity decoder/head transfer is harmful. But R26 still does not beat
R23 plain MAE or the older R16 membrane dbMiM, so it is not a positive
dbMiM++-over-MAE result.

Early3k gives the same direction:

| early3k arm | records | VOI | ARAND at best VOI | best ARAND |
|---|---:|---:|---:|---:|
| R26 encoder-only early3k | 60 | `1.553875` | `0.337422` | `0.332548` |
| R24 full-init early3k | 60 | `2.913022` | `0.574107` | `0.565924` |
| R23 plain MAE early3k | 60 | `1.566102` | `0.389115` | `0.374662` |
| R16 dbMiM early3k | 60 | `1.550130` | `0.338732` | `0.338732` |
| scratch early3k | 60 | `1.563107` | `0.309977` | `0.309226` |

R26 encoder-only early3k slightly improves VOI/ARAND over plain MAE, but it is
essentially tied with R16 dbMiM and scratch remains stronger by ARAND. Treat it
as evidence that encoder-only loading avoids the R24 failure mode, not as a
paper-quality method gain.

R27 fast learned postprocess full A/B/C sweep was stopped after about 8.6 hours
with zero summaries. It was too broad for the user's stated "比较快" requirement
(`full A/B/C x publicEM/scratch x graph_cc/cupy_graph_cc/seeded_rag/waterz x
large threshold grid`). Do not relaunch that full sweep.

R28 replaces R27 with a narrow screen:

- stage: `eval-cremi-fast-learned-postprocess-r28q`;
- UUID: `a957727f-8dc3-4b4c-a66a-975957e03ed6`;
- resources: 2 GPUs on `med-model`;
- setting: crop `64x512x512`, stride `16x80x80`, train calibrator on sample A,
  evaluate sample A and holdout sample C, backends `graph_cc`, `seeded_rag`,
  `waterz`, thresholds `0.10/0.20/0.30/0.50`, anisotropic z/xy grid
  `0.10/0.20/0.30`, CREMI boundary ignore `xy=1,z=0`.

Use R28 only as a fast go/no-go for learnable postprocessing. If it cannot get
close to waterz on holdout C with lower `postprocess_sec`, do not spend more
full-volume GPU time on this direction.

Update at 2026-06-24 10:40 CST:

R28 completed and is a negative fast-postprocess result. The canonical TOS
outputs were downloaded locally under
`outputs/downloaded_r28_fast_postprocess/eval_cremi_fast_learned_postprocess_r28q/`
for inspection.

Held-out sample C best rows:

| arm | best backend/variant | VOI | ARAND at best VOI | postprocess_sec |
|---|---|---:|---:|---:|
| R17 publicEM | waterz / raw pred | `1.274807` | `0.267032` | `1.7108` |
| R17 scratch | waterz / raw pred | `1.287874` | `0.266518` | `1.7130` |

Learned calibration did not beat raw predicted affinities through waterz on
held-out C. The non-waterz fast backends were not waterz-comparable: the best
`graph_cc`/`seeded_rag` rows were around VOI `5.11-5.12` with ARAND around
`0.875-0.877`, and their measured `postprocess_sec` was about `2.0s`, not
faster than waterz on this crop. Therefore do not spend more GPU on the current
tiny-calibrator + graph/RAG fast-postprocess path. If post-processing is
revisited, focus on scale engineering around stable waterz-style fragments
and sparse RAG edge scoring, not dense learned RAG or graph connected
components as a replacement for agglomeration.

Current active/queued tasks after the R28 readout:

| purpose | UUID | GPUs | pool | status at submission/check |
|---|---|---:|---|---|
| R18 no-BCAR long-affinity publicEM official A/B/C re-eval | `d35064aa-e7d7-4819-95b2-777e53c94c50` | 1 | `med-model` | Pending at 2026-06-24 10:28 CST |
| R18 no-BCAR long-affinity scratch official A/B/C re-eval | `b1d1d975-409d-4204-85d2-03fb47f7068c` | 1 | `med-model` | Running at 2026-06-24 10:28 CST |
| fullEM plain MAE R23 pretrain | `9658bbd2-cf61-4a54-a392-a014ee488114` | 4 | saved JSON says `cn-shanghai-changliu-skyinfer-reserved-shared` | Queueing, `实际可用(实例配额):0` |

R18 re-eval rationale: the original no-BCAR long-affinity publicEM/scratch
waterz-only jobs were stopped with only partial A/B evidence. Those partial
rows looked unusually strong but were not comparable to official A/B/C. The
new 1-GPU jobs complete exactly that missing A/B/C waterz evaluation without
retraining. Poll with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r18q --once --logs --siflow-fallback
```

The fullEM plain-MAE baseline remains the most important missing comparison
for the full-data story. Do not call R20 fullEM dbMiM positive or negative
relative to MAE until `pretrain-em-full-plain-mae-r23` finishes and its matched
downstream `finetune-cremi-unetr-aniso-arch-explore-maws-mse-fullem-plainmae-r23q`
has the same official A/B/C waterz summary.

Research directions worth submitting next, in priority order:

1. Finish fullEM plain MAE R23. This is the cleanest missing baseline for
   `dbMiM > MAE`; do not change GPU count/global batch unless explicitly
   recording it as a new ablation.
2. Complete R18 no-BCAR long-affinity A/B/C. If it beats R17 VOI or ARAND, it
   is the only current architecture/loss branch worth expanding. If it fails,
   stop long-affinity work and keep R17 MSE+MAWS as the downstream recipe.
3. New dbMiM objective should be encoder-only at downstream load time. R24/R26
   already showed decoder/head transfer is harmful; the next objective should
   improve the encoder representation while leaving the supervised decoder and
   affinity head randomly initialized.
4. For post-processing, only pursue blockwise/streaming waterz-style
   fragments plus sparse RAG edge scoring. The current learned-calibrator,
   dense learned-RAG, DPP, graph-CC, and seeded-RAG attempts are not positive.

Update at 2026-06-24 12:58 CST:

The R18 no-BCAR long-affinity re-evals completed A/B/C with 60 records each.
This closes the earlier partial A/B uncertainty:

| arm | records | VOI | ARAND at best VOI | best ARAND | conclusion |
|---|---:|---:|---:|---:|---|
| long-affinity no-BCAR publicEM | 60 | `1.185532` | `0.244963` | `0.244963` | negative |
| long-affinity no-BCAR scratch | 60 | `1.033775` | `0.199609` | `0.188417` | scratch much better |
| long-affinity + BCAR-rank publicEM | 60 | `1.035857` | `0.192867` | `0.187889` | publicEM gain over paired scratch but not best |
| long-affinity + BCAR-rank scratch | 60 | `1.080162` | `0.204355` | `0.200926` | paired control |

Interpretation:

- The no-BCAR long-affinity branch should not be expanded. It does not show a
  pretraining gain; publicEM is worse than scratch by a large margin.
- Long-affinity + BCAR-rank has a modest publicEM-over-scratch gain but still
  does not beat the R17 MSE+MAWS publicEM VOI `1.002919`.
- Keep R17 MSE+MAWS as the downstream recipe for dbMiM-vs-MAE method tests.

Because R24/R26 showed pseudo-affinity decoder/head transfer is harmful, R29
adds a narrower encoder-only pretraining idea: edge-biased masking. It keeps
the R16-style `DBMIM3DMAE` encoder pretraining but chooses masked patches from
membrane/gradient-rich regions instead of random patches, then downstream loads
only `pos_embed`, `patch_embed`, `encoder_blocks`, and `norm`.

R29 files:

- `configs/pretrain_public_em_edgemask_dbmim_r29.yaml`
- `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_edgemask_r29q.yaml`
- `scripts/watch_and_submit_public_em_edgemask_r29_finetune.sh`

Poll eventual R29 downstream with:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/poll_dbmim_tos_results.py --group r29_edgemask_vs_mae --once --logs --siflow-fallback
```
