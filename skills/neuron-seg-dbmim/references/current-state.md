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
