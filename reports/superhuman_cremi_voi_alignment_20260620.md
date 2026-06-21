# SuperHuman / CREMI VOI alignment audit

Date: 2026-06-20

## Primary reference

- SuperHuman repo: https://github.com/weih527/SuperHuman
- Local clone: `/volume/med-train/users/dchen02/code/_refs/SuperHuman`
- README reported output on AC3/AC4:
  - waterz: `voi_split=1.095144`, `voi_merge=0.342404`, `voi_sum=1.437549`, `arand=0.168990`
  - LMC: `voi_split=1.144543`, `voi_merge=0.262998`, `voi_sum=1.407541`, `arand=0.122037`
- SuperHuman inference uses `utils.fragment.watershed(output_affs, 'maxima_distance')`, waterz/LMC, and `skimage.metrics.adapted_rand_error` / `variation_of_information`.

## Mistakes found in our previous evaluation

1. Earlier VOI numbers were not SuperHuman-aligned because waterz/LMC was unavailable and the fallback path used graph connected components.
2. `--crop-size 0 0 0` was intended to mean full volume, but `evaluate_cremi_segmentation.py` coerced it to the model window. The logged `raw_shape [32,160,160]` was a center crop, not full CREMI.
3. Waterz can mutate the input fragments during agglomeration. Reusing cached fragments across thresholds polluted threshold sweep results after the first threshold.
4. SuperHuman README and current inference code are not fully consistent: current `inference.py` uses a mean-affinity waterz scoring string, but our stable offline waterz v0.8 source does not include `MeanAffinityProvider`. Newer waterz source contains it but produced ABI/runtime failures in SiFlow (`undefined symbol: mergeUntil`).

## Fixes applied

- `scripts/evaluate_cremi_segmentation.py`
  - Added `normalize_crop_size`; non-positive crop entries remain full-volume sentinels.
  - Added printed/saved affinity distribution diagnostics.
  - Summary now records requested and normalized crop size.
- `scripts/evaluate_cremi_diagnostics.py`
  - Uses shared crop normalization.
- `dbmim/postprocess.py`
  - `waterz_agglomeration` now passes a C-contiguous copy of fragments to waterz to prevent threshold contamination.
  - Added waterz aliases for `hist_quantile`, `hist_quantile_false`, `min`, `max`.
  - `mean` now raises a clear error for waterz v0.8.
- `dbmim/datasets.py` / `train_finetune.py`
  - Added optional SuperHuman-style replicate boundary affinity targets.
- `scripts/submit_siflow_dbmim.py`
  - Bundles waterz v0.8 + Boost headers + offline wheelhouse for SuperHuman-style eval.
  - SuperHuman eval defaults back to stable `hist_quantile` scoring.

## Full-volume evidence

CREMI local files are full standard volumes:

- `sample_A_20160501.hdf`: `volumes/raw`, `volumes/labels/neuron_ids` shape `[125,1250,1250]`
- `sample_B_20160501.hdf`: shape `[125,1250,1250]`
- `sample_C_20160501.hdf`: shape `[125,1250,1250]`

Fixed full-volume eval logs confirmed:

- `raw_shape: [125,1250,1250]`
- `crop: [[0,125],[0,1250],[0,1250]]`

## Full-volume results observed so far

Using stable waterz v0.8 `hist_quantile`, sample A full volume remains very poor:

- pretrained R3, threshold 0.3: `voi_sum=9.958972`, `voi_split=2.360805`, `voi_merge=7.598168`, `arand=0.986622`, `n_pred=531`, `n_gt=37366`
- scratch R3, threshold 0.3: `voi_sum=9.847460`, `voi_split=2.386964`, `voi_merge=7.460497`, `arand=0.986317`, `n_pred=6637`, `n_gt=37366`

The full-volume affinity distributions are over-connected:

- pretrained R3 sample A:
  - z median `0.8502`, y median `0.9816`, x median `0.9869`
  - y/x p05 are `0.9571` / `0.9663`
- scratch R3 sample A:
  - z median `0.9121`, y median `0.9922`, x median `0.9943`
  - y/x p05 are `0.9244` / `0.9384`

This explains high VOI merge: most XY affinities are close to 1, so watershed/waterz under-segments badly.

## SiFlow task IDs

Stopped because of polluted hist sweep or wrong mean scoring:

- `341e8d6b-d0a0-43fa-9ee6-77c7469c0170`: old full-volume hist, stopped.
- `c60e9162-98c7-4aa2-8337-317a0aaf2507`: old full-volume hist, stopped.
- `5fc7c09b-21a8-4822-b575-3abce3958b5b`: fixed full-volume hist, stopped after sample A evidence.
- `fea764d3-d64a-4c79-b570-29673463524f`: fixed full-volume hist, stopped after sample A evidence.
- `ebd032ed-bf92-40fe-b965-7cf8f5170d3e`: mean scoring, failed because waterz v0.8 lacks `MeanAffinityProvider`.
- `d309d05b-65d4-48f4-8fd3-027c2c43c6b6`: same failure.

## Interpretation

The user was correct that SuperHuman-style direct neuron segmentation should report VOI around 1 on its benchmark. Our old high VOI was partly caused by evaluation mistakes. After fixing those mistakes, the current R3 dbMiM/UNETR checkpoints still do not approach SuperHuman-style segmentation on full CREMI: the affinity output is strongly over-connected, especially in XY channels.

The next improvement should not be another blind threshold sweep. Priority should be:

1. Rebalance affinity training toward boundary/non-affinity positives: weighted BCE/MSE or focal/Tversky on boundary class, especially XY edges.
2. Train with SuperHuman-style affinity target padding and verify boundary F1/AP, not only affinity Dice/IoU.
3. Add calibration before waterz: temperature/logit shift or per-channel affine calibration selected on validation VOI.
4. Revisit watershed seed generation: current XY boundary is too low because affinities are too high; seed/boundary thresholds need calibration after logits are fixed.
5. If mean-affinity waterz is required, use newer waterz but fix the SiFlow ABI/runtime issue before treating it as stable.

## 2026-06-20 continuation: calibration probe and R4 training

Added eval-time calibration sweep support to `scripts/evaluate_cremi_segmentation.py`:

- `--calibration-biases`: z/y/x logit-bias triplets.
- `--calibration-temperatures`: logit temperature values.
- Records now include `affinity_variant`, per-channel bias, and temperature.

Submitted SiFlow jobs:

- calibration probe, R3 pretrained, 1 GPU med-model: `29eb99f1-959a-4907-a19a-3a45b9ebb1de`.
- R4 SuperHuman-style pretrained finetune, 4 GPU med-model: `8364c135-4987-4aea-8e29-9f7a6d4f3bed`.

Calibration probe on full `sample_A_20160501.hdf`:

- raw pred, threshold 0.3: `voi_sum=9.958972`, `voi_split=2.360805`, `voi_merge=7.598168`, `n_pred=531`.
- bias `z=-0.5,y=-1,x=-1`, threshold 0.3: `voi_sum=9.951857`, `voi_split=2.370400`, `voi_merge=7.581457`, `n_pred=2734`.
- bias `z=-1,y=-2,x=-2`, threshold 0.5: `voi_sum=9.945377`, `voi_split=2.364737`, `voi_merge=7.580639`, `n_pred=938`.
- bias `z=-1,y=-2,x=-2`, threshold 0.3: `voi_sum=11.564190`, over-splits (`voi_split=5.037334`, `n_pred=8780`).
- bias `z=-1.5,y=-3,x=-3`, threshold 0.3: `voi_sum=8.709598`, `voi_split=6.473930`, `voi_merge=2.235668`, `n_pred=95415`.

Interpretation: simple logit calibration can trade merge for split and improves the best observed full-volume sample-A VOI from 9.96 to 8.71, but this is still far from the expected ~1.x regime and is not a reliable method. The useful direction remains training-time boundary supervision / target balancing, not inference-only bias sweep.

R4 training status at last check:

- task `8364c135-4987-4aea-8e29-9f7a6d4f3bed` Running.
- around epoch 15 / step 15.9k after ~16 min.
- train loss has dropped from ~0.58 early to frequent 0.33-0.50 samples, but remains noisy; boundary dice loss is still high and must be checked by full-volume waterz once a checkpoint is saved.

## 2026-06-20 continuation: root-cause fix and R5 SuperHuman training

The main training bug found after comparing our pipeline against SuperHuman was
not in VOI itself. `EMVolumeDataset.__getitem__` applied random flips and
intensity/noise augmentation to the image tensor before attaching the label.
The instance label was therefore not flipped with the image. This made
supervised affinity finetuning internally inconsistent and explains the
previous over-connected affinity outputs and non-convergent VOI behavior.

Code fixes now applied:

- `dbmim/datasets.py`
  - Added `augment_image_and_label` so z/y/x flips are shared by image and
    label, while intensity/noise remain image-only.
  - Added optional SuperHuman/Kisuk-style 2D instance-border invalidation via
    `widen_instance_boundaries_2d`.
- `train_finetune.py`
  - Added SuperHuman-style weighted MSE affinity supervision
    (`loss.loss_type: weighted_mse`) with per-sample/channel binary balancing.
  - Passes `data.widen_border` and `data.widen_border_radius` into
    `EMVolumeDataset`.
- `configs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5.yaml`
  and `configs/finetune_cremi_real_unetr_aniso_superhuman_scratch_r5.yaml`
  - Use `unetr_aniso`, `32x160x160` crops, synchronized augmentation,
    widened labels, replicate affinity boundary targets, weighted MSE, and
    boundary Dice auxiliary loss.
- `scripts/submit_siflow_dbmim.py`
  - Added R5 training/evaluation stages and calibration output sync.

Local validation:

- `python -m py_compile dbmim/datasets.py train_finetune.py scripts/submit_siflow_dbmim.py` passed.
- A smoke check verified synchronized label flips, `widen_instance_boundaries_2d`,
  and `affinity_loss(... loss_type=weighted_mse)` backward pass.

R5 submitted jobs:

- stopped 8-GPU attempts due real `sci.g21-3` instance availability:
  - pretrained med-model: `26b6b0c5-5c9c-497f-80b7-4daef7a305ce`
  - scratch med-model: `8e5119d9-681d-41f7-afd4-4d318e18e0e6`
  - pretrained shared: `452f68a8-dc34-4a60-a3a6-85e9d9cf86c7`
- active pretrained R5 4-GPU med-model training:
  `f9f74d70-4798-400f-a189-afad4c1d7e4e`
- queued scratch R5 4-GPU med-model control:
  `162150c0-9350-4254-bdce-c599b7fdcfa4`
- active full-volume sample-A waterz/skimage calibration eval for pretrained
  R5 latest checkpoint, 1 GPU med-model:
  `9cabcc88-d7ed-4936-89ca-e2891904b648`

R5 pretrained training evidence:

- Loaded pretrained ViT encoder: `loaded_pretrained_keys=77`.
- Dataset size: 12288 random crops, distributed world size 4, per-GPU batch 2,
  effective global batch 8.
- Loss config:
  `weighted_mse + 0.05 affinity Dice + 0.35 boundary Dice`, channel weights
  `[1.25, 1.0, 1.0]`.
- Early loss dropped much more cleanly than R4:
  - step 600: `train_loss=0.321`, main/old-log `train_bce_loss=0.120`,
    boundary Dice loss `0.552`
  - step 10k: typical `train_loss=0.10-0.19`, main/old-log
    `train_bce_loss=0.05-0.10`, boundary Dice loss `0.14-0.27`
  - step 12.2k: still stable around `train_loss=0.10-0.12`
- Current TOS checkpoint prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5/`
  with `finetuned_latest.pt` already present.

The R5 full-volume eval has confirmed:

- `raw_shape: [125, 1250, 1250]`
- `crop: [[0, 125], [0, 1250], [0, 1250]]`
- checkpoint:
  `outputs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5/finetuned_latest.pt`
- modules available in pod:
  `scipy`, `mahotas`, `cc3d`, `waterz`, `cupy`; `elf` absent.

First R5 full-volume sample-A waterz/skimage results from the latest checkpoint
are now in the expected CREMI scale. Raw `pred` affinity, waterz
`hist_quantile`, `seed_distance=10`, `boundary_threshold=0.5`,
`ignore_label=0`:

| threshold | VOI sum | VOI split | VOI merge | ARAND | Rand F | n_pred | n_gt |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.742093 | 0.449843 | 0.292251 | 0.126184 | 0.873816 | 27535 | 37366 |
| 0.10 | 0.759984 | 0.430677 | 0.329307 | 0.161975 | 0.838025 | 23599 | 37366 |
| 0.20 | 0.747036 | 0.410954 | 0.336082 | 0.162093 | 0.837907 | 19172 | 37366 |
| 0.30 | 0.735735 | 0.396551 | 0.339184 | 0.161714 | 0.838286 | 16231 | 37366 |
| 0.50 | 0.717879 | 0.371610 | 0.346268 | 0.161406 | 0.838594 | 12209 | 37366 |

Interpretation:

- The user's concern was correct: VOI around 8-10 was not a normal outcome for
  this task.
- After fixing synchronized supervised augmentation and switching to
  SuperHuman-style training targets/loss, the same full-volume sample-A eval
  moved from the earlier R4/R3 failure range (`VOI_sum` roughly 8-10) to
  `VOI_sum=0.717879` at threshold 0.50 and `ARAND=0.126184` at threshold 0.05.
- This is a real method/implementation recovery, not a threshold-only artifact:
  the full volume, metric backend, waterz dependency, and sample-A label count
  were all verified in the eval log.

Next decision rule:

- Keep the pretrained R5 run to 30k unless later checkpoints regress.
- Let the scratch R5 control start when quota frees; this is now the clean
  pretraining-effect comparison.
- Add checkpoint sweeps at 10k/20k/30k and sample B/C full-volume evaluation.
- Then run small ablations around `widen_border_radius`, weighted MSE vs BCE,
  and seed/threshold choices; do not spend more cycles on the old R4 path.

## 2026-06-20 09:13 UTC update: R5 30k and ablation wave

The pretrained R5 4-GPU training job
`f9f74d70-4798-400f-a189-afad4c1d7e4e` completed 30k optimizer steps and
uploaded final checkpoints to:

`tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5/`

Important artifacts:

- `checkpoint_step_00030000.pt`
- `finetuned_latest.pt`
- `finetune_log.jsonl`

Final training log row:

- step 30000: `train_loss=0.119863`, main weighted-MSE term
  `0.062729`, affinity Dice loss `0.031882`, boundary Dice loss `0.158688`.

A full-volume sample-A evaluation completed on the final 30k checkpoint:

- SiFlow UUID: `68115eb6-ecd3-4f6b-af50-98c26c35faee`
- resource pool: `med-model`
- checkpoint path inside pod:
  `outputs/finetune_cremi_real_unetr_aniso_superhuman_pretrained_r5/finetuned_latest.pt`
- confirmed full volume:
  `raw_shape [125,1250,1250]`, crop `[[0,125],[0,1250],[0,1250]]`.

The uncalibrated raw `pred` waterz sweep on sample A is already complete:

| variant | threshold | VOI sum | VOI split | VOI merge | ARAND | Rand F | n_pred | n_gt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pred | 0.05 | 0.645302 | 0.367920 | 0.277382 | 0.111310 | 0.888690 | 21346 | 37366 |
| pred | 0.10 | 0.637800 | 0.359174 | 0.278626 | 0.111051 | 0.888949 | 18972 | 37366 |
| pred | 0.20 | 0.628222 | 0.347504 | 0.280718 | 0.110802 | 0.889198 | 16394 | 37366 |
| pred | 0.30 | 0.620097 | 0.336619 | 0.283478 | 0.110683 | 0.889317 | 14557 | 37366 |
| pred | 0.50 | 0.604249 | 0.315553 | 0.288696 | 0.109885 | 0.890115 | 11402 | 37366 |

Calibration improved both VOI and ARAND. The best final pretrained R5
full-volume sample-A row is:

| variant | threshold | VOI sum | VOI split | VOI merge | ARAND | Rand F | n_pred | n_gt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.563006 | 0.320270 | 0.242735 | 0.073386 | 0.926614 | 10315 | 37366 |

The synced final metrics are stored at:

`tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_superhuman_pretrained_r5/`

The scratch R5 4-GPU control completed 30k optimizer steps:

- SiFlow UUID: `162150c0-9350-4254-bdce-c599b7fdcfa4`
- resource pool: `med-model`
- final step 30000: `train_loss=0.122041`, affinity Dice loss `0.032573`,
  boundary Dice loss `0.161410`.
- checkpoint prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_scratch_r5/`
- full-volume sample-A calibration eval submitted manually because the first
  watcher was not still running:
  `a4afc427-3b32-4439-b9b5-cf6f5aec37c7`.

The scratch sample-A eval completed and is currently stronger than the
pretrained R5 arm:

| arm | selected by | variant | threshold | VOI sum | VOI split | VOI merge | ARAND | Rand F | n_pred |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| scratch R5 | best VOI | bias `z=-0.25,y=-0.50,x=-0.50` | 0.50 | 0.546834 | 0.324803 | 0.222031 | 0.066024 | 0.933976 | 11054 |
| scratch R5 | best ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.548672 | 0.334081 | 0.214592 | 0.058093 | 0.941907 | 10551 |
| pretrained R5 | best VOI/ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.563006 | 0.320270 | 0.242735 | 0.073386 | 0.926614 | 10315 |

Interpretation at this point: the R5 method stack is the real gain
(`sync image/label augmentation + widened affinity targets + weighted MSE +
waterz calibration`). On sample A, the dbMiM-pretrained ViT encoder does not
show a positive finetuning effect; scratch is slightly better. Do not claim a
pretraining gain unless the A/B/C full-volume comparison reverses this.

Submitted additional pretrained R5 ablations on `cpt-train`, 4 GPUs each:

| ablation | purpose | UUID | status at 09:32 UTC |
|---|---|---|---|
| `superhuman-nowiden-pretrained-r5` | remove 2D label border widening | `31935cc0-64f0-4e55-bacd-14d043741acb` | Running, ~15.6k/30k; boundary Dice loss still high |
| `superhuman-bce-pretrained-r5` | replace weighted MSE with BCE | `41fe0043-f7e5-4e10-8067-4c5c5618d926` | Running, ~15.9k/30k |
| `superhuman-boundaryhigh-pretrained-r5` | increase boundary Dice weight from 0.35 to 0.55 | `e28c3264-aaa5-491a-a79f-f307bbe5ef43` | Running, ~16.7k/30k |

The first final-checkpoint watcher launch wrote only `watch_start` and was not
alive when checked. A more explicit `nohup` watcher wave also did not survive
long enough in this environment. Do not rely on local watchers for long-running
submission; poll SiFlow/TOS and submit eval manually after
`checkpoint_step_00030000.pt` appears.

Historical watcher logs:

- old scratch watcher, superseded by manual eval:
  `outputs/watchers/eval_scratch_r5_20260620T091340Z_final30k.log`
- no-widen watcher attempt:
  `outputs/watchers/eval_nowiden_pretrained_r5_20260620T093118Z_final30k_nohup.log`
- BCE watcher attempt:
  `outputs/watchers/eval_bce_pretrained_r5_20260620T093118Z_final30k_nohup.log`
- boundary-high watcher attempt:
  `outputs/watchers/eval_boundaryhigh_pretrained_r5_20260620T093118Z_final30k_nohup.log`

Because local background watchers were not reliable in this environment, the
three ablation evals were submitted manually after final checkpoints appeared:

| eval | UUID |
|---|---|
| no-widen pretrained R5 sample-A calibration | `28295ae2-2b2a-4c00-a8b7-5e4c9ba3da16` |
| BCE pretrained R5 sample-A calibration | `adb48dd4-e833-4744-addf-78c3defc2c5e` |
| boundary-high pretrained R5 sample-A calibration | `c605d26c-163b-4b3b-8fa4-512fbe7029c5` |

Submitted A/B/C full-volume follow-up to check whether the sample-A scratch
advantage holds across all CREMI samples:

| arm | UUID | output prefix |
|---|---|---|
| pretrained R5 A/B/C calibration | `812df662-d464-49d3-8622-93da3e1709cc` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_all_superhuman_pretrained_r5/` |
| scratch R5 A/B/C calibration | `98bc4983-7ca2-4f6e-9fe4-31f4a2791ba0` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_all_superhuman_scratch_r5/` |

## 2026-06-20 10:35 UTC update: ablations and CREMI metric-mask alignment

The three sample-A ablation evals completed. Local copies are under
`outputs/tos_fetch/{bce_r5_eval,boundaryhigh_r5_eval,nowiden_r5_eval}/`.

Using the same raw-label `ignore_label=0` metric path as the earlier R5
sample-A comparison:

| arm | best by | variant | threshold | VOI sum | VOI split | VOI merge | ARAND | n_pred | n_gt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| pretrained R5 | VOI/ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.563006 | 0.320270 | 0.242735 | 0.073386 | 10315 | 37366 |
| scratch R5 | VOI | bias `z=-0.25,y=-0.50,x=-0.50` | 0.50 | 0.546834 | 0.324803 | 0.222031 | 0.066024 | 11054 | 37366 |
| BCE pretrained R5 | VOI | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.535448 | 0.261332 | 0.274116 | 0.080269 | 10413 | 37366 |
| BCE pretrained R5 | ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.20 | 0.543432 | 0.326991 | 0.216441 | 0.067376 | 20219 | 37366 |
| boundary-high pretrained R5 | VOI/ARAND | bias `z=-0.25,y=-0.50,x=-0.50` | 0.50 | 0.559225 | 0.318553 | 0.240672 | 0.073207 | 12262 | 37366 |
| no-widen pretrained R5 | VOI | bias `z=-0.50,y=-1.00,x=-1.00` | 0.10 | 4.516494 | 0.463350 | 4.053144 | 0.961479 | 19810 | 37366 |

Interpretation:

- BCE is currently the most credible sample-A VOI improvement:
  `0.535448` versus scratch `0.546834` and pretrained weighted-MSE
  `0.563006`. Its best ARAND is still weaker than scratch's best ARAND.
- Raising `boundary_dice_weight` to `0.55` did not improve over the main R5
  stack.
- Removing 2D border widening is catastrophic (`VOI_sum > 4.5`, ARAND
  `~0.96`), so the SuperHuman/Kisuk-style 2D instance-border invalidation is a
  necessary component of the method stack.

There is a separate metric-mask issue. CREMI's neuron segmentation metric
description says ground-truth pixels close to object boundaries on the same
section are set to background and ignored. The downloaded CREMI train HDF5
labels under `volumes/labels/neuron_ids` have no original zero labels:

| sample | label unique count | zero count |
|---|---:|---:|
| sample A | 37366 | 0 |
| sample B | 1309 | 0 |
| sample C | 1978 | 0 |

The previous R5 numbers therefore used a strict raw-label metric path, not the
CREMI boundary-ignore path. `scripts/evaluate_cremi_segmentation.py` now has
explicit options:

- `--cremi-boundary-ignore-distance-xy`
- `--cremi-boundary-ignore-distance-z`

The implemented default remains `0` to preserve old results. The new
official-style A/B/C eval stages set `--cremi-boundary-ignore-distance-xy 1`
and `--cremi-boundary-ignore-distance-z 0`, matching the "same section"
wording.

Official-style sample-A rows observed so far:

| arm | variant | threshold | VOI sum | VOI split | VOI merge | ARAND | n_pred | n_gt after mask | ignored voxels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained R5 | pred | 0.50 | 0.246585 | 0.130824 | 0.115761 | 0.077635 | 11319 | 10929 | 11.69% |
| scratch R5 | pred | 0.50 | 0.204558 | 0.132668 | 0.071889 | 0.048634 | 11816 | 10929 | 11.69% |
| scratch R5 | pred | 0.10 | 0.223857 | 0.163728 | 0.060129 | 0.046854 | 19405 | 10929 | 11.69% |

This confirms that the user's metric concern was valid: switching from raw
label metrics to the CREMI boundary-ignore metric substantially lowers VOI on
sample A. The strict raw-label metric remains useful as a harder diagnostic,
but headline CREMI-style numbers should use the boundary-ignore setting and
state it explicitly.

Official-style A/B/C jobs currently running:

| arm | UUID | output prefix |
|---|---|---|
| pretrained R5 official A/B/C | `44efdd27-baf7-418b-aa80-3b74207dc70b` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_all_superhuman_pretrained_r5/` |
| scratch R5 official A/B/C | `5aad9226-1c61-40a4-b330-93bdf87abc31` | `tos://agi-data/users/dchen02/dbmim/outputs/eval_cremi_unetr_aniso_superhuman_calibration_official_all_superhuman_scratch_r5/` |

The original raw-label A/B/C jobs are still running and are useful as a strict
diagnostic, but do not mix their numbers with official-style boundary-ignore
numbers in one table.

The first corrected low-encoder-LR job was stopped because DDP parameter names
kept the encoder from being split into a lower-LR group. Its output prefix
`finetune_cremi_real_unetr_aniso_superhuman_lowencoder_pretrained_r5` is
invalid for scientific comparison. The corrected encoder-LR job is:

- train UUID: `c5f2a38a-d6a7-46d2-b46d-73b57a0f000b`
- output prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_encoderlr_pretrained_r5/`
- confirmed optimizer groups:
  encoder `lr=2e-5`, 77 tensors, 3,019,968 params; decoder `lr=8e-5`, 115
  tensors, 15,051,747 params.
- at ~17.5 minutes it was around step 16.2k/30k, with typical
  `train_loss=0.09-0.16`.

## 2026-06-20 10:50 UTC update: raw A/B/C complete and official-A tighter rows

The strict raw-label A/B/C jobs completed and were downloaded under:

- `outputs/tos_fetch/raw_pretrained_r5_abc/`
- `outputs/tos_fetch/raw_scratch_r5_abc/`

Aggregate best rows across samples A/B/C, still using the strict raw-label
`ignore_label=0` metric path:

| arm | selected by | variant | threshold | VOI sum | VOI split | VOI merge | ARAND | Rand F |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pretrained R5 | VOI | bias `z=-0.25,y=-0.50,x=-0.50` | 0.30 | 1.556559 | 0.857899 | 0.698659 | 0.369710 | 0.630290 |
| pretrained R5 | ARAND | pred | 0.05 | 1.650351 | 0.918942 | 0.731410 | 0.356289 | 0.643711 |
| scratch R5 | VOI | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 1.441593 | 0.857869 | 0.583724 | 0.313700 | 0.686300 |
| scratch R5 | ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.10 | 1.452166 | 0.935148 | 0.517018 | 0.312191 | 0.687809 |

Conclusion for strict raw-label A/B/C: scratch is clearly ahead of dbMiM
pretrained at 30k. This reinforces the sample-A result; do not claim a
pretraining gain from the current R5 setup.

The corrected encoder-LR pretrained R5 run also completed:

- train UUID: `c5f2a38a-d6a7-46d2-b46d-73b57a0f000b`
- duration: 32m17s on 4 GPUs
- final step 30000: `train_loss=0.122088`, main loss `0.064404`, affinity
  Dice loss `0.032236`, boundary Dice loss `0.160206`.

Its first official-A eval attempt, `1b8cf096-294e-46d5-a3f6-029c219dda08`,
failed before evaluation because switching the bundle to newer `waterz` caused
old setuptools in the SiFlow image to reject the newer `pyproject.toml`
`project.license` field. The submitter was reverted to stable bundled
`waterz_v08`; the corrected official-A eval is:

- UUID: `8400bb84-5126-4c96-9266-fbe54b5499a5`

The official-style pretrained R5 sample-A sweep also reached a better
calibrated row than the raw-pred row:

| arm | variant | threshold | VOI sum | VOI split | VOI merge | ARAND | n_pred | n_gt after mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pretrained R5 | pred | 0.50 | 0.246585 | 0.130824 | 0.115761 | 0.077635 | 11319 | 10929 |
| pretrained R5 | bias `z=-0.25,y=-0.50,x=-0.50` | 0.50 | 0.233519 | 0.127890 | 0.105629 | 0.070057 | 10727 | 10929 |
| pretrained R5 | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.198980 | 0.133149 | 0.065831 | 0.038487 | 10280 | 10929 |
| scratch R5 | pred | 0.50 | 0.204558 | 0.132668 | 0.071889 | 0.048634 | 11816 | 10929 |
| scratch R5 | bias `z=-0.25,y=-0.50,x=-0.50` | 0.50 | 0.178741 | 0.134729 | 0.044012 | 0.031083 | 11009 | 10929 |
| scratch R5 | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.177157 | 0.142502 | 0.034655 | 0.021736 | 10512 | 10929 |

Official-style sample-A also still favors scratch. The gap is smaller after
pretrained calibration, but scratch remains better in both VOI and ARAND.

Additional jobs submitted:

| job | UUID | purpose |
|---|---|---|
| BCE scratch R5 train | `9f2bbe23-abfa-41c0-b00c-63f76fd76868` | test whether BCE gain is loss-driven rather than pretraining-driven |
| BCE pretrained official-A eval | `47ddaea5-1a31-4c7c-8518-326d4c46b869` | boundary-ignore sample-A check for the current best raw-label VOI ablation |
| encoder-LR pretrained official-A eval | `8400bb84-5126-4c96-9266-fbe54b5499a5` | test whether preserving the pretrained encoder with lower LR helps |

Waterz source note: stable SiFlow bundles currently use local
`/volume/med-train/users/dchen02/code/_refs/waterz_v08`. Newer
`/volume/med-train/users/dchen02/code/_refs/waterz` supports mean-affinity
scoring, but needs a patched/offline-compatible build before use in jobs.

## 2026-06-21 01:35 UTC update: official A/B/C and paired BCE control

The official-style A/B/C evaluations completed with the CREMI boundary-ignore
mask (`--cremi-boundary-ignore-distance-xy 1`,
`--cremi-boundary-ignore-distance-z 0`). Local copies are under:

- `outputs/tos_fetch/official_pretrained_r5_abc/`
- `outputs/tos_fetch/official_scratch_r5_abc/`

The metric mask ignored boundary neighborhoods on each XY section and changed
the effective GT counts to:

| sample | ignored voxels | effective GT labels |
|---|---:|---:|
| sample A | 11.6916% | 10929 |
| sample B | 8.4857% | 1092 |
| sample C | 10.5215% | 1878 |

Aggregate official-style A/B/C best rows:

| arm | selected by | variant | threshold | VOI sum | VOI split | VOI merge | ARAND | Rand F |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pretrained R5 | VOI | bias `z=-0.25,y=-0.50,x=-0.50` | 0.30 | 1.250565 | 0.682528 | 0.568037 | 0.347600 | 0.652400 |
| pretrained R5 | ARAND | pred | 0.05 | 1.337888 | 0.733668 | 0.604220 | 0.332747 | 0.667253 |
| scratch R5 | VOI | bias `z=-0.50,y=-1.00,x=-1.00` | 0.30 | 1.121139 | 0.708184 | 0.412955 | 0.287917 | 0.712083 |
| scratch R5 | ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.10 | 1.125567 | 0.743731 | 0.381836 | 0.287647 | 0.712353 |

This is the expected CREMI-scale regime and confirms that the earlier
`VOI=8-10` range was an implementation/training problem, not the intended
volume-level behavior. It also confirms the current scientific result:
standard dbMiM-pretrained R5 is still weaker than scratch after official-style
A/B/C aggregation.

The BCE-pretrained and encoder-LR official sample-A evaluations also completed:

| arm | selected by | variant | threshold | VOI sum | VOI split | VOI merge | ARAND | Rand F |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BCE pretrained R5 | VOI/ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.20 | 0.185597 | 0.135911 | 0.049687 | 0.032785 | 0.967215 |
| encoder-LR pretrained R5 | VOI | pred | 0.30 | 0.210299 | 0.146192 | 0.064107 | 0.038153 | 0.961847 |
| encoder-LR pretrained R5 | ARAND | bias `z=-0.25,y=-0.50,x=-0.50` | 0.30 | 0.219624 | 0.156769 | 0.062855 | 0.036421 | 0.963579 |

BCE-pretrained is the most credible current pretrained-side sample-A
improvement, but it must be compared to the paired BCE scratch control before
attributing the gain to pretraining. The BCE scratch training job completed:

- train UUID: `9f2bbe23-abfa-41c0-b00c-63f76fd76868`
- final step 30000: `train_loss=0.147030`, BCE/main loss `0.091217`,
  affinity Dice loss `0.028496`, boundary Dice loss `0.155396`.
- output prefix:
  `tos://agi-data/users/dchen02/dbmim/outputs/finetune_cremi_real_unetr_aniso_superhuman_bce_scratch_r5/`

Paired BCE scratch evaluations submitted on 2026-06-21:

| eval | UUID | purpose |
|---|---|---|
| official-style BCE scratch sample A | `df6582ca-c87c-4843-a8ca-23e41030c3b5` | compare directly with BCE-pretrained official-A |
| raw-label BCE scratch sample A | `4ce29ab6-4775-41f5-ac70-c2b62d94b9c4` | compare with the earlier strict raw-label BCE-pretrained row |

Both paired BCE scratch evaluations completed:

| arm | metric mask | selected by | variant | threshold | VOI sum | VOI split | VOI merge | ARAND | Rand F |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| BCE pretrained R5 | official sample A | VOI/ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.20 | 0.185597 | 0.135911 | 0.049687 | 0.032785 | 0.967215 |
| BCE scratch R5 | official sample A | VOI | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.169502 | 0.103343 | 0.066160 | 0.032910 | 0.967090 |
| BCE scratch R5 | official sample A | ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.20 | 0.178548 | 0.144804 | 0.033745 | 0.024062 | 0.975938 |
| BCE pretrained R5 | raw-label sample A | VOI | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.535448 | 0.261332 | 0.274116 | 0.080269 | 0.919731 |
| BCE pretrained R5 | raw-label sample A | ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.20 | 0.543432 | 0.326991 | 0.216441 | 0.067376 | 0.932624 |
| BCE scratch R5 | raw-label sample A | VOI | bias `z=-0.50,y=-1.00,x=-1.00` | 0.50 | 0.517256 | 0.278840 | 0.238416 | 0.066566 | 0.933434 |
| BCE scratch R5 | raw-label sample A | ARAND | bias `z=-0.50,y=-1.00,x=-1.00` | 0.20 | 0.538486 | 0.338938 | 0.199548 | 0.058831 | 0.941169 |

Conclusion: BCE is a real improvement over the weighted-MSE pretrained branch
on sample A, but the paired scratch control is stronger than BCE-pretrained
under both metric masks. The current evidence supports BCE + synchronized R5
training/CREMI post-processing as the useful method change, not a positive
dbMiM-pretraining effect.
