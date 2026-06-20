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

At the time of this report update, waterz/VOI rows were still running. The
decision rule is:

- If R5 full-volume VOI improves substantially versus R4 but remains above
  SuperHuman scale, keep training to 30k and evaluate later checkpoints.
- If R5 remains near VOI 8-10 after synchronized supervision, stop blind
  training and switch to targeted method changes: boundary head calibration,
  direct boundary AP/F1 validation, watershed seed ablation, and possibly a
  stronger SuperHuman-style 2D U-Net affinity baseline as a teacher/checkpoint
  sanity reference.
