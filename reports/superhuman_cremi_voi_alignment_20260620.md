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
