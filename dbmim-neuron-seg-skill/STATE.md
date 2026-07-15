# dbMiM Neuron Segmentation State

Updated: 2026-07-15T02:52Z

## Current Direction

The current goal has been upgraded from a dbMiM ablation into an independent
Stem-Micro post-processing system aimed at a Nature Methods-style methods
paper. The target is a fast, high-quality, generalizable post-processing method
for arbitrary EM volumes and MICrONS/CAVE-scale deployment.

The practical path is still a robust hierarchy solver (`waterz`, MWS/GASP, or
signed graph partitioning) with learned or label-free calibration, not direct
GPU connected components. The learnable component should select/calibrate the
solver and predict merge ranks or uncertainty; it should not replace global
connectivity with a brittle hard threshold.

## MEC Validation Protocol Update

The high-quality R57 MEC11/12 y-axis slabwise result is exploratory evidence,
not a deployment selection. Official validation MEC08/10 invalidates its
threshold neighborhood: calibrated R57 waterz with y-axis slabs `768/256`
gives aggregate full ARAND/VOI `0.40385415920940215` / `1.5406254233101775`
at threshold `0.33`, `0.40389850376670944` / `1.5407018490152256` at `0.34`,
and `0.4038303339256861` / `1.5375763983790238` at `0.35`.

The fixed-calibration full-block low-threshold control confirms this is not
remedied by a scalar cut. R57 bias `(-0.20,-0.45,-0.45)`, temperature `1.20`,
and one shared waterz threshold reaches at best mean MEC08/10 ARAND/VOI
`0.2522797956118525` / `1.0622358210386302` at `0.16`. MEC08 is much better
than MEC10, so the issue is a generalization/domain failure rather than a
globally misplaced threshold. The result is archived at:

`tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r57_seed777_fullblock_val_waterz_lowthr_tgrid`

Two label-free auto-calibration probes were stopped after near-one ARAND and
about 18 VOI on MEC08; they are negative controls, not candidates.

The active response is a controlled 16-GPU validation-only pair. R58 starts
from public R57 and trains MEC01--07 plus MEC09 with weighted MSE, boundary
Dice, BCAR ranking, membrane-edge weighting, and stronger intensity
augmentation. The first R58 pod was stopped early when its preflight showed
waterz was absent; the launcher now installs and verifies bundled waterz
before training. The valid retry is `c34cc14e-eb87-4d43-a05d-a45aea2edba5`
(`mec8nm-r58-balanced-wz-b9fbf56e`), an 8-GPU
`cn-shanghai/changliu/med-model` job.

R59 is a strict R58 control with only a learnable local affinity postprocessor
and DPP loss added (`dpp_weight=0.04`): UUID
`dd467b76-4bb8-4266-ac09-6b8204121e96` (`mec8nm-r59-dpp-e689538c`). DPP is
checkpointed and loaded by the evaluator, so it is a genuine inference-time
candidate rather than a training-only regularizer. Both jobs use waterz and
evaluate only MEC08/10.

R60 is the separate data-domain control: public R57 is fine-tuned only on
low-boundary training blocks MEC07 and MEC09, while retaining R58's loss and
augmentation. UUID `0791406a-b6ca-4edc-b9c1-b5b2fe3dc21d`
(`mec8nm-r60-simple-d04fe35`) logged `train_ids=[7,9]`, `train_paths=2`,
`dataset_size=8192`, and `waterz=true` at startup. It is limited to `6000`
steps and MEC08/10 evaluation. R58/R59/R60 are three 8-GPU packs, exactly the
24-GPU user cap; do not submit another training candidate before comparing
their validation results.

This pair is motivated by sampled topology statistics, not a raw intensity
assumption. Across 24 `32x160x160` samples per block, training averages
foreground/boundary/instances `0.890260/0.227526/22.942708`; validation
averages `0.985221/0.038641/4.145833`, with MEC08 near a single-object regime.
Per-crop 1/99 normalization already removes global intensity scale. Do not run
a new MEC11/12 evaluation until validation selects a locked R58 or R59
postprocess setting.

Design document:

`/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/docs/stem_micro_nature_methods_design.md`

## Latest Stable Result

- MEC 8nm current best family: R57 checkpoint plus calibrated waterz, bias
  `(-0.20,-0.45,-0.45)`, temperature `1.20`. Stable full-block VOI-priority
  anchor is waterz threshold `0.36`: aggregate full ARAND/VOI
  `0.01245975276529443` / `0.2240353497868297`, seam ARAND/VOI
  `0.013031550879182785` / `0.21150499794760833`, nonseam ARAND/VOI
  `0.012408026169544673` / `0.21977815247519208`. ARAND-priority full-block
  alternative is threshold `0.32`: full ARAND/VOI `0.012404116869218829` /
  `0.2281928385615599`. HDF5 deployment export is verified for both; current
  VOI-priority full-block export is
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock`.
  Best measured held-out quality is now y-axis slabwise waterz, slab
  `768/256`, threshold `0.36`, same calibration: aggregate MEC11+12 full
  ARAND/VOI `0.010137039842231244` / `0.20730538422230316`. This improves both
  held-out samples but is not partition-equivalent to the full-block export, so
  treat it as a validated quality candidate and not yet as final deployment
  proof. X-axis slab `768/256` is negative at aggregate ARAND/VOI
  `0.0162419794351934` / `0.25372104473914314`, so the y-axis gain is
  axis-sensitive rather than a generic XY tiling win.
  An exploratory same-setting threshold sweep on the already-inspected MEC11/12
  test blocks found y-axis `768/256`, threshold `0.34` at aggregate full
  ARAND/VOI `0.008043419441299271` / `0.19705134574877856`, ahead of the
  prior `0.36` y-axis point (`0.010137039842231244` /
  `0.20730538422230316`). `0.32` is close (`0.00808479052751726` /
  `0.20173006746063996`) and `0.38` is clearly worse (`0.017533601493767037`
  / `0.2474961442075552`). This is a promising diagnostic, not a locked
  deployment operating point: MEC08/10 is the official validation split and
  must choose the threshold before another final MEC11/12 audit.
  The follow-up values are `0.33`: `0.008059426682463788` /
  `0.20080293819276018` and `0.35`: `0.008024889280985659` /
  `0.19599678807900686`; `0.35` leads only marginally over `0.34`. A wider
  `1024/384` y-slab is a strong negative (`0.05953844433758815` /
  `0.3909622131062548`, including MEC12 `0.09495406043461507` /
  `0.4493539555728385`), so larger overlap/context is not a safe deployment
  heuristic.
  The existing same-threshold full-block `0.34` control averages
  `0.012486049236254682` / `0.22468420257321753`, confirming that the
  y-axis improvement is primarily a local hierarchy/layout effect, not simply
  scalar threshold tuning.
  Synthetic slab stitching is now prototyped: a full MEC12 synthetic-slab smoke
  test with 4 z-slabs, 12-slice overlap, and 1937 accepted contact pairs was
  partition-equivalent to the source over 195312500 non-ignore voxels, with 0
  split labels and 0 merge labels. Main remaining gap for 2PB deployment is
  real slabwise waterz inference plus cross-slab stitching, not single-block
  quality on these held-out blocks.
- Real slabwise waterz was tested. The first MEC12 slab48/overlap12 run exposed
  a contact-policy bug: accepting a pair by `contact/min(side_counts)` caused
  massive false merges and full ARAND/VOI `0.4114/1.6350`. The current code now
  counts each side over the full overlap and requires both left and right
  overlap ratios to pass. Re-stitching the saved slab48 outputs with this fix
  improved to about `0.0194/0.3104`. A corrected slab96/overlap32 SiFlow run
  improved further to full ARAND/VOI `0.01100604415162898` /
  `0.19367825102181688`, slab-seam `0.011420920051073535` /
  `0.16024846914997412`, but still trails the full-block export for MEC12
  (`0.004733488986317758` / `0.1391462688022312`). Y-axis slab768/overlap256
  then improved MEC12 to `0.0017551312249106665` / `0.11586653310233763` and
  MEC11 to `0.01851894845955182` / `0.2987442353422687`, giving the best
  aggregate above. X-axis slab768/overlap256 worsened MEC12 to
  `0.008688286060755335` / `0.1771920520371964` and MEC11 to
  `0.023795672809631463` / `0.33025003744108983`. Conclusion: corrected ID
  stitching works and axis/slab size is a quality hyperparameter, but
  independent slab-local waterz hierarchies are axis-sensitive and still not
  production-equivalent to full-block waterz.
- Best deployable global R48 waterz after Stem-micro B: bias
  `(-0.20,-0.45,-0.45)`, temperature `1.15`.
  Best VOI row: `VOI=0.956401`, `ARAND=0.173080`, postprocess about
  `17.51s`; best ARAND row: `ARAND=0.172876`, `VOI=0.964369`.
- Strongest candidate improvement: blockwise waterz prod, chunk
  `32x512x512`, halo `8x64x64`, threshold `0.20`, full ABC
  `VOI=0.937941`, `ARAND=0.185819`, postprocess about `22.24s`.
  This needs confirmation because ARAND is worse than global waterz.
- Conservative seeded-RAG is negative: `VOI=1.8250`, `ARAND=0.3759`,
  postprocess about `181s`. It is volume-specific and not a universal EM method.

## MEC Slab Stitching Prototype

Prototype script:

`/volume/med-train/users/dchen02/code/dbMiM/scripts/stitch_mec8nm_slab_segmentations.py`

Real slabwise scripts:

- `/volume/med-train/users/dchen02/code/dbMiM/scripts/infer_mec8nm_slabwise_stitch.py`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_slabwise_stitch_siflow.sh`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/restitch_mec8nm_saved_slabs.py`

Validated smoke tests:

- Controlled connected-object test:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/slab_stitch_synthetic_unit_20260711/stitch/stitch_summary.json`.
  Shape `[72,96,96]`, 3 slabs, accepted contacts `6/6`,
  partition-equivalent true, reference/stitched labels `4/4`, split/merge
  labels `0/0`.
- MEC12 crop diagnostic:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/slab_stitch_smoke_mec8nm12_thr036_crop_diag_20260711/stitch_summary.json`.
  Crop `[64,512,512]`, 3 slabs, accepted contacts `268/268`,
  reference/stitched labels `987/988`, split/merge labels `1/0`; this shows one
  disconnected same-source-ID case in a crop, with no false merge.
- MEC12 full-volume synthetic-slab smoke:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/slab_stitch_smoke_mec8nm12_thr036_full_20260711/stitch_summary.json`.
  Shape `[125,1250,1250]`, 4 z-slabs, slab size `48`, overlap `12`, accepted
  contacts `1937/1937`, partition-equivalent true, reference/stitched labels
  `5963/5963`, split/merge labels `0/0`, elapsed `47.699s`.

Interpretation: the sparse overlap-contact/union remap layer is valid for
recovering global IDs when slab-local labels are consistent up to ID offsets.
The next required production test is not another synthetic split; it is real
slabwise R57 calibrated waterz inference on adjacent overlapping slabs, followed
by the same contact/union stitching and held-out seam/nonseam metric comparison
against the validated full-block export.

Latest real slabwise evidence:

- `mec8nm-r57-slabwise-3abc868a` / UUID
  `26e950f9-ce98-4a5c-be82-26718da05e2c`: slab48/overlap12, first run with the
  old permissive contact policy. Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_fullblock`.
  This succeeded but is not a valid candidate: full ARAND/VOI
  `0.41142833541688406` / `1.634950951942724`, contacts accepted `3751/8574`,
  with massive false merges.
- Saved-slab restitch sweep after fixing bidirectional contact acceptance:
  local
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/restitch_mec8nm12_slabwise_waterz_contact_sweep_20260711`,
  TOS
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/restitch_mec8nm12_slabwise_waterz_contact_sweep_20260711`.
  Best completed points: `min_contact=1, ratio=0.70`,
  ARAND/VOI `0.01939665858432904` / `0.31143769709526037`; `min_contact=1,
  ratio=0.50`, ARAND/VOI `0.019400472486658327` /
  `0.31037431592651155`; `ratio=0.98` is too strict and over-splits.
- `mec8nm-r57-slab96-3be17ec` / UUID
  `752fb9cb-f55e-4a37-b0f3-cdc28868984a`: corrected slab96/overlap32 with
  `min_contact=64`, `min_overlap_ratio=0.5`. Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_slab96ov32_bidir`.
  Local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_slab96ov32_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_slab96ov32_bidir`.
  Full ARAND/VOI `0.01100604415162898` / `0.19367825102181688`; slab-seam
  `0.011420920051073535` / `0.16024846914997412`; inference `78.516s`,
  waterz postprocess `77.323s`, contacts accepted `446/5775`.
- `mec8nm-r57-yaxis-fd50f7cb` / UUID
  `0762e685-662a-4189-a221-9f936e48c9f6`: corrected y-axis slab768/overlap256
  on MEC12. Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_yaxis768ov256_bidir`.
  Local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_yaxis768ov256_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_yaxis768ov256_bidir`.
  Full ARAND/VOI `0.0017551312249106665` / `0.11586653310233763`; slab-seam
  `0.00090018686490001` / `0.05213740987928195`; inference `73.699s`, waterz
  postprocess `72.295s`, contacts accepted `236/4761`; reference equivalence
  false with split/merge labels `777/847`.
- `mec8nm-r57-yaxis11-ca098737` / UUID
  `0ac76616-c011-44ef-948a-ae41c9690172`: same y-axis slab768/overlap256 on
  MEC11. Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_yaxis768ov256_bidir`.
  Local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_yaxis768ov256_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_yaxis768ov256_bidir`.
  Full ARAND/VOI `0.01851894845955182` / `0.2987442353422687`; slab-seam
  `0.016594606689556257` / `0.17560140352178374`; inference `73.304s`,
  waterz postprocess `82.999s`, contacts accepted `675/10545`; reference
  equivalence false with split/merge labels `5477/5729`.
- `mec8nm-r57-xaxis12-fc7a1ad4` / UUID
  `4e1ac535-d18f-434e-9ffc-0da3e1f2c606`: x-axis slab768/overlap256 on MEC12.
  Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_xaxis768ov256_bidir`.
  Local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_xaxis768ov256_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_xaxis768ov256_bidir`.
  Full ARAND/VOI `0.008688286060755335` / `0.1771920520371964`; slab-seam
  `0.1356215866736129` / `0.6768592154736054`; inference `74.081s`, waterz
  postprocess `75.116s`, contacts accepted `675/15350`; reference equivalence
  false with split/merge labels `1875/1822`.
- `mec8nm-r57-xaxis11-59db30a6` / UUID
  `cd189657-ea56-485b-995e-b2796a41fc23`: same x-axis slab768/overlap256 on
  MEC11. Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_xaxis768ov256_bidir`.
  Local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_xaxis768ov256_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_xaxis768ov256_bidir`.
  Full ARAND/VOI `0.023795672809631463` / `0.33025003744108983`; slab-seam
  `0.0521669754117593` / `0.32409360547645377`; inference `74.639s`, waterz
  postprocess `83.775s`, contacts accepted `1273/23072`; reference equivalence
  false with split/merge labels `4832/4781`.

Next direction: do not keep tuning scalar contact thresholds. Larger postprocess
slabs help and y-axis slabwise is now the best held-out score, but the remaining
gap is slab-local hierarchy inconsistency and axis sensitivity. The credible
2PB path is either larger bounded XY postprocess tiles or saving enough
fragment/hierarchy/affinity state to do a sparse cross-slab re-agglomeration,
not just ID remapping of independently cut waterz outputs. The completed
confirmation wave is: `0.33` y-axis `768/256` (UUID
`c08bd83a-9470-485d-be85-fd5973d8fd18`), `0.35` y-axis `768/256` (UUID
`7e91c732-150f-47bd-81be-a6e3b46dc801`), and a negative `0.34` y-axis
`1024/384` stress test (UUID `8525897d-af37-475a-9af2-7ff806e27e0a`). The
next active wave evaluates the fixed `768/256` threshold neighborhood on
validation MEC08/10 before treating a result as deployable.

## Stem-micro SiFlow Packs

All packs are on `cn-shanghai/changliu`, pool `med-model`, instance
`sci.g21-3`, 8 GPUs per pod. Each pack runs four 2-GPU stage commands via
`CUDA_VISIBLE_DEVICES`.

| Pack | UUID | Status | Stages |
|---|---|---|
| `stem-micro-pack-a` | `5930d7d6-85ec-428f-b78b-fd429d0f862e` | Succeeded | `postproc_r48_waterz_global_fast`, `postproc_r48_waterz_auto_median`, `postproc_r48_waterz_scoring_robust`, `postproc_r48_fast_metric_overhead` |
| `stem-micro-pack-b` | `c3858a35-02d8-45ae-9438-344246b1ea49` | Succeeded | `postproc_r48_waterz_wide_calib`, `postproc_r48_waterz_auto_quantile`, `postproc_r48_waterz_no_label_default`, `postproc_r48_waterz_channel_temp` |
| `stem-micro-pack-c` | `2eeca7c5-ff67-45dc-9029-9d3609d807ea` | Failed wrapper; useful stages complete | `postproc_r48_waterz_seed_density`, `postproc_r48_blockwise_waterz_prod`, `postproc_r48_blockwise_waterz_largehalo`, `postproc_r48_mws_affogato_probe` |
| `stem-micro-pack-d` | `d7ebc7ec-d5c5-44d4-87e3-1f30e707fca2` | Succeeded | `postproc_r48_lrag_waterz_looc`, `postproc_r48_lrag_waterz_all`, `postproc_r48_lrag_gt_looc`, `postproc_r48_lrag_abc_ensemble` |
| `stem-micro-pack-e` | `de8d35eb-d5b8-431d-8c22-ade09901ae19` | Running at `2026-07-04T06:43:36Z` | `postproc_r48_blockwise_waterz_calib`, `postproc_r48_blockwise_waterz_prod_lite`, `postproc_r48_blockwise_waterz_chunk40`, fixed `postproc_r48_mws_affogato_probe` |

Pack C failure root cause: optional `r48-mws-affogato-probe` raised
`ModuleNotFoundError: No module named 'affogato'` during `find_spec` on a
submodule. The waterz/blockwise stages exited 0 and uploaded results. The probe
script has been fixed to record missing optional modules instead of crashing.

Pack E is running. It was submitted specifically to confirm whether blockwise
waterz is a real gain and whether blockwise calibration with temperature `1.15`
can recover the global waterz ARAND improvement.

## Stem-Micro Method Design

Main proposed method:

`raw EM + backbone affinities -> calibrated edge probabilities -> fragments -> local hierarchy -> cross-block seam graph -> sparse global ID remap -> uncertainty/version metadata`

The paper-level claims should be built around:

- calibrated hierarchical post-processing that preserves waterz/MWS robustness;
- a minimal-hyperparameter policy that chooses calibration, solver, hierarchy
  cut, and halo/reconciliation confidence from volume statistics;
- merge-rank distillation or score calibration rather than direct binary
  connected components;
- blockwise execution with stable IDs, re-agglomeration artifacts, and
  CAVE/Neuroglancer-compatible outputs;
- accuracy, runtime, memory, seam quality, and proofreading-burden metrics.

Do not spend the next main wave on pure GPU CC, seeded-RAG, or scalar
learned-RAG threshold sweeps. They remain ablations. The credible Nature
Methods route is calibrated blockwise hierarchy plus signed/repulsive graph
constraints and deployment-grade artifacts.

Old `archx-pack-a/b/c` tasks were stopped after the user requested Stem-micro
naming:

- `06623ed2-0838-4666-9b32-0b46f80719af`
- `4e24e026-53f3-4098-ae5d-3c2d30366469`
- `0255832c-b5ad-4f4c-a138-b9565cc14ad5`

## Output Locations

Sub-stage outputs are uploaded under:

`tos://agi-data/users/dchen02/dbmim/outputs/<stage_output>`

Pack logs are uploaded under:

`tos://agi-data/users/dchen02/dbmim/outputs/<pack_name>`

A machine-readable state snapshot is also written to:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/analysis_postprocess_compare/task_status.json`

## Code Changes To Notice

- `scripts/evaluate_cremi_segmentation.py` now supports label-free
  auto-calibration via `--auto-calibration-targets` and
  `--auto-calibration-quantiles`.
- `../nnEM-Seg-diff-postprocess/scripts/submit_siflow_postprocess_experiments.py`
  now has `stem-micro-pack-a/b/c/d/e`.
- `../nnEM-Seg-diff-postprocess/scripts/dbmim_transfer/probe_external_postprocess_backends.py`
  probes optional MWS/GASP/affogato/mwatershed availability without blocking the
  main waterz experiments.
- `scripts/evaluate_cremi_blockwise_scale.py` now supports
  `--calibration-biases` and `--calibration-temperatures`, so blockwise waterz
  can test the same logit calibration as global waterz.

## Result Files

Latest parsed comparison table:

`/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/outputs/tos_fetch/stem_micro_20260704/stem_micro_summary_table.csv`

Fetched TOS outputs:

`/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/outputs/tos_fetch/stem_micro_20260704/`

## Monitoring Hint

Use the SiFlow SDK with credentials loaded from the secret env files; do not
print credentials. Query `Running`, `Queueing`, `Pending`, `Succeeded`,
`Failed`, and `Stopped` tasks for owner `dchen02`, resource pool `med-model`,
and task names containing `stem-micro-pack`.

Before local GitHub/Hugging Face/TOS/SiFlow SDK network calls, clear inherited
proxy variables:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```

The proxy `192.168.32.28:18000` repeatedly caused local Git and SDK commands to
hang; direct access worked after unsetting it. Do not add permanent proxy
exports to repo scripts.

## 2026-07-11 R58/R59 Official-Validation Handoff

The R58/R59/R60 bounded training wave has completed. Keep the data partition
strict: training is MEC01--07 plus MEC09; official selection is MEC08 plus
MEC10; MEC11 plus MEC12 remain frozen until exactly one full-block candidate
is selected.

- R58 crop screen: zero bias, temperature `1.15`, threshold `0.50` gives
  mean MEC08/MEC10 ARAND/VOI `0.02254600366656634` /
  `0.3111834507513943`.
- R59 crop screen: best ARAND is `0.02268401440455803` with VOI
  `0.28737279263420173` at fixed bias `(-0.20,-0.45,-0.45)`, temperature
  `1.00`, threshold `0.50`; best VOI is `0.27994066510403215` at zero bias,
  temperature `1.15`, threshold `0.50`, with ARAND `0.022743702753899475`.
- R60 crop screen is negative, best mean `0.05509759264176034` /
  `0.44643670309410544`; it is eliminated and must not receive full-block
  GPU time.

Crop scores are not deployment scores. Full-block chunk normalization shifts
R58 affinity p05 from approximately `0.09/0.19/0.17` on crops to
`0.876/0.938/0.931` (`z/y/x`) on MEC08. The initial R58 high-grid MEC08
point at zero bias, temperature `1.00`, threshold `0.25` overmerged
(`0.4500899672038762` ARAND, `1.677020781243909` VOI, `n_pred=5096`,
`n_gt=62`), so the paired low threshold grid is required before interpreting
the model.

Active work is deliberately bounded to three 8-GPU full-block waterz jobs
only (24 GPUs total): R58 high grid UUID
`4d7bf61b-0cd6-4bea-9596-5e694551d5be`, R58 low grid UUID
`335a419d-5858-410e-b1d2-db1f3589a7e2`, and R59 comprehensive grid UUID
`f2a1acaf-329c-4858-a163-1e758a0a20d4`. They evaluate MEC08/MEC10 at
`32x512x512` chunks with `8x64x64` halo across temperature `1.00/1.10/1.15/1.20`.
When all three finish, merge the R58 high/low grids, compare paired full-block
MEC08/MEC10 records by mean ARAND and mean VOI, choose a single configuration,
and only then run one locked MEC11/MEC12 audit. Do not submit any additional
GPU task while these run.

Selection is fixed before grid completion: retain only paired `region=full`,
`backend=waterz` MEC08/MEC10 rows under one checkpoint/variant/temperature/
threshold; remove configurations Pareto-dominated in mean ARAND and mean VOI;
then minimize `sqrt(mean_ARAND * mean_VOI)`. Exact ties use the lower
worst-block geometric value, then lower mean inference plus postprocess time.
Do not choose a distinct threshold for each validation block.

### Prepared Normalization Control

The evaluator and blockwise launcher now support
`INFERENCE_NORMALIZATION=volume|window`. The default `volume` preserves the
active jobs exactly. The opt-in `window` mode applies the existing 1/99
normalizer to every model-sized `32x160x160` window, matching the training
crop scope instead of the `40x576x576` blockwise extended-chunk scope. Its
value is emitted in throughput/metric records and included in group keys.

This control has passed Python compilation, launcher syntax, CLI, and CPU
dummy-model smoke checks, and has been copied consistently into the next-job
bundle. It has not been run on MEC08/10, must not be mixed with `volume`
metrics, and must not trigger a new GPU job while the three active validation
jobs are running.

### 2026-07-11 Contract Pivot

The original R58/R59 `volume` full-block grids were stopped after their saved
MEC08 partial stdout showed systemic severe splitting across their tested
ranges. They are diagnostic negatives only, not completed paired official
validation. Preserve and consult:

- `outputs/live_task_logs/mec8nm_r58_high_volume_stopped_20260711T0638Z.log`
  for stopped UUID `4d7bf61b-0cd6-4bea-9596-5e694551d5be`;
- `outputs/live_task_logs/mec8nm_r58_low_volume_stopped_20260711T0645Z.log`
  for stopped UUID `335a419d-5858-410e-b1d2-db1f3589a7e2`;
- `outputs/live_task_logs/mec8nm_r59_volume_stopped_20260711T0645Z.log`
  for stopped UUID `f2a1acaf-329c-4858-a163-1e758a0a20d4`.

The active replacement is R58 with `INFERENCE_NORMALIZATION=window`, zero
bias, temperature `1.15`, and the same MEC08/10 full-block geometry. UUID
`448f6438-6394-4d2b-b08c-51ca47c8c213` covers threshold `0.12--0.65`; UUID
`d92eab49-b3ed-4987-abd2-776cd1058c76` covers the non-overlapping
`0.70--0.98` extension. The first task logged
`inference_normalization=window`, but its first MEC08 point remained poor
(`0.44129475` ARAND, `1.64796710` VOI, `n_pred=10257` at `0.12`). Do not
submit R59 window variants until the complete paired R58 window curve shows
whether high waterz thresholds recover the fragment topology. At record time
the low/mid task was Running and the high extension was Pending; allocation
was at most 16 GPUs.

### Seed-NMS Control, 24-GPU Cap Reached

`maxima_distance` historically ignored `seed_distance` for normal mahotas
regional maxima, using it only for a no-seed grid fallback. The evaluator now
has default-off `seed_nms_radius`; positive values perform deterministic
spatial suppression of regional maxima before watershed. It is propagated to
the waterz fragment cache key, output records, group keys, and launcher. CPU
smokes passed: a 144-peak map becomes 27 seeds at radius 8, and mahotas
fragment labels fall from 1258 to 126 at radius 10.

Third active 8-GPU official-validation task: UUID
`a10cc330-aa5a-4579-85ae-2ad97dc7d592`, task
`mec8nm-r58-winnms10-54d0b780`, run
`eval_mec8nm_r58_seed1789_fullblock_val_window_t115_nms10_grid`. It differs
from R58 window legacy-seed runs only by `seed_nms_radius=10`, and evaluates
MEC08/10 with thresholds `0.02--0.95`. Together with UUIDs
`448f6438-6394-4d2b-b08c-51ca47c8c213` and
`d92eab49-b3ed-4987-abd2-776cd1058c76`, active allocation is exactly 24 GPU.
Do not submit anything else before these jobs complete; keep MEC11/12 frozen.

### Prepared R61 Topology-Mix Train Control

An audit of 128 bounded train crops found 17 (`13.28%`) with MEC08/10-like
simple topology (`fg>=.95`, `boundary<=.05`, `unique<=5`), mainly from
MEC01/07/09. R60's restriction to MEC07/09 therefore was not a valid test of
explicit simple-crop mixture: it still drew random locations and shortened
training.

`EMVolumeDataset` now accepts default-off `simple_sampling_*` options. When
selected by probability it scores bounded label-aligned candidates from the
same source block toward high foreground, low boundary, and low unique count;
otherwise it preserves existing random/boundary behavior. Default R58 behavior
is unchanged. MEC07 smoke: random crops averaged `0.95145/0.09305/9.75`
(foreground/boundary/unique), while forced simple selection produced
`1.0/0.0/1.0` in all eight test draws.

`configs/finetune_mec8nm_r61_seed1981_topology_mix_stage1.yaml` is a strict
R58 control with all eight train blocks and only `simple_sampling_probability`
`.50`, attempts `4`, min foreground `.95`, max boundary `.05`, max unique
`5`, plus new seed/output values. It passed config-delta, dataset construction,
YAML, compilation, and diff checks. It is prepared only; do not submit it until
the active R58 postprocess wave releases the 24-GPU cap and its official
MEC08/10 evidence is reviewed.

Attempts were reduced from 16 after warm-cache I/O measurement: random reads
were about `0.02328s` per crop, probability `.50` with four candidates
`0.04480s`, and probability `.50` with sixteen `0.11567s`. The four-candidate
MEC07 simple smoke retained 32/32 eligible samples versus 14/32 random, so it
keeps the topology signal without the fivefold read-cost regression.

### R61 Running, Legacy-Low Stopped

Stopped legacy low/mid R58 window UUID
`448f6438-6394-4d2b-b08c-51ca47c8c213` after preserving
`outputs/live_task_logs/mec8nm_r58_window_legacy_low_stopped_20260711T0730Z.log`.
Its complete MEC08 and substantial MEC10 prefix remained severely split, so
the slot was reassigned rather than letting an already ruled-out range run.

R61 is submitted and running: UUID `261ae989-d95c-49f1-9bd1-9b0195d69a79`,
task `mec8nm-r61-topomix-81ae3497`, output
`finetune_mec8nm_r61_seed1981_topology_mix_stage1`. Startup verified R61
runtime config, all train IDs `[1,2,3,4,5,6,7,9]`, 8 paths, 8 GPUs,
`waterz=true`, and 214 R57 pretrained keys. Its post-train evaluation is crop
MEC08/10 only; it is not a full-block selection or test run.

Active allocation is now exactly 24 GPUs: R58 legacy-high, R58 NMS=10, and
R61 topology-mix. Do not submit a fourth job; continue freezing MEC11/12.

### Legacy-High Stopped After Paired Failure

Saved latest legacy-high log:
`outputs/live_task_logs/mec8nm_r58_window_legacy_high_stopped_20260711T0731Z.log`.
Stopped UUID `d92eab49-b3ed-4987-abd2-776cd1058c76` after 13 paired MEC08/10
high-threshold records (`.70--.94`) showed a best at `.86` of mean
ARAND/VOI `0.40993242/1.47345979` and balanced `0.77718655`; remaining MEC08
`.96/.98` did not improve. This is a conclusive negative, not a candidate.

Current active allocation is two 8-GPU jobs (16 GPUs): R58 NMS=10 and R61
topology-mix. Keep the released slot idle until fresh evidence justifies a
new controlled run, and continue freezing MEC11/12.

### R58 Seed-NMS=10 Full-Block Result

R58 seed-NMS=10 completed successfully on 2026-07-11. Task UUID
`a10cc330-aa5a-4579-85ae-2ad97dc7d592`
(`mec8nm-r58-winnms10-54d0b780`) held R58 fixed except for
`seed_nms_radius=10`, and used window normalization, zero bias, temperature
`1.15`, and MEC08/MEC10 full-block evaluation only. Its TOS result prefix is
`tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r58_seed1789_fullblock_val_window_t115_nms10_grid`.
All 12 uploaded objects completed without failure. The independently downloaded
records are at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r58_winnms10_20260711/`.

The complete 15-threshold paired CSV leaves only `.60` and `.80` on the
mean-ARAND/mean-VOI Pareto frontier. Fixed selection chooses `.80`: mean
ARAND/VOI `0.41581373/1.48174353`, balanced `0.78493905`, worst-block
geometric `0.83014687`, and mean inference-plus-postprocess `86.131s`. The
per-block result is MEC08 `0.44237703/1.55782009` (`n_pred=217`) and MEC10
`0.38925043/1.40566697` (`n_pred=595`). It is `51.6297%` worse than the R57
official anchor balanced value `0.51766846` at threshold `.16`.

NMS was mechanically effective but scientifically negative: MEC08 initial
region-graph nodes fell from about `192874` in the legacy path to `53492`
(about `3.6x` fewer), yet neither ARAND nor VOI recovered. R58 NMS=10 is
rejected. Do not expand to an NMS-radius sweep, do not mix its records with
legacy-seed metrics, and do not run MEC11/12. The code path remains default-off
for separately justified future controls.

R61 topology-mix training plus its crop MEC08/MEC10 screen is the only active
work (8 GPUs). Keep the released capacity idle. A full-block MEC08/MEC10
validation must still succeed after R61's crop screen before any test audit can
be considered.

### R61 Topology-Mix Crop Screen Rejected

R61 finished successfully: UUID `261ae989-d95c-49f1-9bd1-9b0195d69a79`, task
`mec8nm-r61-topomix-81ae3497`, step `12000` in about `1132s`. Its checkpoints
remain in the R61 TOS train prefix, but it is not a deployment candidate. The
final log is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r61_topology_mix_20260711T074620Z.log`.
The task uploaded 19 main objects, including the complete val crop records and
no MEC11/12 crop output.

R61's standard volume-normalized crop CSV has 40 paired MEC08/MEC10 waterz
configurations. The only Pareto point is fixed bias `(-.20,-.45,-.45)`,
temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.037270746433/0.373094946903`, balanced `0.117921699281`; MEC08 is
`0.042754743742/0.409151788696` and MEC10 is
`0.031786749123/0.337038105111`. Downloaded evidence:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r61_topology_mix_20260711/cremi_segmentation_metrics.csv`.

For a same-code, same-crop comparison, the downloaded R58 CSV at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r58_balanced_crop_20260711/cremi_segmentation_metrics.csv`
selects zero bias, `t=1.15`, `.50`, with
`0.022546003667/0.311183450751`, balanced `0.083761227436`. R61 is strictly
dominated: ARAND `+65.3098%`, VOI `+19.8955%`, balanced `+40.7832%`. Do not
spend an MEC08/MEC10 full-block job on R61 and do not run MEC11/12.

R61 also exposed a launcher-only robustness issue: val-only runs attempted an
ignored upload of a missing test-crop directory. The stage launcher now uploads
only requested split directories that exist; `bash -n` plus val-only and
missing-directory smoke checks passed. It changes no experiment result.

All MEC jobs are now terminal and SiFlow reports zero running tasks. Keep GPU
capacity idle until a new, separately justified training or postprocess
hypothesis has local evidence. There is no current test/deployment candidate.

### R62 Strict-Accept Topology Control Running

The R61 audit found that the original simple branch was too permissive: under
`p=.50` and four candidates, 62/128 draws tried simple sampling but only 16
actually satisfied the requested topology; 46 selected a noneligible
"closest" crop, with MEC02 and MEC04--06 accepting none. A default-off
`simple_sampling_require_eligible` control now makes noneligible attempts fall
back to conventional sampling. This preserves old behavior when false.

Local R62-policy simulation at `.50` had 62 attempts, 16 true accepted simple
crops, and 28/128 final eligible samples. Its raw fg/boundary/unique profile
`.8879/.2192/21.03` is closer to random R58
`.8763/.2452/24.02` than R61's forced-closest `.9038/.1983/19.45`. Config
delta, real HDF5 accepted/fallback branch smoke, build-dataset, compilation,
and launcher syntax passed.

Submitted one controlled R62 task: UUID
`e0dead92-7a8d-466e-90bc-b42c4822efa5`, task
`mec8nm-r62-topoaccept-8a44b36c`, config
`configs/finetune_mec8nm_r62_seed1993_topology_accept_stage1.yaml`. It retains
R61's all-eight-block, `p=.50`, four-candidate setup and changes only seed,
output, and `simple_sampling_require_eligible=true`. It runs on 8 Shanghai
`med-model` GPUs with val-only waterz crop evaluation; TOS output is
`tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r62_seed1993_topology_accept_stage1`.
Startup confirmed eight train paths, eight GPUs, and waterz; step 420 was
healthy when recorded. Do not submit another job. R62 needs paired crop gains
over R58 before any full-block work; MEC11/12 remain frozen.

### R62 Strict-Accept Crop Result Rejected

R62 completed successfully: UUID `e0dead92-7a8d-466e-90bc-b42c4822efa5`, task
`mec8nm-r62-topoaccept-8a44b36c`, 12000 steps in `1133.85s`. It uploaded the
standard 19 train/val objects and no test output. Official downloaded evidence
is under
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r62_topology_accept_20260711/`.

Its 40 paired crop configurations have two non-duplicate Pareto choices; fixed
selection picks zero bias, `t=1.15`, `.50`: mean ARAND/VOI
`0.029136764797/0.352282465789`, balanced `0.101313233823`; MEC08
`0.052355928557/0.497306034002`, MEC10
`0.005917601036/0.207258897576`. This improves R61's balanced result by
`14.0843%`, but R58 remains strictly better on both metrics:
`0.022546003667/0.311183450751`, balanced `0.083761227436`. R62 is
`+29.2325%` ARAND, `+13.2073%` VOI, and `+20.9548%` balanced versus R58.

The gate makes MEC10 much better but MEC08 materially worse. Its selected M08
affinity p05/Dice/IoU is also below R58 (`.063719/.155971/.135943`,
`.96616346/.93454183` versus `.119049/.219758/.197723`,
`.97928067/.95940243`), a compatible diagnostic but not a causal result.
Reject R62 before blockwise validation. Keep the new gate default-off as a
correctness control only; do not sweep its rate or attempts without new local
evidence. All MEC tasks are terminal, GPU capacity remains idle, and MEC11/12
plus deployment remain frozen because no crop-selected full-block candidate
exists.

### R59 Global-Normalization Full-Block Validation Rejected

The broad first global-normalization task UUID
`dfe11836-34ba-4e3e-bf96-c8dcd3d609f2` was intentionally stopped after its
six calibration variants, 27 thresholds, and three regions proved too broad
for a diagnostic. Its retained partial log is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/mec8nm_r59_globalnorm_stopped_20260711T0852Z.log`.
It did not produce a selection artifact and did not read MEC11/MEC12.

The lean replacement UUID `a972c171-2b37-4283-a15c-b370fa956753`
(`mec8nm-r59-globalhigh-4bb3d3b9`) completed successfully in `33m32s`. It
evaluated only MEC08/MEC10 with the R59 DPP checkpoint, full-volume `global`
normalization, chunks `(32,512,512)`, halo `(8,64,64)`, waterz,
`min_boundary=4`, zero affinity bias, temperatures `1.00/1.15/1.20`,
thresholds `.55--.95`, and `region=full`. All result uploads to
`tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r59_seed1789_fullblock_val_global_highfull`
succeeded. The independently downloaded evidence is under
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r59_globalhigh_20260711/`,
and the full task log is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r59_globalhigh_20260711T093001Z.log`.

The official CSV has exactly 54 rows: 27 complete MEC08/MEC10 pairs, all with
`region=full`, `backend=waterz`, `inference_normalization=global`, and
`min_boundary=4`. The Pareto set is only `t=1.20` at `.85/.90/.95`; fixed
balanced selection chooses `t=1.20, threshold=.85`:

- mean ARAND/VOI: `0.392960605259/1.382466356900`;
- balanced geometric value: `0.737058217753`;
- worst-block geometric value: `0.846541532551`;
- mean inference plus postprocess: `66.581s`;
- MEC08: `0.441996426221/1.621353757227`, `n_pred=219`;
- MEC10: `0.343924784297/1.143578956573`, `n_pred=577`.

Relative to the R57 full-block anchor
`0.252279795600/1.062235821000`, balanced `0.517668461277`, this is worse by
`55.7638%` ARAND, `30.1468%` VOI, and `42.3804%` on the primary balance.
Global normalization therefore does not rescue the full-block failure; R59 is
rejected. Do not mix it with crop metrics, and do not run MEC11/MEC12,
deployment, or 2PB inference from it.

The next single controlled experiment is R63 photometric domain
randomization. It retains the R59 architecture, train blocks, loss, optimizer,
and seed, changing only intensity augmentation to `p=.75`, gain `.55--1.35`,
bias `-.18--.18`, plus `p=.50` 3x3 XY mean blur. This directly addresses the
measured MEC08 intensity and gradient shift and does not mix in rejected
topology sampling. R63 must first pass a MEC08/MEC10 crop screen.

### R63 Photometric-Domain Run Active

The first R63 submission UUID `b4115cf1-ef43-4659-8616-d5853838b5b3`
(`mec8nm-r63-photodomain-6b28e34f`) failed before training or MEC data
download because its R57-derived bundle lacked the root
`third_party/waterz` source directory. The terminal log is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r63_photodomain_waterz_bundle_failure_20260711T094120Z.log`.
This is a packaging preflight failure, not an experimental result; it did not
read MEC08/MEC10/MEC11/MEC12 or create a model checkpoint.

The R63 bundle was corrected by adding the same WaterZ source used by the
successful R59 evaluation bundle. Static bundle preflight found both
`third_party/waterz` and
`external/dbMiM/third_party/boost_1_84_0`; `setup.py --name` returned
`waterz`. The resubmission UUID is
`2bb42bb7-cbbf-47a3-b280-a0422c5e8682`
(`mec8nm-r63-photo-wzfix-b9c874d0`), still one 8-GPU Shanghai changliu
`med-model` pack and still using the R63 val-only command.

Startup is verified: WaterZ `0.8` built and imported in the pod,
`cuda=true`, `gpu_count=8`, runtime train IDs are `[1,2,3,4,5,6,7,9]`,
eight paths were materialized from TOS, and 214 R57 pretrained keys loaded.
The distributed trainer reports world size 8 and dataset size 32768; it had
reached step 220 without an exception when recorded. `EVAL_SPLITS=val` and
`EVAL_BACKENDS=waterz` remain encoded in the task command, so only MEC08/MEC10
crop evaluation can follow training. MEC11/MEC12, deployment, and 2PB
inference remain frozen.

### R63 Crop Recheck Rejected

R63 training itself completed successfully in the repaired task UUID
`2bb42bb7-cbbf-47a3-b280-a0422c5e8682` after 12000 steps (`1256s`). Its first
embedded crop evaluation is invalid: the stage bundle paired the current crop
evaluator, which always supplies `seed_nms_radius=0`, with an older
`dbmim.postprocess.watershed_fragments` signature. All 80 waterz rows failed
with `TypeError`, yet the old launcher did not use `--fail-on-backend-error`
and uploaded an empty records file. Those empty artifacts are not model
evidence.

The recovery was checkpoint-only, never a retrain. The first recheck UUID
`361a2b49-7337-45b9-ae05-d19649168bc3` correctly failed before evaluation
because the crop CLI does not expose `--seed-nms-radius`; its terminal log is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r63_crop_recheck_cli_failure_20260711T101642Z.log`.
After removing that invalid CLI argument, corrected UUID
`f69d5bad-4d2f-4ddf-87b7-dcf731489e5e` reused the remote R63
`finetuned_latest.pt`, succeeded in `309s`, and enforced both
`--fail-on-backend-error` and an 80-row output assertion. The full task log is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r63_photodomain_recheck_20260711T102507Z.log`; independently downloaded output is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r63_photodomain_recheck_20260711/`.

The successful summary has `num_records=80`, `failures=0`, waterz available,
and sample records exactly `{mec8nm_08.h5,mec8nm_10.h5}`. Removing the 16
duplicate `pred`/zero-bias baseline rows leaves 32 paired calibrations. Its
two-point Pareto set selects zero bias, `t=1.15`, threshold `.50`:

- mean ARAND/VOI: `0.024314198449/0.293735246416`;
- balanced geometric value: `0.084509982090`;
- worst-block geometric value: `0.129345245600`;
- mean inference plus postprocess: `4.770887s`;
- MEC08: `0.042331074359/0.395222488738`, Dice/IoU
  `0.971972989475/0.945474031449`, `n_pred=357`;
- MEC10: `0.006297322539/0.192248004094`, Dice/IoU
  `0.988690440691/0.977633694002`, `n_pred=151`.

Against the same-code R58 crop winner zero bias, `t=1.15`, `.50`
(`0.022546003667/0.311183450751`, balanced `0.083761227436`), R63 lowers
mean VOI `5.6070%` but raises ARAND `7.8426%`, primary balance `0.8939%`, and
worst-block balance `23.1032%`. The trade-off is asymmetric: MEC10 improves
strongly, while MEC08 ARAND rises `183.1844%` and VOI `54.3067%`. R63 is
rejected before full-block validation; do not use its checkpoint for MEC11/12,
deployment, or 2PB inference.

The launch path is hardened: `scripts/run_mec8nm_stage1_siflow.sh` now passes
`--fail-on-backend-error`, and the new
`scripts/run_mec8nm_crop_eval_siflow.sh` is validation-only, pulls a named
TOS checkpoint, asserts 80 waterz rows for MEC08/MEC10, and has no test-split
path. Keep these checks in all future crop screens.

### R64 Bright-Gamma Follow-Up Prepared

R63's strong symmetric gain/bias plus XY blur reduced aggregate gradients but
also moved the normalized training distribution farther from MEC08 and harmed
MEC08 crop quality. A deterministic 128-crop audit is retained at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_aug_distribution_audit_20260711T1028Z.txt`.
After per-crop percentile normalization, MEC08 has mean/std/p05/p50
`.618153/.263203/.127728/.681904` and z/y/x gradients
`.199492/.075573/.084305`. R58 default augmentation differs from its first
ten intensity/gradient statistics by L2 `0.038494`; R63 strong augmentation
is worse at `0.064802` and has excessive one-saturation (`.043720` versus
MEC08 `.011196`).

The targeted gamma-only simulation is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_gamma_candidate_audit_20260711T1030Z.txt`.
Applying an asymmetric bright gamma `p=.90`, exponent `.75--.85`, while
retaining R59's ordinary intensity/noise policy, reduces the same MEC08 L2 to
`0.010121`; its mean/std/p05/p50 are `.612423/.268897/.129770/.661763`.
It corrects the low/mid-tone mismatch without R63's broad gain/bias or blur.

`augment_gamma_probability` is now a validated, default-`.35` dataset and
training-config control. A 40-seed reference test proves the default path is
bitwise identical to the historical augmentation; the R64 real-HDF5 dataset
smoke confirms probability `.90`, range `(.75,.85)`, image
`(1,32,160,160)`, and label `(32,160,160)`. Config
`configs/finetune_mec8nm_r64_seed1789_brightgamma_stage1.yaml` is exactly R59
apart from its output directory, gamma probability, and gamma range. R64 is a
single controlled MEC08-targeted brightness hypothesis and remains val-crop
only until evidence proves otherwise.

### R64 Bright-Gamma Crop Result Rejected

R64 completed successfully in SiFlow UUID
`f4e357c7-a607-4404-94ee-77ebd403dcd5`
(`mec8nm-r64-brightgamma-13903c64`) in `27m05s`. It trained the fixed eight
MEC01--07/MEC09 paths for 12000 steps on one eight-GPU `med-model` pack, then
ran only the MEC08/MEC10 WaterZ crop screen. WaterZ was built and importable,
there were zero backend failures, and task logs contain no MEC11/MEC12 path.
The independently fetched validation evidence is under
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r64_brightgamma_20260711/`.

The CSV has exactly 80 `waterz` rows for MEC08/MEC10. Removing the 16
duplicate `pred`/zero-bias records leaves 32 paired calibrations. Its sole
Pareto point is bias `(-.20,-.45,-.45)`, temperature `1.15`, threshold `.50`:

- mean ARAND/VOI: `0.021321377783/0.267121674862`;
- balanced geometric value: `0.075467888162`;
- worst-block geometric value: `0.108898647791`;
- MEC08: `0.035600267251/0.333113102975`, Dice/IoU
  `.976117487142/.953349295436`, `n_pred=216`;
- MEC10: `0.007042488314/0.201130246749`, Dice/IoU
  `.989626908612/.979466612926`, `n_pred=101`.

Relative to the fixed R58 crop winner, R64 improves mean ARAND `5.4317%`,
mean VOI `14.1594%`, and primary balance `9.9012%`. That aggregate gain is
not admissible: MEC08 ARAND regresses `138.1569%`, MEC08 VOI regresses
`30.0573%`, and worst-block balance rises `3.6434%`. The large MEC10 gain
therefore masks an unacceptable MEC08 failure. Reject R64 before full-block
validation; do not use its checkpoint for MEC11/MEC12, deployment, or 2PB
inference. Do not submit another training run until a new single-variable
MEC08-preserving hypothesis has local evidence.

### R64 Intermediate-Bias Crop Recheck Rejected

The R64 default grid exposed a calibration discontinuity rather than a simple
threshold issue: zero bias at `t=1.15,.50` preserves M08 but fails M10, while
`(-.20,-.45,-.45)` reverses that tradeoff. The validation-only crop runner now
accepts environment-provided threshold/bias/temperature grids and derives its
expected row count from the grid. Its historical default remains 80 rows; the
new R64 scan correctly required 96 rows.

Checkpoint-only UUID `5f7a7ad8-bb8d-49ef-81cd-3b4dbc58dd99`
(`mec8nm-r64-biasmid-dcec396f`) succeeded in `6m36s` with no training. It
evaluated only M08/M10 with fixed temperature `1.15`, the original eight
thresholds, and five global bias triplets linearly spanning zero to
`(-.20,-.45,-.45)`. WaterZ was available, the in-task assertion reported 96
rows, failures were zero, and no MEC11/MEC12 path was read. Downloaded evidence
is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r64_intermediate_bias_recheck_20260711/`.

The 96 rows form 48 paired calibration settings with no duplicate-key removal.
There are two Pareto points. The primary geometric rule still selects the
original `(-.20,-.45,-.45), t=1.15, .50` point exactly, so its unacceptable
M08 result is unchanged. The new intermediate Pareto point
`(-.05,-.1125,-.1125), t=1.15, .50` has mean ARAND/VOI
`0.020017610135/0.295218562188` and balance `0.076873728169`, but its
worst-block balance is `0.112877655965` and MEC08 remains
`0.032764371052/0.388878675428`. It is worse than the selected extreme point
on the primary balance and still fails the M08 protection condition. The
remaining intermediate biases are non-Pareto and substantially worse.

Therefore a denser fixed global calibration grid cannot rescue R64. Preserve
the crop-only runner improvement, but reject R64 conclusively before full-block
validation. MEC11/MEC12, deployment, and 2PB inference remain frozen.

### R65 Balanced-Gamma Dose Control Running

R64's full bright-gamma dose and its denser fixed calibration scan are both
negative. A 64-crop, validation-only intensity audit at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_gamma_dose_audit_20260711T1137Z.txt`
shows that R59's historical gamma policy is closest to M10 while R64's `.90`
bright-gamma policy is closest to M08. The midpoint `p=.50`, exponent
`.75--.85` minimizes the maximum first-ten-statistic L2 across M08/M10 among
the audited doses: `.016754/.025294` versus R59's worst `.037129` and R64's
worst `.035862`. This is distribution evidence only, not a selection result.

`configs/finetune_mec8nm_r65_seed1789_balancedgamma_stage1.yaml` holds R64
fixed except for output path and `augment_gamma_probability: .50`; gamma range,
seed, model, loss, optimizer, train blocks, ordinary intensity/noise policy,
and crop shape are unchanged. YAML delta and real-HDF5 dataset smoke pass.
The staged bundle also passes shell/Python/WaterZ/hash checks.

SiFlow UUID `b449842a-53b8-4781-93cb-797a65dd94a1`
(`mec8nm-r65-gammamid-b5ff05e0`) is the sole active 8-GPU
`cn-shanghai/changliu/med-model` task. Startup confirms WaterZ, CUDA 8 GPUs,
the eight fixed training IDs, 214 loaded R57 keys, and val-only WaterZ
evaluation. It may only screen MEC08/MEC10 crops. Do not start another GPU
task, and do not read MEC11/MEC12 or deploy anything before the same paired
crop gate is applied to R65.

### R65 Crop Gate and Full-Block Validation Rejected

R65 SiFlow UUID `b449842a-53b8-4781-93cb-797a65dd94a1`
(`mec8nm-r65-gammamid-b5ff05e0`) completed successfully in `26m46s`: 12000
steps, WaterZ available, CUDA 8 GPUs, 80 valid M08/M10 crop rows, zero backend
failures, and no MEC11/MEC12 path. Its only crop Pareto point is bias
`(-.20,-.45,-.45)`, temperature `1.00`, threshold `.50`:

- mean ARAND/VOI: `0.014667305322/0.278149880175`;
- balanced geometric value: `0.063872601465`;
- worst-block geometric value: `0.088341945947`;
- MEC08: `0.021810413186/0.357824464262`, Dice/IoU
  `.970439696074/.942576778089`, `n_pred=325`;
- MEC10: `0.007524197458/0.198475296088`, Dice/IoU
  `.985955525689/.972299964712`, `n_pred=164`.

Against R58, R65 improves crop mean ARAND `34.9450%`, mean VOI `10.6155%`,
primary balance `23.7444%`, and worst-block balance `15.9213%`. Its M08
ARAND/VOI are nevertheless `45.9062%/39.7053%` higher. This was strong enough
aggregate and worst-block evidence to justify one restricted full-block
validation, but never a test/deployment release.

Checkpoint-only recheck UUID `a7fe194b-2799-49f4-b056-8caec109bfb2`
(`mec8nm-r65-biasmid-3b192eed`) succeeded in `6m44s` with fixed `t=1.00`,
five global bias triplets, eight thresholds, only M08/M10, zero failures, and
an in-task 96-row assertion. After removing 16 exact `pred`/zero-bias rows,
40 paired configurations remain; its only Pareto point is exactly the default
R65 crop point above. Intermediate fixed bias cannot reduce the M08 tradeoff.
Evidence is under
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r65_balancedgamma_20260711/`
and
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r65_intermediate_bias_recheck_20260711/`.

The generic blockwise runner now accepts an optional named TOS checkpoint, and
`scripts/run_mec8nm_blockwise_val_checkpoint_siflow.sh` forcibly exports only
M08/M10 before delegating to it. This allowed official full-block UUID
`6ed95f00-32ff-48de-a2ea-c32cec72d5ea`
(`mec8nm-r65-fullval-1798381e`) to run without a test-split path. It succeeded
in `16m33s`, with exactly 26 zero-failure full-region waterz rows: global
normalization, chunk `(32,512,512)`, halo `(8,64,64)`, fixed bias
`(-.20,-.45,-.45)`, `t=1.00`, `min_boundary=4`, and thresholds `.35--.95`.

Its only full-block Pareto/selected point is threshold `.90`: mean ARAND/VOI
`0.401543070259/1.443839745931`, balance `0.761422251147`, and worst-block
balance `0.849147790727`. MEC08 is `0.446298691940/1.615626448203`
(`n_pred=255`); MEC10 is `0.356787448577/1.272053043658` (`n_pred=612`).
Relative to the R57 official full-block anchor
`0.252279795612/1.062235821039`, R65 is `59.1658%` worse in ARAND,
`35.9246%` worse in VOI, and `47.0869%` worse in primary balance. Crop gains
therefore do not transfer through global full-block inference. Reject R65
before MEC11/MEC12, deployment, or 2PB inference. The next hypothesis must
address the crop-to-full-block normalization/topology mismatch, not gamma dose.
Full-block evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r65_fullblock_global_20260711/`.

The normalization mechanism itself was then tested without retraining. UUID
`ab25730e-b922-4e22-a1f9-bce15ba46209`
(`mec8nm-r65-fullvol-de7e19fb`) completed successfully on one 8-GPU
`cn-shanghai/changliu/med-model` pack. Its checkpoint-only entrypoint hard
locked samples to M08/M10. The downloaded final CSV has exactly 22 rows: the
two validation blocks times 11 declared thresholds, every row is
`region=full`, `backend=waterz`, `inference_normalization=volume`,
`min_boundary=4`, fixed bias `(-.20,-.45,-.45)`, and `t=1.00`; the failure
ledger is empty. No M11/M12 path was supplied or read.

The only Pareto point is threshold `.65`: mean ARAND/VOI
`0.395251787736/1.420713606624`, balance `0.749359455054`, and worst-block
balance `0.859067495726`. MEC08 remains fragmented at
`0.448038919944/1.647171549973` (`n_pred=784`, versus `n_gt=62`), while MEC10
is `0.342464655527/1.194255663275` (`n_pred=1502`). This is only
`1.5668%/1.6017%/1.5842%` better than R65 global in ARAND/VOI/balance and is
still `56.6720%/33.7475%/44.7566%` worse than the R57 official full-block
anchor. Therefore per-extended-chunk normalization is not a sufficient repair:
the primary issue is learned affinity/topology transfer, not merely one global
intensity scaling choice. Reject R65 volume before M11/M12, deployment, or
2PB inference. Evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r65_fullblock_volume_20260711/`.

### R57/R59 Member-Calibrated Logit Ensemble Rejected

The evaluator now supports compatible multi-checkpoint logits through
`LogitEnsemble`: each member's learned postprocessor runs first, its declared
z/y/x bias and temperature are applied second, and only then are logits mixed
before the outer sigmoid/WaterZ grid. This preserves a meaningful endpoint:
weight one recovers the member's own calibration rather than applying one
incompatible global calibration to every checkpoint. Deterministic stub tests,
real R57/R59 checkpoint construction, staged Python compilation, shell syntax,
and primary-checkpoint byte identity all passed.

Remote smoke UUID `c06b4e23-d1fe-4e95-86f4-4ba3c4e75c5d`
(`mec8nm-r57r59-ens-smoke-a085-61def15d`) used only center M08/M10 crops,
R57/R59 weights `.85/.15`, member biases `(-.20,-.45,-.45)/(0,0,0)`, member
temperatures `1.20/1.20`, and neutral outer calibration. It completed with
WaterZ/CUDA 8, exactly six rows, and an empty failure ledger. This establishes
only the dual-checkpoint execution path, not a selection result.

The sole formal full-block run UUID
`ee52f7b1-5348-408f-82d7-61bc6e7d9372`
(`mec8nm-r57r59-ens-full-a085-4734ed30`) then held those weights and member
calibrations fixed. Its entrypoint forcibly used M08/M10 only, `volume`
normalization, chunks `(32,512,512)`, halos `(8,64,64)`, WaterZ,
`min_boundary=4`, neutral outer calibration, and the 11 declared thresholds
`.10,.12,.16,.20,.24,.28,.32,.40,.50,.60,.70`. It succeeded with exactly 22
rows and an empty failure ledger; no M11/M12 path was present.

The only Pareto point is threshold `.70`: mean ARAND/VOI
`0.394638978896/1.731286401767`, balance `0.826579153965`, and worst-block
balance `0.916424680291`. M08 is `0.456917433662/1.838043665601`
(`n_pred=916`), M10 is `0.332360524129/1.624529137932` (`n_pred=948`). Against
the R57 official full-block anchor, ARAND/VOI/balance are
`56.4291%/62.9851%/59.6735%` worse. It also has `25.2317%` worse VOI and
`12.1457%` worse balance than R59 global, and `21.8603%` worse VOI and
`10.3048%` worse balance than R65 volume. Reject R57/R59 logit reweighting;
do not expand this family into more weights, M11/M12 evaluation, deployment, or
2PB inference. Smoke and full evidence are under
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r57r59_ensemble_smoke_a085_20260711/`
and
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r57r59_ensemble_fullblock_a085_20260711/`.

### R66 Local-Shape-Descriptor Auxiliary Head Rejected

R66 preserved the R57 data regime exactly: MEC01--07/MEC09 training only,
MEC08/MEC10 validation only, the R57 initialization, seed, crop geometry,
affinity loss, optimizer, and ordinary augmentation. The sole architectural
change expanded the affinity head from three to seven outputs. The first three
channels remained WaterZ affinities; four new channels received a modest
foreground-plus-centroid-offset local-shape-descriptor loss with weight `.15`.
The expanded-loader transfer copied the R57 affinity channel bias/scale values
exactly and left the new descriptor branch independently initialized.

SiFlow UUID `4f60e8f4-eda4-45b0-a21d-81fbd981ff68` completed successfully on
one 8-GPU Shanghai `med-model` pod. Runtime checks showed WaterZ, CUDA 8,
214 inherited R57 keys, the eight fixed train paths, and the declared LSD loss.
The validation-only evaluator asserted 80 raw WaterZ M08/M10 rows with finite
metrics and no failures. After removing 16 exact `pred` / zero-bias,
temperature-1.0 duplicates, the 32 paired configurations contain no crop-gate
pass relative to R58.

The best-balance R66 point is bias `(-.20,-.45,-.45)`, temperature `1.15`,
threshold `.22`: mean ARAND/VOI `0.049656091996/0.411782761303`, balance
`0.142994834444`, MEC08 `0.032558920140/0.373421472399`, and MEC10
`0.066753263853/0.450144050207`. Against the R58 crop anchor it is
`120.2434%` worse in ARAND, `32.3280%` worse in VOI, and `70.7172%` worse in
balance; MEC08 regresses `117.8111%/45.7949%` in ARAND/VOI. Reject R66 before
full-block work. Do not inspect MEC11/MEC12, deploy, or schedule 2PB inference
from it. Detailed evidence is in
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r66_lsd_crop_report_20260711.md`.

### R67 Soft-Negative Weighted-MSE Control Running

The post-R66 boundary audit samples 16 deterministic training crops per allowed
block. With historical radius-1 widening, train affinity-positive fractions are
`.788879/.852044/.853948` (z/y/x), while M08/M10 raw labels are
`.960546/.969814/.968969`. R58-style `weighted_mse` with `weight_alpha=1.0`
therefore gives rare train negatives equal aggregate loss mass and individual
negative weights about `3.74/5.76/5.85`. Since all completed full-block models
fragment, R67 reduces only `weight_alpha` to `.5`; every R58 data, model,
seed, optimizer, augmentation, boundary-Dice, BCAR, membrane, and schedule
setting remains fixed.

R67 UUID `94db3a0b-6084-4085-a6ad-1c05f3df07f0` is the only active pack: one
8-GPU Shanghai `med-model` job. The hardened entrypoint fixes MEC01--07/MEC09,
M08/M10 `val`, WaterZ, and an 80-row metric assertion. Startup has verified
CUDA 8, WaterZ, 214 loaded R57 keys, the fixed eight train paths, and remote
`weighted_mse` `weight_alpha=.5`. It is crop-screen-only until a strict paired
gate result is available; do not use M11/M12, full-block inference, deployment,
or 2PB work yet. The exact rationale is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r67_softneg_plan_20260711.md`.

### R67 Soft-Negative Crop Gate Rejected

R67 completed its 12000-step 8-GPU training and its M08/M10-only WaterZ crop
sweep successfully. The remote assertion reports exactly 80 rows, WaterZ only,
finite metrics, and no evaluator failures. The downloaded CSV collapses 16
exact `pred` / zero-bias, temperature-1.0 duplicates into 32 paired
configurations. The reusable paired analyzer finds zero strict R58 gate passes.

The best-balance R67 point is bias `(-.20,-.45,-.45)`, temperature `1.00`,
threshold `.50`: mean ARAND/VOI `0.052563021514/0.399776827409`, balance
`0.144960263451`, M08 `0.016472016743/0.300693537996` (`n_pred=154`), and
M10 `0.088654026284/0.498860116822` (`n_pred=81`). Relative to the R58 anchor,
it is `133.1368%/28.4698%/73.0637%` worse in mean ARAND/VOI/balance. M08
regresses `10.1937%/17.3997%`; M10 regresses `194.1040%/36.2117%`. Reducing
negative boundary pressure reduces predicted instances but replaces
fragmentation with harmful M10 merges. Reject R67 before full-block validation,
M11/M12, deployment, or 2PB inference. Detailed evidence is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r67_softneg_crop_report_20260711.md`.

### R68 Multiscale-Affinity Auxiliary Rejected

R68 UUID `b8939e9c-ffa2-48e5-bed9-a7b173bda89f` completed successfully on one
8-GPU Shanghai `med-model` pod. The runtime contract was satisfied: CUDA 8,
WaterZ, 214 R57 keys, MEC01--07/MEC09 only, six affinity offsets, and BCAR
restricted to the first three WaterZ channels. The M08/M10-only evaluator
returned exactly 80 finite WaterZ rows and no error.

The paired analyzer finds zero strict R58 gate passes. Best balance is zero
bias, temperature `1.00`, threshold `.50`: mean ARAND/VOI
`0.029857502936/0.358199176768`, balance `0.103416309023`; M08 is
`0.052169534998/0.490800733464` with 428 predictions, while M10 is
`0.007545470874/0.225597620072` with 165. Mean ARAND/VOI regress
`32.4292%/15.1087%` and M08 regresses `249.0012%/91.6232%` versus R58.

All three uncalibrated near-neighbor affinity means and fifth percentiles fall
on both blocks, so the long-range auxiliary creates a shared lower-affinity
shift rather than one bad axis. It improves M10 but causes severe M08
fragmentation. Reject R68 before full-block, M11/M12, deployment, or 2PB
inference; do not blind-sweep this auxiliary weight. Report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r68_multiscale_affinity_crop_report_20260712.md`.

### R69 No-Widen Control Queueing

R69 changes only R58 `data.widen_border` from true to false. It preserves full
R58 loss weighting, including genuine instance-boundary negatives, while
removing the artificial one-pixel zero-label band. Config-delta and hardened
entrypoint checks passed locally and inside the staged offline bundle.

SiFlow UUID `ae98ae06-08b2-4fac-b5bf-936196b79b0f`
(`mec8nm-r69-nowiden-30543597`) is the only submitted wave: one 8-GPU Shanghai
`med-model` task. It is queueing due to 8-card resource fragmentation with one
medium-priority task ahead. Keep only this wave. Its evaluator is fixed to
M08/M10 validation, WaterZ, and 80 rows; full-block, M11/M12, deployment, and
2PB inference remain forbidden before a strict R58 crop-gate pass. Plan:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r69_nowiden_plan_20260712.md`.

### R69 No-Widen Crop Gate Rejected

R69 completed successfully with CUDA 8, WaterZ, 214 R57 keys, the eight fixed
train IDs, 12000 steps, and exactly 80 M08/M10 crop metrics. No paired setting
passes the strict R58 gate. Its best point has mean ARAND/VOI
`0.036164090089/0.396930915975`; M08 is
`0.045930476087/0.448332741337`, and M10 is
`0.026397704092/0.345529090612`. Mean ARAND/VOI regress
`60.4013%/27.5553%`, while M08 regresses `207.2635%/75.0425%` versus R58.
Reject R69 before full-block, M11/M12, deployment, or 2PB inference. Report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r69_nowiden_crop_report_20260713.md`.

### R70 FullEM-R33 External-Pretraining Control Queueing

R70 changes only the R58 initializer from PublicEM R57 to FullEM mixed-mask
R33. The checkpoints have identical 214-key topology and shapes. R33 adds
self-supervised exposure to the inherited approximately 485.8 GB FullEM mirror
collection before CREMI affinity finetuning; no external labels are mixed into
MEC supervision. MEC train remains M01--07/M09 and validation remains M08/M10.

SiFlow UUID `d49b444f-fcd6-44c3-8250-7a2eb176f799`
(`mec8nm-r70-fullem-4d74bb4`) is the sole active 8-GPU Shanghai `med-model`
wave, queue rank one with no tasks ahead. Its bundle has a byte-verified R33
asset and locks WaterZ plus 80 metric rows. Do not run full-block, M11/M12,
deployment, or 2PB inference before a strict crop-gate pass. Plan:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r70_fullem_r33_plan_20260713.md`.

### R70 Rejected; R71 EMA Running

R70 FullEM-R33 initialization returned no strict gate pass. Best mean
ARAND/VOI is `0.053447533042/0.487475984541`, with M08
`0.041557721757/0.460674361248`; mean metrics regress
`137.0599%/56.6523%` versus R58. Reject it before full-block or frozen splits.

R71 changes only R58 `train.ema_decay=0.999` and evaluates EMA weights. UUID
`41bc1e3b-6b6e-4fdf-9de9-6af7ab993381`
(`mec8nm-r71-ema-4ad0d84a`) is running on one 8-GPU Shanghai `med-model` pod.
It remains M08/M10 crop-screen-only with WaterZ and an 80-row assertion. Plan:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r71_ema_plan_20260713.md`.

### R71 EMA Crop Gate Rejected and Current MEC Snapshot

R71 completed successfully with remote `ema_config={'decay': 0.999,
'checkpoint_uses_ema': True}`, CUDA 8, WaterZ, 214 R57 keys, 12000 steps, and
exactly 80 finite M08/M10 rows. Its best aggregate point is uncalibrated,
temperature `1.00`, threshold `.50`: mean ARAND/VOI
`0.020969504209/0.299190914044`, improvements of `6.9924%/3.8538%` over R58.
M10 improves to `0.004363798372/0.165133660030`, but M08 regresses to
`0.037575210047/0.433248168057` (`151.3688%/69.1530%` worse). There are zero
strict gate passes. EMA is a mechanism signal, not a selected checkpoint.

The current selected MEC anchors remain R58 for crop screening and R57 for
full-block validation. R58 crop mean ARAND/VOI is
`0.022546003667/0.311183450751`; R57 official M08/M10 full-block mean is
`0.252279795612/1.062235821039`. Do not promote R71, run full-block from it, or
open M11/M12. Detailed R71 evidence is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r71_ema_crop_report_20260713.md`.

For publication comparisons, keep datasets separate. Released dbMiM CREMI
A/B/C full-volume results are R48 `ARAND=.178252, VOI=.962154` and R57
`ARAND=.178248, VOI=.964617`; these are not numerically comparable to MEC. The
local MEC/EMNeXt data card contains no same-protocol published ARAND/VOI table.
Use `/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_absolute_results_and_publication_comparison_20260713.md`
as the compact current handoff.

### R72 Photometric Consistency Crop Gate Rejected

R72 tested the first preferred post-R71 augmentation-consistency hypothesis.
It preserved R58's seed, R57 initialization, eight training blocks, model,
supervised loss, optimizer, and 12k schedule, adding only a weight-`.05`
stopped-teacher probability consistency term between coordinate-aligned
photometric views. SiFlow UUID `ff377784-06ce-4b4e-a4f0-bb7a7b8d0a2f`
(`mec8nm-r72-consistency-90145185`) completed successfully on one 8-GPU
Shanghai `med-model` pod. Startup verified WaterZ, CUDA 8, 214 R57 keys, and
train IDs `[1,2,3,4,5,6,7,9]`. Training reached step 12000 in 2058 seconds.

The remote assertion and independent TOS fetch confirm exactly 80 finite
WaterZ rows, samples exactly M08/M10, and an empty failure ledger. The paired
analyzer found zero strict R58 gate passes. The only Pareto/best-balance point
is bias `(-.20,-.45,-.45)`, temperature `1.15`, threshold `.50`: mean
ARAND/VOI `0.028005094678/0.369409206168`, balance `0.101712043503`; M08
`0.042332990281/0.473893471718`, M10
`0.013677199074/0.264924940619`. Relative to R58, mean ARAND/VOI regress
`24.2131%/18.7111%`, while M08 regresses `183.1972%/85.0222%`.

Reject R72 before full-block validation. Do not sweep its consistency weight
or photometric ranges, and do not run M11/M12, deployment, or 2PB inference.
The current selected anchors remain R58 crop and R57 M08/M10 full-block.
Detailed evidence is
`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/evidence/mec8nm_r72_consistency_crop_report_20260713.md`.

The MEC line is now isolated into the versionable
`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/` workspace. It contains
MEC configs/scripts, runtime source snapshot, split/manifest contract, skill,
and compact evidence; 20 GB HDF5, checkpoints, bundles, and full logs remain
external assets. `python mec8nm/scripts/verify_workspace.py` checks the frozen
split, R58 anchor integrity, absence of large assets, and legacy-runtime sync.

### R73 Decoder-Aware MEC Domain Pretraining Rejected

R73 initialized the exact UNETR-EM encoder, decoder, and affinity head from
PublicEM-R57 and adapted all shared modules for 4000 steps using only raw
M01--07/M09 crops. The label-free objective was 75% masked reconstruction plus
a weak (`.1`) raw-gradient affinity proxy. The verified checkpoint records
`global_step=4000`, 219 model keys, and all 214 shared R57 keys. The following
supervised stage loaded all 214 segmentation keys and held the R58 seed, data,
loss, optimizer, augmentation, and 12000-step schedule fixed.

Two launcher-only failures occurred around the completed pretraining stage:
UUID `4a011e80-1075-4bba-a117-5c5ec9ca9f9d` stopped before model execution;
UUID `2e267c00-7ee4-41bc-8613-5e2ebed5ecb1` completed 4000 steps but stopped
before verification because of a readonly shell assignment. Its checkpoint
was fetched and verified at SHA256
`32ba0c3aa14a59cb3fa285bb0c5875f96574a1e1bd65da2c3751eeeecf3b2262`.
Successful continuation UUID `0a62d570-a2b6-4208-bfde-71df64eabc59`
(`mec8nm-r73-decaware-domain-r2-f8b87367`) reused that exact checkpoint and
completed. The remote gate asserted exactly 80 finite M08/M10 WaterZ rows and
an empty failure ledger; M11/M12 were absent.

The paired analyzer finds zero strict R58 gate passes. Best balance is zero
bias, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.055075340982/0.486239101228`; M08
`0.037673258094/0.443902435184`; M10
`0.072477423870/0.528575767271`. Relative to R58, mean ARAND/VOI regress
`144.2798%/56.2548%`, and M08 regresses `152.0247%/73.3128%`. Reject R73
before full-block, M11/M12, deployment, or 2PB inference. Do not repeat this
raw-gradient pseudo-affinity objective. A future MEC raw adaptation control
must preserve the R57 decoder/head and isolate changes to the encoder.

### R74 Encoder-Only Domain Adaptation Rejected

R74 ran standard masked reconstruction for 4000 steps on raw M01--07/M09
after loading exactly 77 R57 encoder tensors. Its audited hybrid replaced only
those 77 tensors and preserved all 137 R57 decoder/head tensors bit-identically
(hybrid SHA256
`9195891cabe131f555575c3231c7e02f481d5f984daca832cb7f40a88315fca0`).
The subsequent R58-style stage loaded all 214 hybrid tensors and completed
12000 supervised steps.

SiFlow UUID `9d0d3cb6-113a-46da-a3d6-81b1c7d8fa78`
(`mec8nm-r74-encoderadapt-4741d965`) succeeded with exactly 80 finite M08/M10
WaterZ rows and no failures. No strict R58 crop-gate point exists. Best balance
is zero bias, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.035560839373/0.359082802791`; M08
`0.037953939399/0.356534733043`; M10
`0.033167739347/0.361630872539`. Mean ARAND/VOI regress
`57.7257%/15.3926%`; M08 regresses `153.9024%/39.2018%`.

The M08-local negative-bias point improves M08 `1.4179%/6.5262%`, but the
same setting regresses M10 `119.9936%/39.4416%`. Reject R74 before full-block,
M11/M12, deployment, or 2PB inference. Do not tune the encoder-only MEC raw
adaptation schedule. The next controlled family is training-time halo/context
consistency from the unchanged R57/R58 line.

### R75 Halo-Context Consistency Rejected

R75 started from the exact R57 214-key checkpoint and held the R58 recipe
fixed. A weight-`.05` stopped-teacher term matched affinity probabilities in a
byte-identical central `24x128x128` core after the student's outer `4x16x16`
z/y/x halo was reflect-reconstructed. R72 photometric consistency was off.

SiFlow UUID `1dde395e-27a4-46e8-b5a2-5be9199049ed`
(`mec8nm-r75-halo-consistency-e68c2276`) succeeded in 42m25s. Training reached
12000 steps in 2059 seconds, and the task returned exactly 80 finite M08/M10
WaterZ rows with no failures. The paired analyzer finds zero strict R58 gate
passes. Best balance is bias `(-.20,-.45,-.45)`, temperature `1.00`, threshold
`.50`: mean ARAND/VOI `0.036896625527/0.387556128221`; M08
`0.040070609288/0.377633169630`; M10
`0.033722641766/0.397479086812`. Mean ARAND/VOI regress
`63.6504%/24.5427%`; M08 regresses `168.0624%/47.4393%`.

Reject R75 before full-block, M11/M12, deployment, or 2PB inference. Do not
sweep its halo width, reflection mode, or weight. Future work should not force
context invariance by corrupting the outer training crop.

### R76 Cross-Run Checkpoint Soup Rejected

R76 linearly interpolated all floating tensors from R58 final toward R71 EMA
at alphas `.010,.025,.050,.100,.200`. The audited seed1789 step12000 sources
have identical 214-key topology and equal non-floating state; their relative
L2 distance is only `1.8140%`. Successful SiFlow UUID
`51a8f0f0-3873-4676-9e45-17371f9d1077` completed in 26m01s after two
launcher-only pre-inference failures. Each candidate returned exactly 80
finite M08/M10 WaterZ rows and no failures.

All alphas have zero strict R58 gate passes. Least-regressed alpha `.050` has
mean ARAND/VOI `0.030034391192/0.338540905377`, M08
`0.035741292818/0.342101735986`, and M10
`0.024327489565/0.334980074768`. Mean metrics regress
`33.2138%/8.7914%`; M08 regresses `139.1004%/33.5667%`. Even alpha `.010`
is strongly negative, so do not expand this cross-run soup family. A bounded
R58 same-run step8k/step10k early-stopping screen remains admissible.

### R77 R58 Intermediate Checkpoints Rejected

R77 evaluated the exact R58 step8000 and step10000 checkpoints without
training or interpolation. SiFlow UUID
`756ad839-ab71-463b-84e5-94cba285ee58` succeeded in 11m08s. Both audited
seed1789, non-EMA, 214-key candidates returned exactly 80 finite M08/M10
WaterZ rows and no failures.

Both have zero strict R58 gate passes. Step8000 has mean ARAND/VOI
`0.025127315227/0.308270133998`, M08
`0.043286870594/0.405479642617`, and M10
`0.006967759861/0.211060625378`. Its mean VOI improves `0.9362%`, but mean
ARAND regresses `11.4491%` and M08 regresses `189.5784%/58.3114%`. Step10000
is worse at mean `0.030151783802/0.368098632039`. Do not sweep earlier R58
checkpoints. R58 step12000 remains the crop anchor; R57 remains full-block.

R73--R77 now close the current raw adaptation, synthetic halo consistency,
cross-run soup, and early-stopping families. Require a materially new method
before another GPU wave. Real adjacent overlapping HDF5 windows with shared
augmentation are not equivalent to R75's reflected synthetic halo and remain
a possible future direction.

### R78 Real Paired-Overlap Consistency Rejected

R78 implemented the previously admissible real-context test: same-HDF5 crops
at signed `8x32x32` offsets, one shared geometry/intensity augmentation, exact
coordinate matching over `24x128x128`, and a weight-`.05` stopped-teacher
affinity loss. UUID `cf5a12ba-a390-4f0e-9082-3743dcb5b812` failed before data
or training on a readonly shell assignment. Repaired UUID
`5b34b36a-1b10-416f-a744-76eed2a1874f` succeeded with CUDA 8, WaterZ, 214 R57
keys, the eight allowed train blocks, 12000 steps, exactly 80 finite M08/M10
rows, and no failures.

Best mean ARAND/VOI is `0.021307430975/0.254415142584`, an improvement of
`5.4935%/18.2427%`, but M08 is `0.037685313511/0.346308913501`, a regression
of `152.1054%/35.2093%`. M10 improves to
`0.004929548440/0.162521371668`. Thirteen of 32 settings improve both mean
metrics, but none preserves both M08 metrics, so strict gate passes are zero.

Reject R78 before full-block, M11/M12, deployment, or 2PB inference. Do not
sweep its overlap shift, weight, or teacher direction. R58 remains the crop
anchor and R57 remains the M08/M10 full-block anchor. Generic real-context
invariance is not sufficient; require a materially new training-only mechanism
that explicitly preserves low-boundary/simple-volume topology.

### R79 Topology-Routed Dual-Teacher Distillation Rejected

R79 preserved the R58 student recipe and added only weight-`.10` probability
distillation from frozen R58 and R71-EMA teachers. A soft crop-level route used
raw training-label foreground, boundary, and unique-instance statistics before
widening/augmentation. Sampling and the main supervised objective were
unchanged. A 1024-crop training-only audit found route mean `.110561`, and
both teachers loaded all 214 keys with gradients disabled.

The first local TOS upload failed through the inherited proxy before task
creation. Direct retry of the same tar produced successful SiFlow UUID
`4c50adc6-ee6c-4ccf-bd28-e2341af5e770`
(`mec8nm-r79-topology-teacher-1a465cfa`). It verified CUDA 8, WaterZ, all three
214-key loads, completed 12000 steps in 1755 seconds, and returned exactly 80
finite M08/M10 WaterZ rows with no failures.

Strict gate passes are zero. Best balance is zero bias, temperature `1.00`,
threshold `.50`: mean ARAND/VOI `0.062021381189/0.412646966283`, M08
`0.035591466186/0.337509064870`, and M10
`0.088451296192/0.487784867696`. Mean metrics regress
`175.0881%/32.6057%`; M08 regresses `138.0980%/31.7736%`. R58's exact
temperature-1.15 point is also negative at mean
`0.059614581022/0.441990110322`.

Reject R79 before full-block or frozen splits. Do not sweep route
thresholds/widths, distillation weight, or teachers. R58 remains the crop
anchor and R57 remains full-block. Crop-level R58/R71 teacher routing is now
closed; require a materially different spatial or representation-level
mechanism before another wave.

### R80 Decoder-Feature Instance Embedding Rejected

R80 preserved the exact R58 line and added only a weight-`.02`
discriminative loss on the existing 32-channel final decoder feature map.
Features were L2-normalized and sampled at z/y/x stride `2/4/4`; label zero,
instances with fewer than eight sampled voxels, and instances beyond the 64
largest per crop were excluded. Equal variance/distance terms used margins
`.20/1.0`. No embedding head, parameter, checkpoint key, teacher, extra
affinity, sampling change, or inference operation was added. Six focused
tests, all 37 MEC tests, a CPU CLI smoke run, and a real R57-crop scale audit
passed before submission.

SiFlow UUID `c4ced6cd-7387-4d3f-baf4-7ca2bf3504aa`
(`mec8nm-r80-instance-embedding-9edb4c3d`) succeeded in 28m33s on one 8-GPU
Shanghai `med-model` pod. Startup verified CUDA 8, WaterZ, 214 R57 keys, the
eight permitted training paths, dataset size 32768, and the exact embedding
contract. Training reached step 12000 in 1344 logged seconds with no errors.
The independently fetched result contains exactly 80 finite WaterZ rows, 40
each for M08/M10, and an empty failure list.

There are zero strict R58 gate passes. Best balance is bias
`(-.20,-.45,-.45)`, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.018971582484/0.264526010140`, M08
`0.033522118658/0.371451721207`, and M10
`0.004421046309/0.157600299074`. Mean metrics improve
`15.8539%/14.9935%`, but M08 regresses `124.2546%/45.0258%`. At R58's exact
zero-bias temperature-1.15 point, R80 is worse on both blocks, with mean
`0.062431049101/0.418866062355`.

Reject R80 before full-block or frozen splits. Do not sweep embedding weight,
feature stride, margins, minimum instance size, instance cap, or normalization.
R80 repeats the R71/R78 mechanism pattern: strong M10/aggregate gains do not
preserve M08. R58 remains the crop anchor and R57 remains full-block. A future
wave must use a materially different training-only mechanism with explicit
simple-volume protection, not another teacher blend, generic consistency term,
or global instance-embedding hyperparameter variant. Detailed evidence:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r80_instance_embedding_crop_report_20260714.md`.

### R81 Instance-Pair Boundary-Leak CVaR Rejected

R81 preserved R58 and added a parameter-free cannot-link guard at true
adjacent nonzero instance interfaces. An additional original label tensor was
returned before widening and received the identical augmentation without
advancing the historical RNG stream. For each instance pair, the highest 10%
predicted interface affinities were penalized only above `.50`; interfaces
needed four edges, at most 64 pairs were retained, and the per-pair/per-crop
loss had weight `.05`. Sampling, the ordinary affinity loss, model parameters,
outputs, teachers, EMA, and inference were unchanged.

All 45 MEC tests and a CPU CLI smoke run passed. An eight-block real R57 audit
measured mean raw leak loss about `.067`, contributing about `.0034` after
weighting. SiFlow UUID `76ac0302-57de-4868-abb7-25315cf1c41d`
(`mec8nm-r81-boundary-leak-295cbbc2`) succeeded in 32m15s on one 8-GPU
Shanghai `med-model` pod. It verified CUDA 8, WaterZ, 214 R57 keys, dataset
size 32768, world size 8, and the exact R81 contract. Training reached step
12000 in 1559 logged seconds with no errors. The independent result has exactly
80 finite M08/M10 WaterZ rows and no failures.

Strict R58 gate passes are zero. Best balance is bias
`(-.20,-.45,-.45)`, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.028377941932/0.352231169545`, M08
`0.032977813947/0.375969227906`, and M10
`0.023778069917/0.328493111184`. Mean metrics regress
`25.8668%/13.1908%`; M08 regresses `120.6134%/46.7896%`, while M10 improves
`21.1178%/10.3063%`. M08 remains merge-biased: 202 predictions and VOI merge
`.207795`, versus R58's 242 and `.081985`.

Reject R81 before full-block or frozen splits. Do not sweep leak weight, tail
fraction, maximum affinity, minimum-edge count, or pair cap. Local
voxel-adjacent cannot-link supervision does not control WaterZ region-level
merge ordering. R58 remains the crop anchor and R57 remains full-block. A next
wave must not be another local boundary-weighting/focal/CVaR variant; require a
materially different region/hierarchy-aware training signal or preserve R58
while moving the research delta into independently gated post-processing.
Detailed evidence:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r81_boundary_leak_crop_report_20260714.md`.

### Pretraining and R58 Stem-Micro Audit

Controlled CREMI results support the PublicEM dbMiM objective. With seed 309,
the same anisotropic UNETR and 12k downstream recipe, R45 dbMiM reaches
VOI/ARAND `0.986481/0.186187` versus R47 plain MAE
`1.043065/0.190743`: relative gains of 5.42%/2.39%. The older R17/R23
comparison independently gives a 2.35% VOI gain. This establishes a real
downstream benefit over plain MAE, but there is no strict R58
scratch-versus-PublicEM MEC experiment. On MEC, the initializer-only R70 swap
from PublicEM-R57 to FullEM-R33 regresses crop ARAND/VOI by 137.06%/56.65%.
Keep the PublicEM dbMiM lineage; do not infer that a larger external EM mix is
automatically better.

The separately trained CREMI Stem-Micro scorer was then gated without MEC
training. It uses R58 checkpoint SHA256
`4715c6679808eac5e073159202be1cca77307ab90202bd683f8ab97cacc55e09`,
zero bias, temperature 1.15, and fixed alpha 0.07. Hzzs26 exactly reproduces
the R58 M08/M10 crop WaterZ anchor `ARAND/VOI=0.022546/0.311183`; the CREMI
scorer reaches `0.013019/0.284585` and passes mean plus M08 ARAND/VOI guards.
The SNEMI scorer is rejected because M08 regresses.

Hzzs27 full M08/M10 validation changes the paired same-R58 crop-frozen WaterZ
point from `ARAND/VOI/time=0.418966/1.536809/70.800s` to
`0.091720/0.590987/17.803s`: 78.11% lower ARAND, 61.54% lower VOI, and 3.98x
lower post-processing latency. This is a strong validation/runtime result.

Hzzs28/hzzs29 consume the designated M11/M12 replay once. Against paired
same-R58 WaterZ `0.019254/0.279893/68.807s`, Stem-Micro reaches
`0.016160/0.334732/17.840s`: ARAND improves 16.07% and latency improves 3.86x,
but VOI regresses 19.59%. Both metrics regress on M11 and improve on M12. The
blocks were historically inspected by older dbMiM experiments, so this is not
pristine hidden-test evidence. Do not reopen M11/M12 or promote Stem-Micro as
a uniform quality replacement for WaterZ. The conservative MEC quality line
remains R57 + calibrated WaterZ.

Detailed audit:
`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/evidence/mec8nm_pretraining_stem_micro_audit_20260714.md`.
Machine-readable evidence:
`/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/paper_tmi/data/cross_dataset_hzzs26_20260714/`
through `cross_dataset_hzzs29_20260714/`.


### Hzzs30 Mixed-Domain Post-processing Rejected

Hzzs30 asks whether scorer-side training can improve the frozen R58 setting
without changing affinities or reading MEC08/MEC10/MEC11/MEC12. CREMI A/B,
SNEMI fit z0:70, and MEC01--MEC06 supplied exactly 154,024 RAG edges each to
one dataset-agnostic query scorer; MEC07/MEC09 were excluded from fitting and
used once as internal full-volume holdouts. SiFlow UUID
`1ceae75a-676a-48b9-879b-730791bbac1b` succeeded.

The frozen CREMI scorer has mean MEC07/MEC09 VOI/ARAND/time
`0.228785/0.004687/20.870s`; the mixed scorer regresses to
`1.123105/0.232610/18.634s`. It fails VOI and ARAND separately on both
blocks. Mean split VOI rises from `0.203322` to `1.107652`, despite lower
merge VOI, so the pooled supervision damages merge ordering under the fixed
alpha-0.07 structural stop. Checkpoint SHA256
`882f0ef2140950121b9857628b728aa17f87948e6b13087de972eb31d59c015e` is
rejected. Do not sweep its domain weights, seeds, or architecture on these
holdouts.

This does not change the affinity-pretraining conclusion: retain the PublicEM
dbMiM lineage and R58 crop anchor. It also does not overturn the useful frozen
CREMI scorer, which is 3.47x faster than WaterZ and has 47.37% lower mean VOI
and 95.08% lower mean ARAND on MEC07/MEC09, although WaterZ is better on M09
alone. No post-processor is uniformly best across blocks, and the conservative
full MEC quality line remains R57 + calibrated WaterZ.

Detailed evidence:
`/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/docs/dbmim_transfer/locagg_mixed_domain_hzzs30_plan_20260714.md`.


### Hzzs31 Local-Competition Pooled Scorer Rejected

Hzzs31 was the single predeclared representation follow-up to hzzs30. It kept
the same affinities, data, equal domain counts, MSE objective, seed, model,
alpha, and MEC07/MEC09 gate, changing only the RAG contract from raw features
to `raw_plus_local_competition`. SiFlow UUID
`efde2b93-f8ce-4712-bc2a-0a982705492f` succeeded. The 41-dimensional edge /
42-dimensional node checkpoint loaded correctly and its training MSE/accuracy
improved slightly to `0.022026/0.90133`.

Instance quality still failed all four block guards. MEC07 is
`VOI/ARAND/time=1.788033/0.438546/19.719s`; MEC09 is
`0.711380/0.088428/18.962s`. Mean split VOI is `1.234268`, so local
competition worsens hzzs30's under-merging rather than repairing cross-domain
ordering. Reject checkpoint SHA256
`9de49758a8ac476e9d359cf4cbe6627ffca76f84e9436b258ee354a4f811fdcd`.
The hzzs30/hzzs31 pooled-scorer branch is closed: do not sweep mixtures, seeds,
architectures, or feature combinations, and do not reuse MEC07/MEC09 for
another variant. MEC08/MEC10/MEC11/MEC12 were not read.

Detailed evidence:
`/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/docs/dbmim_transfer/locagg_mixed_localcomp_hzzs31_plan_20260714.md`.


## 2026-07-15 Authoritative MEC and Scale Snapshot

This section supersedes older status language such as "running", "pending",
or "untouched" elsewhere in this historical ledger. All reported LoCoAgg jobs
through hzzs33 are complete. No relevant MEC/LoCoAgg SiFlow task is active.

### What Is Currently Best

There is no single setting that is best under every MEC protocol:

- **Affinity selection on the frozen crop gate:** R58 remains the selected MEC
  affinity model. Its M08/M10 crop mean is
  `ARAND/VOI=0.022546003667/0.311183450751`, with M08
  `0.014948239287/0.256127966754` and M10
  `0.030143768046/0.366238934749`. Zero bias, temperature `1.15`, and WaterZ
  threshold `.50` are frozen. No R59--R81 candidate passes mean ARAND, mean
  VOI, M08 ARAND, and M08 VOI simultaneously.
- **Conservative absolute full-block quality:** R57 plus calibrated WaterZ is
  still the deployment anchor, with official M08/M10 full-block mean
  `ARAND/VOI=0.252279795612/1.062235821039`. This metric scale is not directly
  comparable to the crop gate.
- **Strongest same-affinity validation/runtime research result:** R58 plus the
  frozen CREMI LoCoAgg scorer reaches full M08/M10
  `ARAND/VOI/time=0.091720333957/0.590987031259/17.803s`, versus the paired
  crop-frozen R58 WaterZ point
  `0.418965808348/1.536809043919/70.800s`. This is a 3.98x post-processing
  speedup and a strong validation result, not a universal production default.
- **Generalization replay:** on the one designated M11/M12 replay, WaterZ is
  `0.019254036555/0.279892831006/68.807s` and LoCoAgg is
  `0.016160111808/0.334732313606/17.840s`. ARAND and latency improve, VOI
  regresses, and the two blocks move in opposite directions. Do not reopen
  M11/M12 or use them for another choice.

M01--M07/M09 supplied supervised training. M08/M10 supplied repeated protocol
validation. M11/M12 were consumed by the designated replay and also have older
historical inspection. M07/M09 were later consumed as hzzs30--hzzs32 internal
post-processing diagnostics. Therefore no untouched MEC selection split
remains. A fresh performance claim requires new labeled MEC blocks or an
official hidden evaluation; more tuning on existing blocks is leakage.

### Model, Training, and Pretraining Conclusion

The upstream model is `unetr_aniso_em`: crop `32x160x160`, patch
`4x16x16`, transformer skips, staged anisotropic decoder, and three nearest-
neighbor affinity channels in z/y/x. The stable supervised recipe is MSE plus
MAWS with `membrane_weight=0.75`, `encoder_lr=1e-5`, base LR `8e-5`, weight
decay `.01`, AMP, and batch size 2 per GPU. MEC used a 12k-step schedule from
the R57/PublicEM lineage.

PublicEM dbMiM pretraining is useful in the controlled CREMI comparison: R45
is `VOI/ARAND=0.986481/0.186187` versus same-seed plain MAE R47
`1.043065/0.190743`, gains of 5.42% VOI and 2.39% ARAND. R48 at 20k steps is
`0.962154/0.178252`. There is no strict MEC scratch-versus-PublicEM ablation,
so do not invent a pure MEC pretraining gain. More external EM is not
automatically better: FullEM initializer R70 regresses mean crop ARAND/VOI to
`0.053447533042/0.487475984541`, 137.06%/56.65% worse than R58. Retain the
PublicEM lineage and reject the FullEM swap.

### Hzzs32 Signed-VI Mechanism Result

Hzzs30 pooled raw features and hzzs31 pooled local-competition features both
failed from severe under-merging. Hzzs32 changed the objective/stopping
semantics instead of sweeping the mixture. It uses the exact per-union-voxel
signed local VI change

`Delta_VI = 2 H(B|G) - H(B)`

and maps it to a scorer target whose probability `0.5` is zero merge utility.
It removes the alpha fragment-count stop. UUID
`18a51652-3d17-4caf-9543-b04d068c2514` succeeded. MEC07 is
`VOI/ARAND/time=0.116861/0.001107/18.676s`; MEC09 is
`0.170761/0.004478/19.437s`; the mean is
`0.143811/0.002793/19.057s`. Total VOI and ARAND improve over the frozen CREMI
scorer on both consumed development blocks, and mean split VOI falls from
hzzs31 `1.234268` to `0.111158`.

The strict component gate still fails: MEC07 merge VOI worsens from frozen
`0.018592` to `0.034325`. Hzzs32 is evidence that signed stopping semantics
repair under-merging, not a promotable checkpoint. MEC07/MEC09 cannot be reused
for selection.

### Hzzs33 AxonEM Transfer Failure

Hzzs33 trains one routed-free signed-VI scorer with separately balanced CREMI,
SNEMI, MEC, AxonEM-H, and AxonEM-M edges. UUID
`972a40d0-4a6c-4ec6-a9da-be802235a2e6` succeeded. Overall AxonEM is
`VOI/ARAND/time=0.631755/0.147431/1.566s`. Split VOI improves from frozen
LoCoAgg `0.4195` to `0.2083`, but merge VOI worsens from `0.1112` to `0.4235`.
WaterZ is `0.391034/0.052629/1.494s`. False merges dominate and both
predeclared gates fail.

This localizes the remaining method problem: stopping semantics are now
meaningful, but merge ordering is not acquisition-stable after components
grow. Do not tune alpha, probability threshold, `min_boundary`, domain weights,
seed, or another static pooled scorer. The next algorithmic hypothesis is
component-rescored signed utility: maintain component and boundary sufficient
statistics, re-score only the dirty frontier after each union, train with
endpoint-local listwise regret plus a group-worst loss across fit domains, and
retain `0.5` as the signed-VI zero point. Confirmation requires new untouched
labels or an official hidden evaluation.

### PB-Scale Readiness

The current answer is **not production ready for PB-scale globally coherent
instance segmentation**. Blockwise affinity inference and local segmentation
can run today, and full MEC blocks around `125x1250x1250` have been evaluated,
but the complete pipeline has not demonstrated volume-independent host memory,
global ID consistency, cross-block quality, resumability, failure recovery, or
PB storage/throughput cost.

The promoted LoCoAgg v4 executor uses fixed `16x512x512` RAG tiles and exact
global sufficient-statistic reduction. On full SNEMI it takes about `2.276s`,
allocates about 440 MiB GPU memory, and reaches about 4.48 GiB process RSS.
Experimental v5 adds exact 32-way hash spill and a bounded NPY fragment writer.
It reduces the largest partial-record workset by 19--26x and RSS to about
3.65--3.68 GiB, but total time rises to `4.01--4.08s` because the CPU fragment
stage alone costs `2.05--2.16s`. Dense affinities/fragments remain resident, so
neither path yet proves memory independent of total volume size.

Global WaterZ or one dense global RAG is infeasible at PB scale. The required
architecture is:

1. Chunked Zarr/TOS raw input and slab-wise calibrated affinity output with
   deterministic chunk IDs, halos, checksums, and append-only manifests.
2. GPU fragment generation without retaining full logits plus affinities.
3. Streaming tiled sufficient statistics into append-only hash partitions,
   followed by external reduction and a sparse cross-block RAG.
4. Deterministic overlap reconciliation, stable final `uint64` IDs, orphan and
   duplicate-ID checks, and exact fallback for ambiguous seams.
5. Independently retryable chunks, idempotent outputs, failure recovery, and
   measured bytes/voxel plus GPU-hours/TB.

Scale only through predeclared gates: single-block parity; a `2x2x2` seam and
stable-ID test; `1--10 TB` resumability/I/O/throughput; `100 TB` recovery and
cost; then PB. Report seam VOI/ARAND, split/merge VI, orphan/duplicate IDs,
bytes/voxel, GPU-hours/TB, and peak RAM as volume grows. Because hzzs33 exposes
catastrophic external-domain false merges, use conservative local WaterZ/MWS
plus deterministic reconciliation as the near-term large-volume baseline.
Do not deploy the current learned LoCoAgg checkpoint globally on PB data.

### Next-Agent Start Here

1. Read this snapshot and the LoCoAgg state at
   `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/stem-micro-postprocess-skill/STATE.md`.
2. Preserve R58 as the crop affinity anchor and R57 plus calibrated WaterZ as
   the conservative full-block anchor. Do not reopen R59--R81, M07/M09,
   M08/M10, or M11/M12 for selection.
3. If new untouched labels exist, test component-rescored signed utility once
   with a predeclared group-worst gate. Otherwise prioritize the systems path:
   GPU fragments, calibrated affinity slabs, streaming partitions, and the
   `2x2x2` cross-block seam harness.
4. Do not submit a 2PB/PB task until the single-block, multi-block, TB, and
   100-TB gates are met. No relevant MEC/LoCoAgg job is currently running.

### Repository Integration Contract

The versioned project remains `https://github.com/ydchen0806/dbMiM.git` on the
`main` branch. Preserve its original root layout. The new `mec8nm/` directory
is a self-contained MEC workspace, while root-level runtime files remain for
historical commands and SiFlow bundles. Shared source files must be
byte-identical and are enforced by `mec8nm/scripts/verify_workspace.py`.

`dbmim-neuron-seg-skill/` is the canonical repository skill source;
`mec8nm/skill/` is its portable workspace mirror and
`$CODEX_HOME/skills/dbmim-neuron-seg/` is the installed discovery mirror. Keep
all three copies synchronized and validate the two repository copies before a
merge. Do not commit datasets, checkpoints, generated outputs, credentials,
wheelhouses, or caches. Merge only through a tested integration branch based
on current `origin/main`; do not rewrite existing dbMiM history.
