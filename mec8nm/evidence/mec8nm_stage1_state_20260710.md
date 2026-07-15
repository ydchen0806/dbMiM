# MEC 8nm Stage1 State

Updated: 2026-07-14T12:55Z

## Data

- Source HF dataset: `haoyuan7/mec`
- Local snapshot: `/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/hf_snapshot`
- Local HDF5: `/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/hdf5`
- Manifest: `/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/mec8nm_manifest.json`
- Splits: `/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/mec8nm_splits.json`
- TOS mirror: `tos://agi-data/users/dchen02/dbmim/assets/mec8nm_20260711/`

The dataset has 12 paired raw/myelin-label volumes. HDF5 conversion writes
datasets `raw` and `labels`. Raw volumes were safely stored as `uint8`; label
volumes preserve original integer dtype. Sample 6 had raw/label shape mismatch
and was cropped to the common shape `(125, 1250, 1250)`.

Block-level split:

- train: `1,2,3,4,5,6,7,9`
- val: `8,10`
- test: `11,12`

## Code Added

- `/volume/med-train/users/dchen02/code/dbMiM/scripts/prepare_mec8nm_hdf5.py`
- `/volume/med-train/users/dchen02/code/dbMiM/configs/finetune_mec8nm_r48_seed309_stage1.yaml`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_stage1_siflow.sh`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_blockwise_eval_siflow.sh`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/infer_mec8nm_blockwise_rag.py`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_infer_rag_siflow.sh`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/stitch_mec8nm_slab_segmentations.py`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/infer_mec8nm_slabwise_stitch.py`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_slabwise_stitch_siflow.sh`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/restitch_mec8nm_saved_slabs.py`

## Stage1 Training

- Init checkpoint: local R48 publicEM dbMiM finetuned checkpoint
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/hf_release/dbmim-neuron-segmentation/weights/publicem_dbmim_r48_seed309_long20k/finetuned_latest.pt`
- Recipe: anisotropic EM UNETR affinity model, `32x160x160`, MSE+MAWS,
  `max_steps=5000`, 8-GPU torchrun.
- Bundle root:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/siflow_bundles/mec8nm_stage1_20260710T183017Z`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_stage1/mec8nm-stage1-r48.20260711T023152+0800.tar.gz`
- SiFlow task UUID:
  `2e011931-44df-4715-bbd1-233a1507d890`
- Region/cluster/pool:
  `cn-shanghai/changliu/med-model`, instance `sci.g21-3`, 8 GPUs.
- Output TOS prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r48_seed309_stage1/`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_stage1_20260710/finetune_mec8nm_r48_seed309_stage1`

The pod entrypoint downloads MEC HDF5 from TOS, trains, evaluates val/test
center crops (`64x512x512`) with graph-CC and waterz threshold sweeps, then
uploads train/eval/log outputs to the expected TOS prefix.

Status: completed successfully on SiFlow. Training reached step 5000. TOS
upload reported failed count 0.

Final checkpoint:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_stage1_20260710/finetune_mec8nm_r48_seed309_stage1/train/finetune_mec8nm_r48_seed309_stage1/finetuned_latest.pt`

- size: `217581330` bytes
- sha256: `d5073ce51e254aba306ecc781a7929847fb985a55ebba7c80cad89242e0bdc3b`

Training log:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_stage1_20260710/finetune_mec8nm_r48_seed309_stage1/train/finetune_mec8nm_r48_seed309_stage1/finetune_log.jsonl`

- 250 logged rows, final row step 5000.
- final logged loss: `0.06062418222427368`.

## Stage1 Crop Evaluation

Evaluation used center crops of shape `64x512x512`, metric backend `skimage`,
ignore label `0`, XY boundary ignore distance `1`, Z boundary ignore distance
`0`. The pod had `cc3d`, `cupy`, `mahotas`, and `scipy`; `waterz` was not
installed. Therefore all waterz rows in `failures` are
`ModuleNotFoundError("No module named 'waterz'")`; graph-CC rows completed.

Validation summary:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_stage1_20260710/finetune_mec8nm_r48_seed309_stage1/eval_val_crop64x512/eval_finetune_mec8nm_r48_seed309_stage1_val_crop64x512/cremi_segmentation_summary.json`

- aggregate best by ARAND: ARAND `0.23733932155967724`, VOI-sum
  `1.154644829813929`, threshold `0.5`, variant
  `bias_z+0.00_y+0.00_x+0.00_t1.00`.
- aggregate best by VOI-sum: ARAND `0.23766589794937065`, VOI-sum
  `1.1198299874822157`, threshold `0.16`, variant
  `bias_z+0.00_y+0.00_x+0.00_t1.15`.
- per-sample best ARAND:
  - `mec8nm_08.h5`: ARAND `0.3803503239825222`, VOI-sum
    `1.7644568713675848`, threshold `0.22`, variant `pred`.
  - `mec8nm_10.h5`: ARAND `0.09383105413478021`, VOI-sum
    `0.5192302971202736`, threshold `0.5`, variant
    `bias_z-0.20_y-0.45_x-0.45_t1.00`.

Test summary:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_stage1_20260710/finetune_mec8nm_r48_seed309_stage1/eval_test_crop64x512/eval_finetune_mec8nm_r48_seed309_stage1_test_crop64x512/cremi_segmentation_summary.json`

- aggregate best by ARAND/VOI-sum: ARAND `0.5972012005624793`, VOI-sum
  `2.8682100645188795`, threshold `0.5`, variant
  `bias_z-0.20_y-0.45_x-0.45_t1.00`.
- per-sample best ARAND:
  - `mec8nm_11.h5`: ARAND `0.540691632971493`, VOI-sum
    `2.681576637584919`, threshold `0.5`, variant
    `bias_z-0.20_y-0.45_x-0.45_t1.00`.
  - `mec8nm_12.h5`: ARAND `0.6537107681534655`, VOI-sum
    `3.0548434914528406`, threshold `0.5`, variant
    `bias_z-0.20_y-0.45_x-0.45_t1.00`.

Interpretation: Stage1 proves the MEC data path, R48 initialization, 8-GPU
training, evaluation, and TOS archival path. It is not deployment-ready: test
ARAND is high and recall is low despite strong affinity Dice. The next useful
experiments should improve boundary-aware sampling/postprocessing, not just
extend the same recipe unchanged.

## Stage2 Boundary-Sampling Run

Code/config changes:

- Added optional label-guided boundary crop selection in
  `/volume/med-train/users/dchen02/code/dbMiM/dbmim/datasets.py`.
- Wired the config fields through
  `/volume/med-train/users/dchen02/code/dbMiM/train_finetune.py`.
- Added config:
  `/volume/med-train/users/dchen02/code/dbMiM/configs/finetune_mec8nm_r48_seed309_boundary_stage2.yaml`.
- Parameterized SiFlow entrypoint:
  `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_stage1_siflow.sh`.

Boundary sampling smoke showed the pathological index-7 random crop
`fg=1.0`, `boundary=0.0`, `unique_nonzero=1` was replaced by a crop with
`fg=0.835859375`, `boundary=0.275927734375`, `unique_nonzero=32`.

SiFlow task:

- UUID: `254e57d4-c501-4ad9-b4bf-6072ac354afb`
- Name: `mec8nm-stage2-boundary-bff2de98`
- Bundle root:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/siflow_bundles/mec8nm_boundary_stage2_20260710T190253Z`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_stage2_boundary/mec8nm-stage2-boundary.20260711T030713+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r48_seed309_boundary_stage2/`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_stage2_boundary_20260710/finetune_mec8nm_r48_seed309_boundary_stage2`

Stage2 trained 6000 steps from the stage1 checkpoint with boundary sampling,
small Dice/boundary-Dice terms, and small BCAR terms. It evaluated graph-CC only
to avoid known missing-waterz failures. TOS upload failed count was 0.

Final checkpoint:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_stage2_boundary_20260710/finetune_mec8nm_r48_seed309_boundary_stage2/train/finetune_mec8nm_r48_seed309_boundary_stage2/finetuned_latest.pt`

- size: `217581778` bytes
- sha256: `f5e5632dc3df9eae353831447c13f22c32a0a665bb25aa2466d4c828b20651d4`

Validation summary:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_stage2_boundary_20260710/finetune_mec8nm_r48_seed309_boundary_stage2/eval_val_crop64x512/eval_finetune_mec8nm_r48_seed309_boundary_stage2_val_crop64x512/cremi_segmentation_summary.json`

- aggregate best ARAND/VOI-sum: ARAND `0.23822566826315478`, VOI-sum
  `1.154767437594733`, threshold `0.16`, variant
  `bias_z+0.00_y+0.00_x+0.00_t1.15`.
- per-sample best ARAND:
  - `mec8nm_08.h5`: ARAND `0.38193341657978386`, VOI-sum
    `1.8141275603954503`, threshold `0.16`, variant
    `bias_z+0.00_y+0.00_x+0.00_t1.15`.
  - `mec8nm_10.h5`: ARAND `0.09358338411843681`, VOI-sum
    `0.6580956858043283`, threshold `0.5`, variant
    `bias_z-0.20_y-0.45_x-0.45_t1.00`.

Test summary:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_stage2_boundary_20260710/finetune_mec8nm_r48_seed309_boundary_stage2/eval_test_crop64x512/eval_finetune_mec8nm_r48_seed309_boundary_stage2_test_crop64x512/cremi_segmentation_summary.json`

- aggregate best ARAND/VOI-sum: ARAND `0.582039307504637`, VOI-sum
  `2.7947551534279684`, threshold `0.5`, variant
  `bias_z-0.20_y-0.45_x-0.45_t1.00`.
- per-sample best ARAND:
  - `mec8nm_11.h5`: ARAND `0.5190202232968468`, VOI-sum
    `2.4802935066110687`, threshold `0.5`, variant
    `bias_z-0.20_y-0.45_x-0.45_t1.00`.
  - `mec8nm_12.h5`: ARAND `0.6450583917124271`, VOI-sum
    `3.1092168002448686`, threshold `0.5`, variant
    `bias_z-0.20_y-0.45_x-0.45_t1.00`.

Interpretation: Stage2 improves aggregate test ARAND/VOI modestly versus
stage1 (`0.5820/2.7948` vs `0.5972/2.8682`), mostly from `mec8nm_11`.
Validation is slightly worse and `n_pred` rises sharply on `mec8nm_10` and
`mec8nm_12`, so the recipe increases over-splitting. Keep stage2 as a useful
ablation, not a deployment checkpoint.

## R57 Initialization Run

Config:

`/volume/med-train/users/dchen02/code/dbMiM/configs/finetune_mec8nm_r57_seed777_stage1.yaml`

This uses the same recipe as R48 stage1, but initializes from:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/hf_release/dbmim-neuron-segmentation/weights/publicem_dbmim_r57_seed777_long20k/finetuned_latest.pt`

SiFlow task:

- UUID: `8f50cd3b-03cc-46c4-9b87-e9591a006c31`
- Name: `mec8nm-stage1-r57-bfe9dbc3`
- Bundle root:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/siflow_bundles/mec8nm_r57_stage1_20260710T193336Z`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_stage1/mec8nm-stage1-r57.20260711T033419+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r57_seed777_stage1/`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_stage1_20260710/finetune_mec8nm_r57_seed777_stage1`

R57 trained 5000 steps, evaluated graph-CC only, and uploaded with failed count
0.

Final checkpoint:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_stage1_20260710/finetune_mec8nm_r57_seed777_stage1/train/finetune_mec8nm_r57_seed777_stage1/finetuned_latest.pt`

- size: `217581330` bytes
- sha256: `84a0c009e908085751315abd0d340eb7a77dfa27c9dded5d7d1a6d812efe8d11`

Validation summary:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_stage1_20260710/finetune_mec8nm_r57_seed777_stage1/eval_val_crop64x512/eval_finetune_mec8nm_r57_seed777_stage1_val_crop64x512/cremi_segmentation_summary.json`

- aggregate best by ARAND: ARAND `0.23736922159505552`, VOI-sum
  `1.1416628243575004`, threshold `0.5`, variant
  `bias_z+0.00_y+0.00_x+0.00_t1.00`.
- aggregate best by VOI-sum: ARAND `0.23768478684443545`, VOI-sum
  `1.119068270262251`, threshold `0.16`, variant
  `bias_z+0.00_y+0.00_x+0.00_t1.15`.
- per-sample best ARAND:
  - `mec8nm_08.h5`: ARAND `0.3803507866019864`, VOI-sum
    `1.7619214357159723`, threshold `0.24`, variant
    `bias_z+0.00_y+0.00_x+0.00_t1.15`.
  - `mec8nm_10.h5`: ARAND `0.09408875428979857`, VOI-sum
    `0.5021484075935384`, threshold `0.5`, variant
    `bias_z-0.20_y-0.45_x-0.45_t1.00`.

Test summary:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_stage1_20260710/finetune_mec8nm_r57_seed777_stage1/eval_test_crop64x512/eval_finetune_mec8nm_r57_seed777_stage1_test_crop64x512/cremi_segmentation_summary.json`

- aggregate best ARAND/VOI-sum: ARAND `0.577023601026518`, VOI-sum
  `2.6926708904983006`, threshold `0.5`, variant
  `bias_z-0.20_y-0.45_x-0.45_t1.00`.
- per-sample best ARAND:
  - `mec8nm_11.h5`: ARAND `0.5001902991337606`, VOI-sum
    `2.3253084931011556`, threshold `0.5`, variant
    `bias_z-0.20_y-0.45_x-0.45_t1.00`.
  - `mec8nm_12.h5`: ARAND `0.6538569029192753`, VOI-sum
    `3.0600332878954455`, threshold `0.5`, variant
    `bias_z-0.20_y-0.45_x-0.45_t1.00`.

Interpretation: R57 is the best current center-crop checkpoint by aggregate
test metrics (`ARAND=0.5770`, `VOI=2.6927`). It improves strongly on
`mec8nm_11`, is roughly tied/slightly worse on `mec8nm_12`, and has stable
postprocess settings matching the other runs. It is the next checkpoint to
evaluate on larger crops/full blocks, but it is still not deployment-ready.

## R57 Full-Block Evaluation

Eval script/config changes:

- Added sample selection, calibrated-logit variants, skip-raw-pred mode, and
  per-sample partial output files to
  `/volume/med-train/users/dchen02/code/dbMiM/scripts/evaluate_cremi_blockwise_scale.py`.
- Added batched `waterz.agglomerate(thresholds=[...])` execution for waterz
  threshold sweeps, so one affinity variant can evaluate multiple thresholds
  without repeatedly rebuilding the hierarchy.
- Added eval-only SiFlow launcher:
  `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_blockwise_eval_siflow.sh`.
- Updated the SiFlow eval launcher to install `waterz` from bundled
  `third_party/waterz` with bundled Boost headers when the image does not
  already provide the module.

Single-threshold full-block graph-CC eval:

- SiFlow UUID: `7dbbc61a-4258-4ea4-b799-dedde3103feb`
- Name: `mec8nm-r57-fullblock-eval-640ed361`
- Bundle root:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/siflow_bundles/mec8nm_r57_fullblock_eval_20260710T201041Z`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-fullblock-eval.20260711T041219+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r57_seed777_fullblock_test_graphcc`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_fullblock_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_graphcc`
- Checkpoint asset evaluated:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_stage1_20260710/finetune_mec8nm_r57_seed777_stage1/train/finetune_mec8nm_r57_seed777_stage1/finetuned_latest.pt`
  with sha256 `84a0c009e908085751315abd0d340eb7a77dfa27c9dded5d7d1a6d812efe8d11`.
- Eval setting: test blocks `mec8nm_11.h5` and `mec8nm_12.h5`, full
  `125x1250x1250` blocks, chunk size `32x512x512`, halo `8x64x64`,
  stride `16x80x80`, graph-CC only, calibrated variant
  `bias_z-0.20_y-0.45_x-0.45_t1.00`, z/xy threshold `0.5`.

Summary file:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_fullblock_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_graphcc/blockwise_eval/eval_mec8nm_r57_seed777_fullblock_test_graphcc/blockwise_scale_summary.json`

Aggregate full-region metrics:

- ARAND `0.536017605249852`
- VOI-sum `2.9724227250052198`
- VOI-merge `2.9138148195166558`
- VOI-split `0.058607905488563664`
- seam ARAND/VOI-sum `0.5291133475790308` / `2.8540305192331976`
- nonseam ARAND/VOI-sum `0.5371350410538784` / `2.981639484324427`

Per-sample full-region metrics:

- `mec8nm_11.h5`: ARAND `0.6078519596737333`, VOI-sum
  `3.6565956093970673`, precision `0.9969681808319708`, recall
  `0.2440766443186483`, `n_pred=461737`, `n_gt=635`.
- `mec8nm_12.h5`: ARAND `0.46418325082597056`, VOI-sum
  `2.288249840613372`, precision `0.9995411260816112`, recall
  `0.3660107313052353`, `n_pred=288615`, `n_gt=197`.

Throughput:

- `mec8nm_11.h5`: inference `63.3035s`, `3.085M voxels/s`, 36 chunks.
- `mec8nm_12.h5`: inference `62.4669s`, `3.127M voxels/s`, 36 chunks.
- Whole eval elapsed `332s`; failures `[]`.

Interpretation: full-block inference is fast enough for a first deployment
prototype, and seam/nonseam metrics are close, so the immediate bottleneck is
not chunk seams. The segmentation is still not deployment-ready: graph-CC at
threshold `0.5` has very high precision but low recall and strong merge VOI.
The next cheapest step is a full-block z/xy threshold sweep for the same R57
checkpoint before spending more GPU time on retraining.

Full-region z/xy threshold sweep:

- SiFlow UUID: `97b21148-d971-49ad-9f60-672744979011`
- Name: `mec8nm-r57-threshgrid-808b3fb4`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-threshgrid.20260711T042257+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r57_seed777_fullblock_test_graphcc_threshgrid`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_threshgrid_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_graphcc_threshgrid`
- Eval setting: same R57 checkpoint and calibrated variant as above, full
  regions only, z thresholds `0.16,0.24,0.32,0.40,0.50`, xy thresholds
  `0.16,0.24,0.32,0.40,0.50`.
- Runtime status: completed successfully; elapsed `2529s`; 50 metric rows;
  failures `[]`.

Summary file:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_threshgrid_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_graphcc_threshgrid/blockwise_eval/eval_mec8nm_r57_seed777_fullblock_test_graphcc_threshgrid/blockwise_scale_summary.json`

Threshold sweep result:

- Aggregate best by both ARAND and VOI-sum is still `z_threshold=0.5`,
  `xy_threshold=0.5`: ARAND `0.536017605249852`, VOI-sum
  `2.9724227250052198`.
- `mec8nm_11.h5` best by both ARAND and VOI-sum is `z=0.5`, `xy=0.5`:
  ARAND `0.6078519596737333`, VOI-sum `3.6565956093970673`, precision
  `0.9969681808319708`, recall `0.2440766443186483`, `n_pred=461737`.
- `mec8nm_12.h5` best by both ARAND and VOI-sum is `z=0.5`, `xy=0.5`:
  ARAND `0.46418325082597056`, VOI-sum `2.288249840613372`, precision
  `0.9995411260816112`, recall `0.3660107313052353`, `n_pred=288615`.
- Top aggregate alternatives are worse:
  - `z=0.40`, `xy=0.50`: ARAND `0.5440920145680666`, VOI-sum
    `3.098118183071521`.
  - `z=0.50`, `xy=0.40`: ARAND `0.5449017045838844`, VOI-sum
    `3.1067532037520706`.
  - `z=0.40`, `xy=0.40`: ARAND `0.5465485729923353`, VOI-sum
    `3.1368228596862924`.

Interpretation: simple graph-CC threshold tuning does not rescue R57. Lower
thresholds reduce split VOI but worsen merge VOI and aggregate ARAND/VOI. The
best available graph-CC setting in this grid remains the original conservative
`0.5/0.5` setting, which is still not deployment-ready. The next step must
change the postprocess family or the training signal, not just retune
graph-CC thresholds.

## R57 RAG Agglomeration Evaluation

Crop-level mahotas RAG screening:

- SiFlow UUID: `758445c8-bf86-49e1-97d0-7d122aa5d023`
- Name: `mec8nm-r57-rag-crop-8ac5fd9a`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-rag-crop.20260711T051101+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r57_seed777_crop64_test_mahotas_agglomeration`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_rag_crop_eval_20260710/eval_mec8nm_r57_seed777_crop64_test_mahotas_agglomeration`
- Eval setting: test blocks `mec8nm_11.h5` and `mec8nm_12.h5`, center
  `64x512x512` crops, mahotas watershed fragments plus affinity RAG
  agglomeration, score modes `mean,q25`, z/xy thresholds
  `0.16,0.24,0.32,0.40,0.50`, calibrated R57 affinities.
- Runtime status: completed successfully; elapsed `949s`; 100 metric rows;
  failures `[]`.

Crop screening result:

- Aggregate best is `score_mode=q25`, `z=0.5`, `xy=0.5`: ARAND
  `0.3605692475694753`, VOI-sum `1.7507532590683046`, VOI-merge
  `1.4910674581735273`, VOI-split `0.2596858008947772`.
- `mec8nm_11.h5`: ARAND `0.10921335985992642`, VOI-sum
  `0.9538612865478153`, precision `0.9720127440637686`, recall
  `0.8220889004544235`, `n_pred=18625`, `n_gt=99`.
- `mec8nm_12.h5`: ARAND `0.6119251352790241`, VOI-sum
  `2.5476452315887936`, precision `0.9971251474633972`, recall
  `0.24091964018037257`, `n_pred=16241`, `n_gt=59`.

Interpretation: RAG agglomeration is much better than graph-CC on the crop
screen, especially for `mec8nm_11`. `mec8nm_12` improves modestly versus the
R57 graph-CC crop baseline (`0.6539/3.0600` to `0.6119/2.5476`). This justifies
full-block RAG validation despite higher postprocess cost.

Full-block mahotas RAG validation:

- SiFlow UUID: `db30543d-44f3-4415-b8fb-b0bf9cc419d2`
- Name: `mec8nm-r57-rag-full-dfccc2d6`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-rag-full.20260711T053134+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r57_seed777_fullblock_test_mahotas_agglomeration_q25`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_rag_full_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_mahotas_agglomeration_q25`
- Eval setting: full `125x1250x1250` test blocks, chunk size
  `32x512x512`, halo `8x64x64`, calibrated R57 affinities,
  `mahotas_agglomeration`, `score_mode=q25`, z/xy threshold `0.5`, regions
  `full,seam,nonseam`.
- Runtime status: completed successfully; elapsed `617s`; 6 metric rows;
  failures `[]`.

Summary file:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_rag_full_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_mahotas_agglomeration_q25/blockwise_eval/eval_mec8nm_r57_seed777_fullblock_test_mahotas_agglomeration_q25/blockwise_scale_summary.json`

Full-block RAG metrics:

- Aggregate full-region ARAND `0.47802323170358296`, VOI-sum
  `2.2950284376566454`, VOI-merge `2.1214047768226845`, VOI-split
  `0.17362366083396077`.
- Aggregate seam ARAND/VOI-sum `0.4696152456495122` /
  `2.178971783912311`.
- Aggregate nonseam ARAND/VOI-sum `0.4794701215543917` /
  `2.3012719150379892`.
- `mec8nm_11.h5` full: ARAND `0.5082483202083637`, VOI-sum
  `2.492994975190426`, precision `0.9870399501222566`, recall
  `0.32744337373930654`, `n_pred=202897`, `n_gt=635`, postprocess
  `214.624s`.
- `mec8nm_12.h5` full: ARAND `0.4477981431988022`, VOI-sum
  `2.0970619001228643`, precision `0.999631793112464`, recall
  `0.38146163432436175`, `n_pred=76804`, `n_gt=197`, postprocess
  `124.537s`.
- Inference throughput remains about `3.0M voxels/s`: `65.58s` for
  `mec8nm_11.h5`, `63.75s` for `mec8nm_12.h5`.

Comparison to full-block graph-CC:

- Graph-CC aggregate full: ARAND `0.536017605249852`, VOI-sum
  `2.9724227250052198`.
- RAG aggregate full: ARAND `0.47802323170358296`, VOI-sum
  `2.2950284376566454`.
- RAG improves `mec8nm_11.h5` full ARAND/VOI from
  `0.6078519596737333` / `3.6565956093970673` to
  `0.5082483202083637` / `2.492994975190426`.
- RAG improves `mec8nm_12.h5` full ARAND/VOI from
  `0.46418325082597056` / `2.288249840613372` to
  `0.4477981431988022` / `2.0970619001228643`.

Interpretation: R57 + mahotas RAG q25 is the current best full-block candidate
and is worth engineering further. It is still not the final 2PB deployment
checkpoint because full-block ARAND remains high and same-instance recall is
only `0.327` on `mec8nm_11` and `0.381` on `mec8nm_12`. However, the RAG
postprocess changes the situation materially: graph-CC is confirmed as a
postprocess bottleneck, not just a model bottleneck.

## R57 RAG Inference Export

Deployment-style export code:

- `/volume/med-train/users/dchen02/code/dbMiM/scripts/infer_mec8nm_blockwise_rag.py`
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_infer_rag_siflow.sh`

The export script reuses the same chunked affinity prediction and RAG
postprocess path as the evaluator, but it no longer requires labels unless
`--evaluate` is explicitly passed. It writes per-sample HDF5 segmentations,
manifest JSON/CSV, metric JSON/CSV when labels are present, and partial outputs
after each sample.

Local smoke:

- Command used one `32x160x160` crop from `mec8nm_11.h5`.
- Local machine had no CUDA, so inference fell back to CPU and took `93.14s`
  for one model window.
- Segmentation HDF5 write succeeded:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/smoke_infer_mec8nm_r57_rag_q25/segmentations/mec8nm_11_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.00_mahotas_agglomeration_z0.50_xy0.50_q25.seg.h5`
- Local metric step failed only because local Python lacks `skimage`; the
  SiFlow bundle installs `scikit-image` from the offline wheelhouse.

SiFlow full-block export:

- SiFlow UUID: `7e5b7bc5-1424-4a6b-9205-f5b0f1e409cb`
- Name: `mec8nm-r57-rag-infer-29384524`
- Bundle root:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/siflow_bundles/mec8nm_r57_fullblock_eval_20260710T201041Z`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-rag-infer.20260711T055837+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_rag_q25_test_fullblock`
- Exported inference directory on TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_rag_q25_test_fullblock/inference_export/infer_mec8nm_r57_rag_q25_test_fullblock`
- Local fetched export:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_rag_q25_test_fullblock/infer_mec8nm_r57_rag_q25_test_fullblock`
- Runtime status: completed successfully; `rag_infer_done` code `0`;
  elapsed `622s`; failures `[]`.
- TOS upload: 16 files, total `18.28MB`, including 2 segmentation HDF5 files.

Exported segmentation files:

- `mec8nm_11_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.00_mahotas_agglomeration_z0.50_xy0.50_q25.seg.h5`
  - shape `(125, 1250, 1250)`, dtype `uint32`, chunks `(16, 256, 256)`,
    compression `gzip`, shuffle `true`, max label `220018`, size `13.59MB`.
- `mec8nm_12_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.00_mahotas_agglomeration_z0.50_xy0.50_q25.seg.h5`
  - shape `(125, 1250, 1250)`, dtype `uint32`, chunks `(16, 256, 256)`,
    compression `gzip`, shuffle `true`, max label `83917`, size `4.63MB`.

Export validation metrics match the previous full-block RAG evaluator exactly:

- Aggregate full-region ARAND `0.47802323170358296`, VOI-sum
  `2.2950284376566454`, VOI-merge `2.1214047768226845`, VOI-split
  `0.17362366083396077`.
- Aggregate seam ARAND/VOI-sum `0.4696152456495122` /
  `2.178971783912311`.
- Aggregate nonseam ARAND/VOI-sum `0.4794701215543917` /
  `2.3012719150379892`.
- `mec8nm_11.h5` full: ARAND `0.5082483202083637`, VOI-sum
  `2.492994975190426`, precision `0.9870399501222566`, recall
  `0.32744337373930654`, `n_pred=202897`, `n_gt=635`, inference
  `64.52s`, postprocess `215.03s`.
- `mec8nm_12.h5` full: ARAND `0.4477981431988022`, VOI-sum
  `2.0970619001228643`, precision `0.999631793112464`, recall
  `0.38146163432436175`, `n_pred=76804`, `n_gt=197`, inference
  `63.37s`, postprocess `125.95s`.

Interpretation: the current best R57+RAG path is now packaged as a reusable
inference/export runner and verified on the two held-out full blocks. It is a
valid prototype export path, but still not the final 2PB production recipe
because the segmentation quality is not yet high enough and the postprocess is
single-volume in-memory RAG rather than a distributed slab/merge system.

## R57 Waterz Full-Block Evaluation

Waterz dependency packaging:

- The existing full-block eval bundle was refreshed with:
  `/volume/med-train/users/dchen02/code/dbMiM/scripts/evaluate_cremi_blockwise_scale.py`
  and
  `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_blockwise_eval_siflow.sh`.
- Bundled waterz source:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/siflow_bundles/mec8nm_r57_fullblock_eval_20260710T201041Z/third_party/waterz`
- Source copied from the known-good postprocess bundle:
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/outputs/siflow_bundles/postprocess_bundle_r48-waterz-fixed_20260703T163917/third_party/waterz`
- Validation before submit:
  `python -m py_compile scripts/evaluate_cremi_blockwise_scale.py ...bundle.../evaluate_cremi_blockwise_scale.py`
  and `bash -n` on the local and bundled eval launchers.

SiFlow waterz threshold grid:

- SiFlow UUID: `f04252df-4042-4bb2-8d08-3c69b03ca299`
- Name: `mec8nm-r57-waterz-full-4237e3b6`
- Bundle root:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/siflow_bundles/mec8nm_r57_fullblock_eval_20260710T201041Z`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-waterz-full.20260711T061724+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r57_seed777_fullblock_test_waterz_tgrid`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_waterz_tgrid_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_waterz_tgrid`
- Runtime status: completed successfully; elapsed `16m1s`; TOS download
  verified 12 objects, failed count 0; `failures=[]` in the eval summary.
- Eval setting: full `125x1250x1250` test blocks `mec8nm_11.h5` and
  `mec8nm_12.h5`, chunk size `32x512x512`, halo `8x64x64`, stride
  `16x80x80`, backend `waterz`, thresholds `0.16,0.18,0.20,0.22,0.24`,
  scoring `hist_quantile`, calibration bias `z=-0.20,y=-0.45,x=-0.45`,
  temperatures `1.0,1.15`, full region only, metric backend `skimage`.

Summary file:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_waterz_tgrid_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_waterz_tgrid/blockwise_eval/eval_mec8nm_r57_seed777_fullblock_test_waterz_tgrid/blockwise_scale_summary.json`

Best deployable unified operating point:

- Backend: `waterz`
- Affinity variant:
  `blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.15`
- Threshold: `0.24`
- Aggregate full-region ARAND `0.02603170952473144`
- Aggregate full-region VOI-sum `0.3077737698910076`
- VOI-merge `0.15242010600856548`
- VOI-split `0.1553536638824421`
- Rand precision `0.9911304829720811`
- Rand recall `0.9576743669995477`
- Rand F-score `0.9739682904752686`
- Average postprocess time per threshold/sample row `15.865640125412028s`

Per-sample metrics at the same unified operating point:

- `mec8nm_11.h5`: ARAND `0.021725528587133502`, VOI-sum
  `0.3320929888523554`, VOI-merge `0.09938396125008425`, VOI-split
  `0.23270902760227116`, precision `0.9831269798945438`, recall
  `0.9734696295858958`, F-score `0.9782744714128665`, `n_pred=17755`,
  `n_gt=635`.
- `mec8nm_12.h5`: ARAND `0.030337890462329375`, VOI-sum
  `0.28345455092965977`, VOI-merge `0.20545625076704674`, VOI-split
  `0.07799830016261304`, precision `0.9991339860496183`, recall
  `0.9418791044131996`, F-score `0.9696621095376706`, `n_pred=7191`,
  `n_gt=197`.

Per-sample diagnostic best rows differ slightly:

- `mec8nm_11.h5` best VOI uses `t=1.0`, threshold `0.24`: ARAND
  `0.017748987273128125`, VOI-sum `0.31510414903886097`.
- `mec8nm_11.h5` best ARAND uses `t=1.0`, threshold `0.20`: ARAND
  `0.017699448654642458`, VOI-sum `0.3249260908357521`.
- `mec8nm_12.h5` best VOI uses `t=1.15`, threshold `0.24`: ARAND
  `0.030337890462329375`, VOI-sum `0.28345455092965977`.
- `mec8nm_12.h5` best ARAND uses `t=1.15`, threshold `0.16`: ARAND
  `0.030059555261500392`, VOI-sum `0.29409652619864635`.

Throughput:

- `mec8nm_11.h5`: inference `64.9753s`, `3.006M voxels/s`, 36 chunks.
- `mec8nm_12.h5`: inference `62.0621s`, `3.147M voxels/s`, 36 chunks.
- Waterz postprocess for a 5-threshold batched sweep is about `15.9s` per
  threshold/sample row after amortizing hierarchy construction.

Comparison to the previous full-block best:

- R57 + mahotas RAG q25 aggregate full: ARAND `0.47802323170358296`,
  VOI-sum `2.2950284376566454`.
- R57 + waterz aggregate full: ARAND `0.02603170952473144`, VOI-sum
  `0.3077737698910076`.
- `mec8nm_11.h5` improves from RAG `0.5082483202083637` / `2.492994975190426`
  to waterz `0.021725528587133502` / `0.3320929888523554` at the unified
  operating point.
- `mec8nm_12.h5` improves from RAG `0.4477981431988022` / `2.0970619001228643`
  to waterz `0.030337890462329375` / `0.28345455092965977`.

Interpretation: the first waterz grid completely changed the MEC readiness
picture. The R57 affinities were much stronger than graph-CC/RAG suggested; the
previous bottleneck was primarily postprocessing. The current best candidate is
R57 + calibrated waterz, not RAG. This first waterz run still lacked
seam/nonseam validation and segmentation HDF5 export, so a focused refinement
and export run were submitted next.

Focused waterz refinement and seam check:

- SiFlow UUID: `d8273e03-f2b7-4441-be88-feb7921434a4`
- Name: `mec8nm-r57-waterz-refine-27b497c4`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-waterz-refine.20260711T063946+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r57_seed777_fullblock_test_waterz_refine_tgrid`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_waterz_refine_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_waterz_refine_tgrid`
- Runtime status: completed successfully; elapsed `49m24s`; 108 metric rows;
  TOS download verified 12 objects, failed count 0; `failures=[]`.
- Eval setting: full held-out blocks 11 and 12, same chunk/halo/stride as
  above, backend `waterz`, thresholds `0.22,0.24,0.26,0.28,0.30,0.32`,
  temperatures `1.10,1.15,1.20`, regions `full,seam,nonseam`.

Summary file:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_waterz_refine_eval_20260710/eval_mec8nm_r57_seed777_fullblock_test_waterz_refine_tgrid/blockwise_eval/eval_mec8nm_r57_seed777_fullblock_test_waterz_refine_tgrid/blockwise_scale_summary.json`

New best deployable unified operating points:

- Best by full-region VOI: `temperature=1.20`, waterz threshold `0.32`,
  aggregate ARAND `0.012404116869218829`, VOI-sum `0.2281928385615599`,
  VOI-merge `0.08810013197019645`, VOI-split `0.14009270659136344`,
  precision `0.9917512735794931`, recall `0.9834751756202845`, F-score
  `0.9875958831307812`.
- Best by full-region ARAND: `temperature=1.20`, waterz threshold `0.24`,
  aggregate ARAND `0.012395526274616486`, VOI-sum `0.23767385917828776`.
- Difference between the ARAND-best and VOI-best points is negligible in
  ARAND (`8.59e-06`) but meaningful in VOI (`0.00948`), so the exported
  prototype uses the VOI-best `t=1.20, threshold=0.32`.

Seam check at `t=1.20, threshold=0.32`:

- Full: ARAND `0.012404116869218829`, VOI-sum `0.2281928385615599`.
- Seam: ARAND `0.01270425209935755`, VOI-sum `0.2123555093184763`.
- Nonseam: ARAND `0.012414726894636186`, VOI-sum `0.22443904647870777`.

The seam metrics are close to full/nonseam and not worse in VOI, so the current
chunk/halo/stride setting does not show an obvious seam artifact on held-out
MEC blocks 11 and 12.

Per-sample full metrics at the exported operating point:

- `mec8nm_11.h5`: ARAND `0.020041091024243007`, VOI-sum
  `0.3121338651679854`, VOI-merge `0.10028389928175853`, VOI-split
  `0.2118499658862269`, precision `0.9841393415079978`, recall
  `0.9758138415482437`, F-score `0.979958908975757`, `n_pred=10454`,
  `n_gt=635`.
- `mec8nm_12.h5`: ARAND `0.00476714271419465`, VOI-sum
  `0.14425181195513437`, VOI-merge `0.07591636465863437`, VOI-split
  `0.06833544729649998`, precision `0.9993632056509885`, recall
  `0.9911365096923254`, F-score `0.9952328572858054`, `n_pred=4223`,
  `n_gt=197`.

Comparison to previous candidates:

- R57 + mahotas RAG q25 aggregate full: ARAND `0.47802323170358296`,
  VOI-sum `2.2950284376566454`.
- First R57 + waterz grid best aggregate full: ARAND `0.02603170952473144`,
  VOI-sum `0.3077737698910076`.
- Focused R57 + waterz refinement best aggregate full: ARAND
  `0.012404116869218829`, VOI-sum `0.2281928385615599`.

The refinement improved the first waterz grid by about 52% relative ARAND and
26% relative VOI, mainly because `temperature=1.20` was dramatically better on
`mec8nm_12`.

## R57 Waterz Inference Export

Deployment-style waterz export changes:

- `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_infer_rag_siflow.sh`
  now installs bundled `third_party/waterz` with bundled Boost headers when the
  runtime image lacks the `waterz` module.
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/infer_mec8nm_blockwise_rag.py`
  now includes `thr<value>` in segmentation filenames, so waterz exports are
  not confused with z/xy thresholded graph/RAG exports.
- The updated infer script and launcher were synced into:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/siflow_bundles/mec8nm_r57_fullblock_eval_20260710T201041Z`
- Validation before submit: `python -m py_compile` on local and bundled
  `infer_mec8nm_blockwise_rag.py`, and `bash -n` on local and bundled
  `run_mec8nm_infer_rag_siflow.sh`.

SiFlow export:

- SiFlow UUID: `0628804d-97bd-434a-9162-db7556825840`
- Name: `mec8nm-r57-waterz-export-ad15eead`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-waterz-export.20260711T073445+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock`
- Exported inference directory on TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock`
- Local fetched export:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock`
- Runtime status: completed successfully; `rag_infer_done` code `0`; elapsed
  `412s`; task duration `7m28s`; failures `[]`.
- TOS upload: 19 files, total `21.35MB`, including 2 segmentation HDF5 files;
  upload failed count 0.

Export summary:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/inference_summary.json`

Exported segmentation files:

- `mec8nm_11_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.20_waterz_thr0.32_z0.50_xy0.50_mean.seg.h5`
  - local path:
    `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/segmentations/mec8nm_11_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.20_waterz_thr0.32_z0.50_xy0.50_mean.seg.h5`
  - shape `(125, 1250, 1250)`, dtype `uint32`, chunks `(16, 256, 256)`,
    compression `gzip` level 4, shuffle `true`, max label `528494`, local
    size `15334671` bytes.
- `mec8nm_12_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.20_waterz_thr0.32_z0.50_xy0.50_mean.seg.h5`
  - local path:
    `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr032_test_fullblock/segmentations/mec8nm_12_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.20_waterz_thr0.32_z0.50_xy0.50_mean.seg.h5`
  - shape `(125, 1250, 1250)`, dtype `uint32`, chunks `(16, 256, 256)`,
    compression `gzip` level 4, shuffle `true`, max label `209960`, local
    size `6965011` bytes.

Export validation metrics match the focused evaluator exactly:

- Aggregate full-region ARAND `0.012404116869218829`, VOI-sum
  `0.2281928385615599`, VOI-merge `0.08810013197019645`, VOI-split
  `0.14009270659136344`.
- Aggregate seam ARAND/VOI-sum `0.01270425209935755` /
  `0.2123555093184763`.
- Aggregate nonseam ARAND/VOI-sum `0.012414726894636186` /
  `0.22443904647870777`.
- `mec8nm_11.h5` full: ARAND `0.020041091024243007`, VOI-sum
  `0.3121338651679854`, precision `0.9841393415079978`, recall
  `0.9758138415482437`, `n_pred=10454`, `n_gt=635`, inference `64.09s`,
  waterz postprocess `72.32s`.
- `mec8nm_12.h5` full: ARAND `0.00476714271419465`, VOI-sum
  `0.14425181195513437`, precision `0.9993632056509885`, recall
  `0.9911365096923254`, `n_pred=4223`, `n_gt=197`, inference `59.37s`,
  waterz postprocess `59.12s`.

Interpretation: R57 + calibrated waterz is now not just the best evaluation
candidate but also a verified deployment-style export path for labeled held-out
MEC blocks. The remaining production gap is no longer per-block segmentation
quality on these two blocks; it is large-volume execution: slab scheduling,
memory caps, cross-slab ID stitching, and output metadata/storage layout for
2PB-scale inference.

High-threshold guardrail:

- SiFlow UUID: `0f95ca7e-89c6-4d87-9b36-b6f5481e7204`
- Name: `mec8nm-r57-waterz-guard-36f102fe`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-waterz-guard.20260711T075021+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r57_seed777_fullblock_test_waterz_guardrail_tgrid`
- Local fetched outputs:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/mec8nm_r57_waterz_guardrail_eval_20260711/eval_mec8nm_r57_seed777_fullblock_test_waterz_guardrail_tgrid`
- Runtime status: completed successfully; elapsed `19m29s`; 24 full-region
  metric rows; TOS download verified 12 objects, failed count 0;
  `failures=[]`.
- Eval setting: temperatures `1.18,1.20,1.22`, thresholds
  `0.30,0.32,0.34,0.36`, full region only.

Guardrail result:

- Best by full-region VOI: `temperature=1.20`, waterz threshold `0.36`,
  aggregate ARAND `0.01245975276529443`, VOI-sum `0.2240353497868297`,
  precision `0.9919125895974171`, recall `0.9832063764280745`.
- Best by full-region ARAND remains `temperature=1.20`, waterz threshold
  `0.32`, aggregate ARAND `0.012404116869218829`, VOI-sum
  `0.2281928385615599`.
- The `0.36` point improves VOI by `0.004157488774730205` absolute versus
  `0.32`, with an ARAND cost of `0.000055635896075601`.

0.36 waterz deployment export:

- SiFlow UUID: `14a75e70-7ccd-4d80-98cf-2344e28ee087`
- Name: `mec8nm-r57-waterz-export36-b6432ec1`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_fullblock_eval/mec8nm-r57-waterz-export36.20260711T081317+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock`
- Exported inference directory on TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock`
- Local fetched export:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock`
- Runtime status: completed successfully; `rag_infer_done` code `0`; elapsed
  `414s`; task duration `7m30s`; failures `[]`.
- TOS upload: 19 files, total `21.21MB`, including 2 segmentation HDF5 files;
  upload failed count 0.

0.36 export summary:

`/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/inference_summary.json`

0.36 exported segmentation files:

- `mec8nm_11_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.20_waterz_thr0.36_z0.50_xy0.50_mean.seg.h5`
  - local path:
    `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/segmentations/mec8nm_11_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.20_waterz_thr0.36_z0.50_xy0.50_mean.seg.h5`
  - shape `(125, 1250, 1250)`, dtype `uint32`, chunks `(16, 256, 256)`,
    compression `gzip` level 4, shuffle `true`, max label `528494`, local
    size `15259645` bytes.
- `mec8nm_12_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.20_waterz_thr0.36_z0.50_xy0.50_mean.seg.h5`
  - local path:
    `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/segmentations/mec8nm_12_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.20_waterz_thr0.36_z0.50_xy0.50_mean.seg.h5`
  - shape `(125, 1250, 1250)`, dtype `uint32`, chunks `(16, 256, 256)`,
    compression `gzip` level 4, shuffle `true`, max label `209905`, local
    size `6898373` bytes.

0.36 export validation metrics:

- Aggregate full-region ARAND `0.01245975276529443`, VOI-sum
  `0.2240353497868297`, VOI-merge `0.09257016288979208`, VOI-split
  `0.13146518689703762`.
- Aggregate seam ARAND/VOI-sum `0.013031550879182785` /
  `0.21150499794760833`.
- Aggregate nonseam ARAND/VOI-sum `0.012408026169544673` /
  `0.21977815247519208`.
- `mec8nm_11.h5` full: ARAND `0.0201860165442711`, VOI-sum
  `0.3089244307714282`, precision `0.9843706821974486`, recall
  `0.975299276681919`, `n_pred=7839`, `n_gt=635`, inference `63.65s`,
  waterz postprocess `72.38s`.
- `mec8nm_12.h5` full: ARAND `0.004733488986317758`, VOI-sum
  `0.1391462688022312`, precision `0.9994544969973856`, recall
  `0.9911134761742302`, `n_pred=3173`, `n_gt=197`, inference `59.80s`,
  waterz postprocess `59.33s`.

Current recommendation: keep both `0.32` and `0.36` recorded. Use `0.32` if
strictly optimizing ARAND; use `0.36` if optimizing VOI/proofreading burden.
The stronger deployment export is currently `t=1.20, threshold=0.36` because
its seam/nonseam metrics are stable and its aggregate VOI is best.

## Slab Stitching Prototype

Prototype script:

`/volume/med-train/users/dchen02/code/dbMiM/scripts/stitch_mec8nm_slab_segmentations.py`

Purpose: validate the sparse cross-slab ID remap layer needed for 2PB-scale
deployment. The script takes an existing segmentation HDF5, creates synthetic
overlapping slabs with independent label-ID offsets, builds overlap contact
pairs between adjacent slabs, unions accepted cross-slab IDs, and composes a
stitched output from non-overlapping slab cores. It writes:

- stitched HDF5 dataset `segmentation`;
- `overlap_contacts.csv`;
- `stitch_summary.json`, including partition-equivalence diagnostics versus
  the source segmentation.

Acceptance policy used in the current smoke tests: merge a cross-slab pair when
`contact_voxels >= 1` and `min(overlap_fraction_a, overlap_fraction_b) >= 0.5`.
This is intentionally permissive for synthetic-slab validation; real slabwise
inference should sweep/contact-calibrate this against GT and seam metrics.

Synthetic connected-object unit test:

- Source:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/slab_stitch_synthetic_unit_20260711/synthetic_connected.seg.h5`
- Output directory:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/slab_stitch_synthetic_unit_20260711/stitch`
- Shape: `[72,96,96]`; slabs: `3`; contacts accepted: `6/6`.
- Result: partition-equivalent to source; reference labels `4`, stitched
  labels `4`, split labels `0`, merge labels `0`.

MEC12 crop diagnostic on the exported `t=1.20,threshold=0.36` segmentation:

- Input:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/inference_export/infer_mec8nm_r57_waterz_t120_thr036_test_fullblock/segmentations/mec8nm_12_blockwise_halo_bias_z-0.20_y-0.45_x-0.45_t1.20_waterz_thr0.36_z0.50_xy0.50_mean.seg.h5`
- Output directory:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/slab_stitch_smoke_mec8nm12_thr036_crop_diag_20260711`
- Crop start `[0,256,256]`, crop shape `[64,512,512]`, axis `0`,
  slab size `32`, overlap `8`.
- Result: contacts accepted `268/268`; reference labels `987`, stitched labels
  `988`; reference split labels `1`, stitched merge labels `0`; elapsed
  `4.626s`.
- Interpretation: the overlap-only rule made no false merge in this crop, but
  one source label appeared as disconnected/non-contact pieces across slab
  cores and therefore could not be recovered from overlap contacts alone. This
  is a useful diagnostic for disconnected same-ID components, not a failure of
  the sparse union step.

MEC12 full-volume synthetic slab smoke on the same exported segmentation:

- Output directory:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/slab_stitch_smoke_mec8nm12_thr036_full_20260711`
- Summary JSON:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/slab_stitch_smoke_mec8nm12_thr036_full_20260711/stitch_summary.json`
- Stitched HDF5:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/slab_stitch_smoke_mec8nm12_thr036_full_20260711/mec8nm_12_thr036_full_synthetic_slab_stitched_axis0_slab48_ov12.seg.h5`
- Shape: `[125,1250,1250]`; slabs: `4`; axis `0`; slab size `48`;
  overlap `12`.
- Contacts: `1937`; accepted contacts: `1937`.
- Result: partition-equivalent to source over `195312500` non-ignore voxels;
  reference labels `5963`, stitched labels `5963`, split labels `0`, merge
  labels `0`; elapsed `47.699s`.

Conclusion: the sparse overlap-contact/union stitching layer can recover stable
global IDs exactly on a full MEC12 block when each synthetic slab comes from the
same underlying full-block segmentation. The remaining production question is
real slabwise inference, where slab-local waterz may produce different local
hierarchies near borders. The next test should run the actual R57 calibrated
waterz inference independently on adjacent overlapping slabs, emit local slab
HDF5 files plus border contact tables, stitch those IDs, and compare stitched
ARAND/VOI plus seam/nonseam metrics against the validated full-block export.

## Real Slabwise Waterz Validation

Implemented real slabwise inference/stitching:

- `/volume/med-train/users/dchen02/code/dbMiM/scripts/infer_mec8nm_slabwise_stitch.py`
  runs independent model+postprocess inference for each overlapping slab, saves
  slab-local HDF5 files, builds overlap contacts, stitches IDs, writes full
  stitched HDF5, computes full/slab-seam/slab-nonseam/chunk-seam metrics, and
  optionally compares partition equivalence against a reference full-block
  segmentation.
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/run_mec8nm_slabwise_stitch_siflow.sh`
  is the TOS-bootstrap SiFlow launcher with offline wheel/waterz install,
  MEC/checkpoint/reference download, and TOS upload.
- `/volume/med-train/users/dchen02/code/dbMiM/scripts/restitch_mec8nm_saved_slabs.py`
  reuses saved slab HDF5 files to sweep contact/stitching thresholds without
  rerunning GPU inference or waterz.

Important contact-logic fix:

- The first real slabwise run exposed a false-merge bug in the contact policy.
  The old code counted `left_overlap_voxels` and `right_overlap_voxels` only on
  voxels where both slabs were nonzero, and accepted a pair if
  `contact / min(left_count,right_count) >= ratio`. This let a large object
  union with many small labels.
- The current code counts each side over the full overlap region and accepts a
  cross-slab pair only if both `contact/left_overlap_voxels` and
  `contact/right_overlap_voxels` pass the threshold. The synthetic unit test was
  rerun after this change and remained partition-equivalent true.

MEC12 slab48/overlap12 first full-block run:

- SiFlow UUID: `26e950f9-ce98-4a5c-be82-26718da05e2c`
- Name: `mec8nm-r57-slabwise-3abc868a`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_slabwise_stitch/mec8nm-r57-slabwise.20260711T085142+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_fullblock`
- Local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_fullblock/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_fullblock`
- Setting: `mec8nm_12`, R57, bias `(-0.20,-0.45,-0.45)`,
  temperature `1.20`, waterz threshold `0.36`, z-axis slab size `48`,
  overlap `12`, 4 slabs.
- Runtime: task succeeded; 20 TOS objects, 12.24MB; local fetch verified 17
  files, failed count 0.
- Old permissive contact policy result: contacts accepted `3751/8574`, full
  ARAND/VOI `0.41142833541688406` / `1.634950951942724`, slab-seam ARAND/VOI
  `0.41496077692293654` / `1.6465590323045165`.
- Interpretation: this was a stitching-policy failure, not a useful production
  result. Reference equivalence to the full-block export showed massive false
  merges (`stitched_merge_labels=1735`) and splits (`reference_split_labels=1933`).

MEC12 slab48/overlap12 saved-slab restitch after bidirectional contact fix:

- Local sweep:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/restitch_mec8nm12_slabwise_waterz_contact_sweep_20260711`
- TOS sweep:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/restitch_mec8nm12_slabwise_waterz_contact_sweep_20260711`
- Uploaded 34 objects, 146.54MB, failed count 0.
- Completed variants summary:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/restitch_mec8nm12_slabwise_waterz_contact_sweep_20260711/completed_variant_summary.csv`
- Best ARAND in completed sweep: `min_contact=1`, `min_overlap_ratio=0.70`,
  contacts accepted `396/8574`, full ARAND/VOI `0.01939665858432904` /
  `0.31143769709526037`, slab-seam ARAND/VOI `0.024697000855473994` /
  `0.3477222305389267`.
- Best VOI in completed sweep: `min_contact=1`, `min_overlap_ratio=0.50`,
  contacts accepted `523/8574`, full ARAND/VOI `0.019400472486658327` /
  `0.31037431592651155`, slab-seam ARAND/VOI `0.024697283127992353` /
  `0.34273408713918885`.
- `min_contact=64` gave effectively identical metrics at the same ratios.
- `min_overlap_ratio=0.98` is too strict: full ARAND/VOI
  `0.4554590647255914` / `1.6712255417722486`, recall `0.37831735832501034`.

MEC12 slab96/overlap32 full-block run with corrected contact logic:

- SiFlow UUID: `752fb9cb-f55e-4a37-b0f3-cdc28868984a`
- Name: `mec8nm-r57-slab96-3be17ec`
- Bundle TOS:
  `tos://agi-data/users/dchen02/dbmim/bundles/mec8nm_r57_slabwise_stitch/mec8nm-r57-slab96.20260711T092338+0800.tar.gz`
- Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_slab96ov32_bidir`
- Local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_slab96ov32_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_slab96ov32_bidir`
- Setting: `mec8nm_12`, same R57 calibration/waterz threshold, z-axis slab
  size `96`, overlap `32`, `min_contact=64`, `min_overlap_ratio=0.5`, 2 slabs.
- Runtime: task succeeded; 18 TOS objects, 13.69MB; local fetch verified 15
  files, failed count 0; elapsed `302.124s`; total inference `78.516s`, total
  waterz postprocess `77.323s`.
- Contacts accepted `446/5775`.
- Full ARAND/VOI `0.01100604415162898` / `0.19367825102181688`.
- Slab-seam ARAND/VOI `0.011420920051073535` / `0.16024846914997412`.
- Slab-nonseam ARAND/VOI `0.010990858167303053` /
  `0.19286883829589557`.
- Chunk-seam ARAND/VOI `0.020453231241111314` /
  `0.2624569192595848`; chunk-nonseam ARAND/VOI
  `0.008795132292319874` / `0.17278901648557426`.
- Reference equivalence to the validated full-block segmentation is still not
  exact, but much closer than slab48: reference split labels `807`, stitched
  merge labels `775`, stitched labels `5928` versus reference labels `5963`.

MEC11+MEC12 y-axis slab768/overlap256 full-block runs with corrected contact
logic:

- MEC12 SiFlow UUID: `0762e685-662a-4189-a221-9f936e48c9f6`
- MEC12 name: `mec8nm-r57-yaxis-fd50f7cb`
- MEC12 Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_yaxis768ov256_bidir`
- MEC12 local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_yaxis768ov256_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_yaxis768ov256_bidir`
- MEC11 SiFlow UUID: `0ac76616-c011-44ef-948a-ae41c9690172`
- MEC11 name: `mec8nm-r57-yaxis11-ca098737`
- MEC11 Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_yaxis768ov256_bidir`
- MEC11 local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_yaxis768ov256_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_yaxis768ov256_bidir`
- Setting: same R57 calibration/waterz threshold, y-axis slab size `768`,
  overlap `256`, `min_contact=64`, `min_overlap_ratio=0.5`, 2 slabs per
  sample.
- MEC12 runtime: task succeeded; 18 TOS objects, 13.15MB; local fetch verified
  15 files, failed count 0; elapsed `295.438s`; total inference `73.699s`,
  total waterz postprocess `72.295s`; contacts accepted `236/4761`.
- MEC12 full ARAND/VOI `0.0017551312249106665` /
  `0.11586653310233763`; slab-seam `0.00090018686490001` /
  `0.05213740987928195`; slab-nonseam `0.001755021201838347` /
  `0.11529666821561399`; chunk-seam `0.0023193102490866035` /
  `0.11706699857863406`; chunk-nonseam `0.0016240493486473495` /
  `0.11187815761626685`.
- MEC12 reference equivalence is not exact: reference split labels `777`,
  stitched merge labels `847`, stitched labels `6082` versus reference labels
  `5963`.
- MEC11 runtime: task succeeded; 18 TOS objects, 28.41MB; local fetch verified
  18 files, failed count 0; elapsed about `304s`; total inference `73.304s`,
  total waterz postprocess `82.999s`; contacts accepted `675/10545`.
- MEC11 full ARAND/VOI `0.01851894845955182` /
  `0.2987442353422687`; slab-seam `0.016594606689556257` /
  `0.17560140352178374`; slab-nonseam `0.018606739269794392` /
  `0.29854838633382563`; chunk-seam `0.013663815380538069` /
  `0.23116866677447367`; chunk-nonseam `0.019741863947272842` /
  `0.30429315699726406`.
- MEC11 reference equivalence is not exact: reference split labels `5477`,
  stitched merge labels `5729`, stitched labels `23664` versus reference labels
  `23221`.
- Held-out aggregate across MEC11+12 improves versus the validated full-block
  threshold-0.36 export: y-axis full ARAND/VOI
  `0.010137039842231244` / `0.20730538422230316` versus full-block
  `0.01245975276529443` / `0.2240353497868297`, an absolute delta of
  `-0.002322712923063186` ARAND and `-0.01672996556452655` VOI-sum.

MEC11+MEC12 x-axis slab768/overlap256 full-block runs with corrected contact
logic:

- MEC12 SiFlow UUID: `4e1ac535-d18f-434e-9ffc-0da3e1f2c606`
- MEC12 name: `mec8nm-r57-xaxis12-fc7a1ad4`
- MEC12 Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_xaxis768ov256_bidir`
- MEC12 local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_xaxis768ov256_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec12_xaxis768ov256_bidir`
- MEC11 SiFlow UUID: `cd189657-ea56-485b-995e-b2796a41fc23`
- MEC11 name: `mec8nm-r57-xaxis11-59db30a6`
- MEC11 Output TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_xaxis768ov256_bidir`
- MEC11 local fetch:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_xaxis768ov256_bidir/infer_mec8nm_r57_waterz_slabwise_t120_thr036_mec11_xaxis768ov256_bidir`
- Setting: same R57 calibration/waterz threshold, x-axis slab size `768`,
  overlap `256`, `min_contact=64`, `min_overlap_ratio=0.5`, 2 slabs per
  sample.
- MEC12 runtime: task succeeded; 18 TOS objects, 16.60MB; local fetch verified
  key outputs, failed count 0; total inference `74.081s`, total waterz
  postprocess `75.116s`; contacts accepted `675/15350`.
- MEC12 full ARAND/VOI `0.008688286060755335` /
  `0.1771920520371964`; slab-seam `0.1356215866736129` /
  `0.6768592154736054`; slab-nonseam `0.007160802613843553` /
  `0.15967196138702794`; chunk-seam `0.009744381730203311` /
  `0.17318560315897635`; chunk-nonseam `0.008419032347280475` /
  `0.17288627307088467`.
- MEC12 reference equivalence is not exact and much worse than y-axis:
  reference split labels `1875`, stitched merge labels `1822`, stitched labels
  `5879` versus reference labels `5963`.
- MEC11 runtime: task succeeded; 18 TOS objects, 30.24MB; local fetch verified
  key outputs, failed count 0; total inference `74.639s`, total waterz
  postprocess `83.775s`; contacts accepted `1273/23072`.
- MEC11 full ARAND/VOI `0.023795672809631463` /
  `0.33025003744108983`; slab-seam `0.0521669754117593` /
  `0.32409360547645377`; slab-nonseam `0.02296797080439994` /
  `0.3253283710377509`; chunk-seam `0.01626882102590088` /
  `0.2542755072467261`; chunk-nonseam `0.025696701139095168` /
  `0.33636848877645775`.
- MEC11 reference equivalence is not exact: reference split labels `4832`,
  stitched merge labels `4781`, stitched labels `23416` versus reference labels
  `23221`.
- Held-out aggregate across MEC11+12 worsens versus both full-block and
  y-axis: x-axis full ARAND/VOI `0.0162419794351934` /
  `0.25372104473914314`, an absolute delta of `+0.0037822266698989693`
  ARAND and `+0.029685694952313435` VOI-sum versus full-block.

Interpretation:

- The corrected contact policy is mandatory. It reduced the slab48 result from
  ARAND/VOI `0.4114/1.6350` to about `0.0194/0.3104` without rerunning
  inference.
- Larger z slabs materially help: slab96/overlap32 improves MEC12 to
  `0.0110/0.1937`, approaching but still not matching the validated full-block
  export for the same sample (`0.004733488986317758` /
  `0.1391462688022312`).
- The y-axis `768/256` variant unexpectedly improves both held-out samples and
  is now the best measured MEC11+12 score for this R57/waterz family. This is
  a real segmentation change, not a partition-equivalent deployment transform:
  both MEC11 and MEC12 still have split/merge differences relative to the
  full-block export. The likely mechanism is that local y-axis waterz
  hierarchies avoid some full-block merge errors while preserving seam quality.
- The x-axis `768/256` variant does not reproduce the y-axis gain: MEC11 and
  MEC12 both worsen, the held-out aggregate degrades to `0.01624/0.25372`, and
  MEC12 slab-seam quality is especially poor. This makes the y-axis result a
  useful candidate and diagnostic, but not a robust axis-agnostic deployment
  recipe.
- Axis/slab choice is therefore a quality hyperparameter, not just an
  implementation detail. For 2PB deployment, independent slab waterz plus
  overlap ID union is not guaranteed to reproduce full-block waterz exactly.
  The next production method should explain or eliminate this axis sensitivity,
  test larger XY tiles, and still move toward saving enough hierarchy/affinity
  state for a sparse cross-slab re-agglomeration step.

## Exploratory Y-Axis Threshold Sweep

The original R57 training configuration contains only MEC01--07 and MEC09;
the documented block split is MEC08/10 validation and MEC11/12 test. The
following runs intentionally re-used MEC11/12 to diagnose the slabwise
operating point after the axis result. They are useful quality evidence, but
they must not be used as the sole final deployment selection criterion.

- Fixed setting: R57, calibrated waterz bias `(-0.20,-0.45,-0.45)`,
  temperature `1.20`, y-axis slabs `768/256`, seam width `16`,
  `min_contact=64`, overlap ratio `0.5`. The combined two-block jobs did not
  download a full-block reference H5, so they report GT quality but not
  partition-equivalence diagnostics.
- `threshold=0.32`: aggregate full ARAND/VOI `0.00808479052751726` /
  `0.20173006746063996`; MEC11 `0.014398438631627708` /
  `0.2827383733454798`, MEC12 `0.001771142423406813` /
  `0.12072176157580011`. TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr032_mec11_12_yaxis768ov256_bidir`.
- `threshold=0.34`: aggregate full ARAND/VOI `0.008043419441299271` /
  `0.19705134574877856`; MEC11 `0.014325571291376571` /
  `0.2778847974468879`, MEC12 `0.001761267591221971` /
  `0.11621789405066923`. This is the best exploratory point by both metrics,
  improving the prior y-axis `0.36` point by `0.002093620400931973` ARAND and
  `0.0102540384735246` VOI. TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr034_mec11_12_yaxis768ov256_bidir`.
- `threshold=0.38`: aggregate full ARAND/VOI `0.017533601493767037` /
  `0.2474961442075552`; MEC12 collapses to `0.01722913759283029` /
  `0.20265513715250927`. This is a clear upper-threshold guardrail. TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr038_mec11_12_yaxis768ov256_bidir`.
- Fine sweep completion: `threshold=0.33` gives aggregate full ARAND/VOI
  `0.008059426682463788` / `0.20080293819276018`; `threshold=0.35` gives
  `0.008024889280985659` / `0.19599678807900686`. `0.35` is the numerical
  leader in this already-inspected test sweep, but only by `0.000018530160313612`
  ARAND and `0.0010545576697717` VOI relative to `0.34`; that difference needs
  validation rather than test-driven deployment selection. TOS outputs:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr033_mec11_12_yaxis768ov256_bidir` and
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr035_mec11_12_yaxis768ov256_bidir`.
- Wider y-axis context is not monotonically beneficial: at `threshold=0.34`,
  slab `1024/384` gives aggregate full ARAND/VOI `0.05953844433758815` /
  `0.3909622131062548`; MEC12 is especially bad at `0.09495406043461507` /
  `0.4493539555728385`, driven by split VOI `0.3713846383685524`. Do not
  validate or deploy this geometry. TOS:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/infer_mec8nm_r57_waterz_slabwise_t120_thr034_mec11_12_yaxis1024ov384_bidir`.
- All three task records succeeded and local exports are under
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/tos_fetch/` with the
  corresponding run name.

The pre-existing full-block grid also contains a direct `threshold=0.34`
control: MEC11 `0.020213424084903564` / `0.3095975491427005`, MEC12
`0.004758674387605799` / `0.13977085600373457`, aggregate
`0.012486049236254682` / `0.22468420257321753`. The y-axis result therefore
improves same-threshold full-block by `0.0044426297949554105` ARAND and
`0.027632856824438973` VOI, so the gain is principally a slab-local hierarchy
effect rather than scalar-threshold tuning. It is also better than the
previous full-block `0.36` reference by `0.004416333323995159` ARAND and
`0.02698400403805113` VOI. The next decision is deliberately validation-led:
run the local threshold neighborhood on held-out validation MEC08/10, select
there, then record a final locked-parameter MEC11/12 audit. The completed
task UUIDs are `8525897d-af37-475a-9af2-7ff806e27e0a` (`1024/384`),
`c08bd83a-9470-485d-be85-fd5973d8fd18` (`0.33`), and
`7e91c732-150f-47bd-81be-a6e3b46dc801` (`0.35`).

## Next Steps

1. Do not deploy stage1 or stage2 directly. Both are center-crop baselines with
   high held-out ARAND; stage2 also shows obvious over-splitting.
2. Do not spend more time on simple graph-CC z/xy threshold tuning for this
   R57 checkpoint. The full-block grid confirms `0.5/0.5` is already best in
   the tested range.
3. Treat R57 + calibrated waterz as the current best family. The stable
   full-block export operating point remains
   `bias_z=-0.20,bias_y=-0.45,bias_x=-0.45`, `temperature=1.20`,
   `waterz threshold=0.36`, with aggregate full-block ARAND/VOI
   `0.01246/0.22404` on held-out MEC blocks 11 and 12. The exploratory
   y-axis `768/256`, threshold `0.34` point is now the numerical leader on
   already-inspected MEC11/12 (`0.008043/0.197051`), but is neither
   partition-equivalent nor selected on the official validation split. Keep
   it provisional until MEC08/10 selects the operating point.
4. The narrow threshold search has identified a sharp useful neighborhood:
   `0.34` beats `0.32` and `0.36`, while `0.38` is materially worse. Validate
   `0.33/0.34/0.35` on MEC08/10 and stop scalar tuning after that decision;
   the immediate production blocker remains slab hierarchy/stitching, not an
   unbounded threshold search.
5. Keep the RAG export as a validated fallback path, not as the headline
   candidate. It is useful for comparing export mechanics, but quality is now
   far behind waterz.
6. The synthetic slab/stitching prototype is now validated on both a controlled
   connected-object unit test and full MEC12 exported segmentation. Real
   slabwise waterz has been tested on MEC12 z-axis and MEC11+12 y-axis. Use the
   corrected bidirectional contact policy; the old one is invalid. z-slab
   `96/32` is much better than `48/12`; y-slab `768/256` is the current best
   held-out score but remains non-equivalent to full-block waterz; x-slab
   `768/256` is worse than full-block and exposes axis sensitivity.
7. Package a checkpoint/postprocess recipe for 2PB inference only after a
   candidate is selected on MEC08/10 and audited once on MEC11/12, with real
   slabwise seam/nonseam metrics, memory limits, and TOS output layout on the
   intended deployment axes. Next tests should prioritize larger XY
   postprocess tiles, axis-aware diagnostics, and hierarchy-aware cross-slab
   re-agglomeration rather than more scalar contact-threshold sweeps.

## Official Validation Gate And R58 Robustness Wave

The MEC11/12 y-axis threshold sweep above was exploratory only. The official
validation blocks are MEC08 and MEC10, and they now rule out selecting that
R57 slabwise operating point for deployment:

- R57, calibrated waterz bias `(-0.20,-0.45,-0.45)`, temperature `1.20`,
  y-axis slab `768/256`, and the corrected bidirectional contact rule gave
  aggregate full MEC08+MEC10 ARAND/VOI `0.40385415920940215` /
  `1.5406254233101775` at threshold `0.33`, `0.40389850376670944` /
  `1.5407018490152256` at `0.34`, and `0.4038303339256861` /
  `1.5375763983790238` at `0.35`.
- Therefore neither the numerical MEC11/12 leader (`0.35`) nor the prior
  `0.34` point is a validated or deployable setting. The failure is not a
  narrow threshold-selection issue.

A separate full-block fixed-calibration low-threshold control completed:

- SiFlow UUID: `0158897c-656c-4ab6-9e7d-628fdeb53127`
  (`mec8nm-r57-vallowthr-2b62c087`).
- TOS output:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r57_seed777_fullblock_val_waterz_lowthr_tgrid`.
- It used R57, the same fixed bias/temperature, full blocks with
  `32x512x512` chunks and `8x64x64` halos, and waterz thresholds
  `0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32`.
- The best single shared threshold is `0.16`, but still has mean MEC08/MEC10
  full ARAND/VOI `0.2522797956118525` / `1.0622358210386302`.
  MEC08 reaches its best point at `0.20` (`0.15125425078989474` /
  `0.8447865938513939`), while MEC10 remains poor: its best ARAND is
  `0.34864368479038776` at `0.02` with VOI `1.9334350039519026`, and its
  best VOI is `1.267388784204806` at `0.16` with ARAND
  `0.3520898455730189`.
- This establishes a fixed-calibration full-block validation anchor and
  confirms that lowering a scalar waterz threshold cannot repair the MEC10
  domain failure.

Two label-free auto-calibration probes were stopped after a conclusive
negative early result on MEC08. Q50 targets `(0.45,0.35,0.35)` and
`(0.55,0.45,0.45)` produced near-one ARAND and about 18 VOI at low thresholds;
they are not candidates for this checkpoint.

R58 is the controlled robustness response, not another threshold sweep:

- Config:
  `/volume/med-train/users/dchen02/code/dbMiM/configs/finetune_mec8nm_r58_seed1789_balanced_stage1.yaml`.
- It starts from the public R57 checkpoint, trains only MEC01--07 and MEC09
  for `12000` steps, uses stronger gamma/noise augmentation, `weighted_mse`
  with ratio weight alpha `1.0`, boundary Dice weight `0.15`, BCAR ranking
  weight `0.10`, and normalized membrane-edge weighting `1.25`.
- The first submission, UUID `02f3ad0a-724f-4314-b3b1-21f7e5f2ad25`, was
  intentionally stopped during the first minute after its module probe showed
  `waterz=false`. The original stage1 launcher did not install bundled waterz,
  so it would have failed only after training when invoking the requested
  waterz validation.
- The launcher now conditionally installs `third_party/waterz` with the
  bundled Boost headers whenever `EVAL_BACKENDS` contains `waterz`, then
  verifies import success before downloading/training. The valid retry is
  SiFlow UUID `c34cc14e-eb87-4d43-a05d-a45aea2edba5`
  (`mec8nm-r58-balanced-wz-b9fbf56e`), submitted on
  `cn-shanghai/changliu/med-model` as one 8-GPU pack.
- Expected TOS output:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r58_seed1789_balanced_stage1/`.
- The entrypoint is intentionally `EVAL_SPLITS=val`; it will evaluate only
  MEC08/10 after training. A full-block calibrated-waterz validation sweep
  will select any R58 operating point before a single locked MEC11/12 audit.

The data topology check explains why this is a real generalization problem.
Using 24 deterministic `32x160x160` crops per block, the eight train blocks
average foreground `0.890260`, boundary fraction `0.227526`, and `22.942708`
nonzero instances per crop. MEC08/10 average foreground `0.985221`, boundary
fraction `0.038641`, and only `4.145833` nonzero instances; MEC08 alone is
near a single-object regime (`0.995469` foreground, `0.010093` boundary,
`1.375` nonzero instances). Per-crop 1/99 normalization is already used, so
this is primarily an object-topology shift rather than global intensity scale.

R59 is the concurrent, single-variable topology-aware loss ablation:

- Config:
  `/volume/med-train/users/dchen02/code/dbMiM/configs/finetune_mec8nm_r59_seed1789_dpp_stage1.yaml`.
- It is identical to R58 in seed, public-R57 initialization, MEC01--07/09
  training data, augmentation, weighted MSE, boundary Dice, BCAR, and
  membrane weighting. It only adds a learnable affinity postprocessor and
  differentiable postprocess (DPP) loss weight `0.04`; DPP trains two-hop
  affinity paths, same-object smoothness, and boundary contrast.
- SiFlow UUID: `dd467b76-4bb8-4266-ac09-6b8204121e96`
  (`mec8nm-r59-dpp-e689538c`), one 8-GPU
  `cn-shanghai/changliu/med-model` pack. Startup verified `waterz=true` and
  an 88-parameter `differentiable_postprocess` optimizer group. It evaluates
  only MEC08/10 and writes to
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r59_seed1789_dpp_stage1/`.

The stage1 launcher now treats train and eval pipeline failures explicitly:
it records each exit code and uploads the checkpoint/logs before exiting. This
prevents an evaluation-only dependency failure from discarding a completed
remote training run.

R60 is the data-domain counterpart, completing the 24-GPU bounded validation
wave without leaking validation/test blocks:

- Config:
  `/volume/med-train/users/dchen02/code/dbMiM/configs/finetune_mec8nm_r60_seed1977_simpledomain_stage1.yaml`.
- It keeps the R58 architecture, weighted loss, augmentation, and public-R57
  initialization, but restricts training to the two low-boundary train blocks
  MEC07 and MEC09 for `6000` steps. This tests the topology-domain hypothesis
  directly rather than changing waterz settings.
- SiFlow UUID: `0791406a-b6ca-4edc-b9c1-b5b2fe3dc21d`
  (`mec8nm-r60-simple-d04fe35`), one 8-GPU
  `cn-shanghai/changliu/med-model` pack. Its runtime config logged
  `train_ids=[7,9]`, `train_paths=2`, `dataset_size=8192`, `waterz=true`, and
  214 loaded pretrained keys. It evaluates only MEC08/10 and writes to
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r60_seed1977_simpledomain_stage1/`.

R58, R59, and R60 are the complete active training wave: three 8-GPU packs,
for the user-approved 24-GPU maximum. No additional training task should be
submitted until their MEC08/10 metrics have been compared.

## 2026-07-11 R58/R59 Crop Screen And Full-Block Gate

All three training tasks have now completed successfully. Their crop-level
MEC08/10 outputs are screening evidence only: model and operating-point
selection remains reserved for the matching full-block official validation.

- R58 crop screen: the shared zero-bias, temperature `1.15`, threshold `0.50`
  point has mean ARAND/VOI `0.02254600366656634` /
  `0.3111834507513943`. Per block: MEC08 `0.014948239287021159` /
  `0.2561279667535462`; MEC10 `0.03014376804611152` /
  `0.3662389347492424`.
- R59 crop screen: its best ARAND point is fixed calibration
  `(-0.20,-0.45,-0.45)`, temperature `1.00`, threshold `0.50`, with mean
  `0.02268401440455803` / `0.28737279263420173`. Its best VOI point is zero
  bias, temperature `1.15`, threshold `0.50`, with mean
  `0.022743702753899475` / `0.27994066510403215`. R59 therefore preserves
  R58-level crop ARAND while improving crop VOI, but it is not yet selected.
- R60 crop screen is materially worse: its best shared point is zero bias,
  temperature `1.15`, threshold `0.50`, with mean
  `0.05509759264176034` / `0.44643670309410544`. Do not spend a full-block
  evaluation on R60.

The first R58 full-block MEC08 record exposes an important implementation
property rather than a final negative result. Per-chunk 1/99 normalization
pushes the R58 full-block affinity p05 to `z/y/x = 0.876/0.938/0.931`, whereas
the crop-scale values are roughly `0.09/0.19/0.17`; a crop threshold of `0.50`
does not transfer directly. The first high-grid point (zero bias, temperature
`1.00`, threshold `0.25`) overmerged MEC08 with ARAND/VOI
`0.4500899672038762` / `1.677020781243909` and `n_pred=5096` versus `n_gt=62`.
It must not be read as an R58 conclusion before the complementary low grid
finishes.

The only active GPU work at this checkpoint is exactly three 8-GPU full-block
official-validation jobs, satisfying the 24-GPU cap:

- R58 high grid: UUID `4d7bf61b-0cd6-4bea-9596-5e694551d5be`, task
  `mec8nm-r58-fullval-826fa2d`, thresholds `0.25--0.65`, temperatures
  `1.00/1.10/1.15/1.20`.
- R58 low grid: UUID `335a419d-5858-410e-b1d2-db1f3589a7e2`, task
  `mec8nm-r58-lowval-5ce23587`, thresholds `0.02--0.24`, the same
  temperature grid, with affinity quantile recording enabled.
- R59 full grid: UUID `f2a1acaf-329c-4858-a163-1e758a0a20d4`, task
  `mec8nm-r59-fullval-c4973ffd`, thresholds `0.02--0.65`, the same
  temperature grid.

All three use MEC08 and MEC10 only, full `32x512x512` blocks with
`8x64x64` halos and waterz. Once their records are fetched, compare only
paired MEC08/MEC10 configurations by mean ARAND and mean VOI, select one
checkpoint/configuration, then perform at most one locked MEC11/12 audit.
No MEC11/12 test job is permitted before that validation selection.

The full-block selection protocol is fixed before seeing the final grids:

1. Keep only `region=full`, `backend=waterz` rows that contain both MEC08 and
   MEC10 under the same checkpoint, affinity variant, temperature, and
   threshold. Per-block threshold selection is invalid.
2. Report mean ARAND, mean VOI, and the balanced geometric value
   `sqrt(mean_ARAND * mean_VOI)` for every paired configuration. Remove any
   configuration that is Pareto-dominated in both mean ARAND and mean VOI.
3. Select the non-dominated configuration with the lowest balanced geometric
   value; an exact tie is broken by the lower worst-block geometric value, then
   lower mean `inference_sec + postprocess_sec`. This preserves quality as the
   primary criterion while keeping robustness and 2PB throughput explicit.

### Normalization-Scope Control Prepared, Not Yet Evaluated

`scripts/evaluate_cremi_segmentation.py` and
`scripts/evaluate_cremi_blockwise_scale.py` now expose
`--inference-normalization {volume,window}`. `volume` remains the default and
is byte-for-byte the behavior used by the active R58/R59 jobs: each blockwise
extended chunk is normalized before its model windows are evaluated. `window`
normalizes each `32x160x160` model window instead, matching the per-crop scope
used during training. The launcher forwards `INFERENCE_NORMALIZATION`, and the
value is written into both throughput and metric records so configurations
cannot be pooled accidentally.

A CPU dummy-model smoke test passed for both modes (correct shape, finite
outputs, nonzero mean absolute difference `0.7043293118476868` on deliberately
nonstationary raw input); Python compilation, CLI help, shell syntax, and
bundle root/external checksums also passed. This is only a prepared control.
Do not claim or select it until a separate paired full-block MEC08/10 run is
authorized after the active `volume` grids finish.

## 2026-07-11 Full-Block Contract Pivot

The initial `volume`-normalization full-block wave was stopped deliberately,
after preserving its remote stdout under
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/`. This is
an efficiency decision based on partial MEC08 evidence, not a paired official
MEC08/10 result and not a deployment conclusion:

- R58 high `volume` grid, UUID `4d7bf61b-0cd6-4bea-9596-5e694551d5be`, was
  stopped after the high/low boundary was already continuous and severely
  split: `t=1.00` around threshold `0.24/0.25` was about
  ARAND/VOI `0.45009/1.67833`; `t=1.10` remained near
  `0.44478/1.62795` at `0.25`; `t=1.15` began at
  `0.44163/1.59254` at `0.25`.
- R58 low `volume` grid, UUID `335a419d-5858-410e-b1d2-db1f3589a7e2`, and
  R59 `volume` grid, UUID `f2a1acaf-329c-4858-a163-1e758a0a20d4`, were then
  stopped after their MEC08 partial records showed the same `~0.445` ARAND,
  `~1.64--1.69` VOI, and thousands of fragments across the tested low range.
  Their saved logs are `mec8nm_r58_low_volume_stopped_20260711T0645Z.log` and
  `mec8nm_r59_volume_stopped_20260711T0645Z.log`.

The active controlled replacement is R58 only, still MEC08/10 only and with
the same checkpoint/chunk/halo/waterz settings. It changes normalization scope
to `window`, uses the crop-screen-leading zero bias and temperature `1.15`,
and splits the threshold range into non-overlapping jobs:

- low/mid grid UUID `448f6438-6394-4d2b-b08c-51ca47c8c213`, task
  `mec8nm-r58-winval-396ccd64`, run
  `eval_mec8nm_r58_seed1789_fullblock_val_window_t115_grid`, thresholds
  `0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65`.
- high extension UUID `d92eab49-b3ed-4987-abd2-776cd1058c76`, task prefix
  `mec8nm-r58-winhigh`, run
  `eval_mec8nm_r58_seed1789_fullblock_val_window_t115_highgrid`, thresholds
  `0.70--0.98` in `0.02` steps. It reuses the verified R58 window bundle and
  was `Pending` when recorded.

The first R58 window throughput record confirms the new contract was actually
applied (`inference_normalization=window`). Its affinity quantiles remain
high (`p05 z/y/x = 0.8396/0.9111/0.9022`), close to old volume-mode values,
and the first full MEC08 point (`threshold=0.12`) is still poor:
ARAND/VOI `0.44129475/1.64796710`, `n_pred=10257`. This means normalization
scope alone is not yet a demonstrated remedy. The high extension is justified
because at the old upper bound `0.65` hundreds of fragments remained; do not
submit R59 window work until the combined R58 window threshold curve shows
whether the missing high-threshold regime can repair the topology.

Current active/pending allocation is two R58 window 8-GPU packs (16 GPUs),
below the 24-GPU cap. The next decision remains official-validation-led: merge
the two R58 window ranges only after both MEC08 and MEC10 are available, then
apply the fixed paired selection protocol. MEC11/12 remain frozen.

### Seed-NMS Postprocess Control

Inspection of `dbmim/postprocess.py` found that the historical
`maxima_distance` mahotas path accepted `seed_distance` but did not use it to
suppress nearby regional maxima; it only affected the no-seed grid fallback.
This is consistent with the approximately 200k initial waterz fragment nodes
seen in the running full MEC08 logs. An explicit, default-off
`seed_nms_radius` was added:

- `0` preserves old regional-maxima behavior exactly.
- positive radii select spatially separated regional maxima with deterministic
  distance-priority and spatial hashing before watershed.
- the radius is part of fragment cache keys, CLI output records, and blockwise
  summary grouping, so legacy and NMS metrics cannot be mixed.

Validation passed: Python compilation, launcher syntax, CLI exposure, and
diff checks; a 144-peak synthetic map reduces to 27 seeds at radius 8; a
mahotas fragment smoke test reduces `1258` labels to `126` at radius 10 while
preserving valid output shape. The root bundle is intentionally external-code
based (`external/dbMiM`); the updated postprocess and evaluators were checksum
verified there before packaging.

The third active official-validation task is now UUID
`a10cc330-aa5a-4579-85ae-2ad97dc7d592`
(`mec8nm-r58-winnms10-54d0b780`), run
`eval_mec8nm_r58_seed1789_fullblock_val_window_t115_nms10_grid`. It holds all
R58 settings fixed, uses `inference_normalization=window`, and changes only
`seed_nms_radius=10`; its shared MEC08/10 threshold grid is
`0.02,0.05,0.08,0.12,0.16,0.20,0.24,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95`.

There are now exactly three running 8-GPU R58 window jobs (24 GPUs): low/mid
legacy-seed, high legacy-seed, and NMS=10. Do not submit any further GPU work
until they finish. R59 window work remains deferred. All three still use only
MEC08/10; MEC11/12 remain frozen.

### Prepared R61 Topology-Mix Training Control

A bounded label audit of 16 random `32x160x160` crops per train block found
17/128 (`13.28%`) crops meeting a MEC08/10-like simple-topology screen
(`foreground >= 0.95`, `boundary <= 0.05`, `unique <= 5`). These candidates
are concentrated in MEC01 (2/16), MEC07 (10/16), and MEC09 (5/16); MEC02--06
had none in this bounded sample. This explains both the source/target topology
gap and why R60's whole-block restriction is not equivalent to controlled
simple-crop sampling.

`EMVolumeDataset` now has default-off `simple_sampling_*` controls. With a
positive probability, it evaluates a bounded number of label-aligned candidate
crops from the same source block and chooses the closest high-foreground,
low-boundary, low-instance candidate; all other draws retain the old random or
boundary-sampling behavior. It is default `0.0`, preserving R58 and all prior
configs without consuming an extra RNG draw. On MEC07, 8 random smoke crops
averaged foreground/boundary/unique `0.95145/0.09305/9.75` with 4/8 eligible;
8 forced-simple crops were `1.0/0.0/1.0` with 8/8 eligible.

Prepared but not submitted config:
`configs/finetune_mec8nm_r61_seed1981_topology_mix_stage1.yaml`. It is a
strict R58 control on all MEC01--07 plus MEC09 with only seed/output changes
and `simple_sampling_probability=0.50`, `attempts=4`,
`min_foreground=0.95`, `max_boundary_fraction=0.05`, and `max_unique=5`.
Config-delta, dataset construction, YAML, Python compilation, and diff checks
passed. Do not submit R61 while the active 24-GPU R58 postprocess validation
wave runs.

The four-candidate cap is intentional. Warm-cache MEC07 reads measured about
`0.02328s` per random sample, `0.04480s` for probability `0.50` with four
candidates, `0.11567s` for probability `0.50` with sixteen, and `0.19154s`
for forced sixteen-candidate selection. Four candidates preserve the intended
topology signal while avoiding a fivefold HDF5 training-input slowdown. A
32-crop MEC07 check still gave 32/32 eligible forced-simple crops versus 14/32
random crops.

### R61 Submitted After Legacy-Low Stop

The legacy low/mid R58 window grid was stopped after its complete MEC08 curve
and a substantial MEC10 prefix had remained in the same severe-splitting
regime. Its final preserved stdout is
`outputs/live_task_logs/mec8nm_r58_window_legacy_low_stopped_20260711T0730Z.log`;
the task UUID is `448f6438-6394-4d2b-b08c-51ca47c8c213`. The R58 high-threshold
legacy grid and NMS=10 grid remain running to finish their MEC08/10 evidence.

R61 was then submitted as one 8-GPU pack:

- UUID `261ae989-d95c-49f1-9bd1-9b0195d69a79`, task
  `mec8nm-r61-topomix-81ae3497`.
- Run/TOS output:
  `finetune_mec8nm_r61_seed1981_topology_mix_stage1` under
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/`.
- Runtime verified the R61 config, training IDs `[1,2,3,4,5,6,7,9]`, eight
  paths, eight visible GPUs, offline `waterz=true`, and 214 pretrained keys
  from `assets/r57/finetuned_latest.pt`.
- It performs only its standard MEC08/10 crop waterz screen after training.
  It is not a full-block candidate and does not unlock MEC11/12.

Current allocation is exactly three 8-GPU jobs (24 GPUs): R58 high legacy
thresholds, R58 NMS=10, and R61 topology-mix training. Do not submit more work
until one completes or is stopped with fresh evidence.

### Legacy-High Stopped After Paired Failure

The legacy high-threshold R58 window task UUID
`d92eab49-b3ed-4987-abd2-776cd1058c76` was stopped after its latest stdout was
saved to `outputs/live_task_logs/mec8nm_r58_window_legacy_high_stopped_20260711T0731Z.log`.
It had already produced 13 paired MEC08/MEC10 configurations (`0.70--0.94`).
Its best partial pair was threshold `0.86`: mean ARAND/VOI
`0.40993242/1.47345979`, balanced geometric value `0.77718655`. This is a
conclusive negative relative to the R57 full-block validation anchor; remaining
`0.96/.98` MEC08 values were not improving the trend.

The active allocation is now two 8-GPU jobs (16 GPUs): R58 NMS=10 full-block
MEC08/10 validation and R61 topology-mix training plus crop validation. No new
job is submitted into the released slot. MEC11/12 remain frozen.

### R58 Seed-NMS=10 Full-Block Result

R58 seed-NMS=10 completed successfully on 2026-07-11. The task was UUID
`a10cc330-aa5a-4579-85ae-2ad97dc7d592`
(`mec8nm-r58-winnms10-54d0b780`) and evaluated MEC08/MEC10 only with window
normalization, zero affinity bias, temperature `1.15`, and
`seed_nms_radius=10`. The TOS output prefix is
`tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r58_seed1789_fullblock_val_window_t115_nms10_grid`.
It uploaded 12 objects with no failed uploads. The downloaded official records
are under
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r58_winnms10_20260711/`;
the final task stdout is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r58_nms10_20260711T073830Z.log`.

All 15 thresholds have paired `region=full`, `backend=waterz` MEC08/MEC10
rows. The Pareto frontier contains only thresholds `.60` and `.80`; applying
the fixed protocol selects `.80`:

- mean ARAND/VOI: `0.41581373/1.48174353`;
- balanced geometric value: `0.78493905`;
- worst per-block geometric value: `0.83014687`;
- mean inference plus postprocess time: `86.131s`;
- MEC08: ARAND/VOI `0.44237703/1.55782009`, `n_pred=217`;
- MEC10: ARAND/VOI `0.38925043/1.40566697`, `n_pred=595`.

This is conclusively worse than the R57 official full-block anchor at
threshold `.16` (`0.25227980/1.06223582`, balanced `0.51766846`): the NMS
result is `51.6297%` higher on the primary balanced criterion. NMS did execute
(MEC08 initial region-graph nodes fell from approximately `192874` in the
legacy path to `53492`, about `3.6x` fewer), but the substantial fragment
suppression did not recover ARAND or VOI. Reject R58 NMS=10 as a candidate;
do not start a radius sweep, do not mix it with legacy-seed metrics, and do not
run MEC11/12. The NMS implementation remains default-off for future separately
justified controls.

Only R61 topology-mix training and its crop MEC08/MEC10 screen remain active
(8 GPUs). Keep the released capacity idle until that fresh crop evidence is
available; R61 still requires a successful crop screen and a new full-block
MEC08/MEC10 validation before any test audit is permitted.

### R61 Topology-Mix Crop Screen Rejected

R61 completed successfully on 2026-07-11 (UUID
`261ae989-d95c-49f1-9bd1-9b0195d69a79`, task
`mec8nm-r61-topomix-81ae3497`). Training reached step `12000` in about
`1132s`; the model and checkpoints are retained under
`tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r61_seed1981_topology_mix_stage1/train/`.
The task's 19 TOS objects include the validation records and no MEC11/12 crop
output. Final stdout is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r61_topology_mix_20260711T074620Z.log`.

The standard volume-normalized MEC08/MEC10 crop screen has 40 paired waterz
configurations. Its sole Pareto winner is the fixed bias
`(-0.20,-0.45,-0.45)`, temperature `1.15`, threshold `.50`:

- mean ARAND/VOI: `0.037270746433/0.373094946903`;
- balanced geometric value: `0.117921699281`;
- MEC08: `0.042754743742/0.409151788696`;
- MEC10: `0.031786749123/0.337038105111`.

The independently downloaded crop CSV is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r61_topology_mix_20260711/cremi_segmentation_metrics.csv`.
Using the same CSV grouping code, R58's raw crop CSV at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r58_balanced_crop_20260711/cremi_segmentation_metrics.csv`
selects zero bias, temperature `1.15`, `.50` with
`0.022546003667/0.311183450751` and balanced `0.083761227436`. R61 is
strictly dominated by R58 on both crop metrics: ARAND is `65.3098%` higher,
VOI is `19.8955%` higher, and balanced is `40.7832%` higher. Reject R61 before
full-block validation; do not submit an R61 MEC08/MEC10 blockwise job and do
not run MEC11/12.

The R61 val-only run exposed a benign but misleading uploader message: the
stage launcher always attempted to upload a nonexistent test-crop directory.
`scripts/run_mec8nm_stage1_siflow.sh` now loops over requested evaluation
splits and uploads only directories that exist. Bash syntax and val-only/missing
directory smoke checks passed. This changes no training or evaluation result.

All MEC GPU work is now complete and the SiFlow running-task count is zero.
Keep the GPU allocation idle. The current evidence has eliminated R58 legacy
postprocess, R58 NMS=10, R59/R60, and R61 topology mix; no test or deployment
candidate exists yet.

### R62 Strict-Accept Topology Control Running

Post-R61 CPU sampler audit isolated a defect in the original topology-mix
policy. With R61's `p=.50`, four-candidate policy, 62/128 draws entered the
simple branch but only 16/62 (`25.8%`) actually met
`fg>=.95`, `boundary<=.05`, `unique<=5`; the other 46 forced a merely
"closest" lower-complexity crop. MEC02 and MEC04--06 had zero accepted simple
crops in this audit. This explains why R61 reduced global boundary/instance
complexity without reliably producing the desired validation-like topology.

`EMVolumeDataset` now has default-off
`simple_sampling_require_eligible`. When true, only an actually eligible
candidate is selected; otherwise the dataset falls back to the ordinary
boundary/random path. Default false preserves all historical configs. A
read-only simulation of the new `.50` policy had 62 attempted branches,
16 accepted simple crops (`12.5%` of all draws), and 28/128 final eligible
crops. Its raw foreground/boundary/unique means were
`.8879/.2192/21.03`, substantially closer to random R58
(`.8763/.2452/24.02`) than R61's forced-closest policy
(`.9038/.1983/19.45`) while retaining more eligible crops than random
(`28` versus `19`). Config delta, true/false branch smoke, real HDF5 fallback,
dataset construction, Python compilation, and launcher syntax all passed.

R62 is the single controlled follow-up, not a test or full-block run:

- UUID `e0dead92-7a8d-466e-90bc-b42c4822efa5`, task
  `mec8nm-r62-topoaccept-8a44b36c`;
- config `configs/finetune_mec8nm_r62_seed1993_topology_accept_stage1.yaml`;
- R61 control values retained: all eight train blocks, `p=.50`, four
  candidates, same thresholds/loss/pretrained R57 asset; only seed, output,
  and `simple_sampling_require_eligible=true` differ;
- exactly 8 GPUs on Shanghai `med-model`, with `EVAL_SPLITS=val` and
  `EVAL_BACKENDS=waterz`; the output prefix is
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r62_seed1993_topology_accept_stage1`.

Startup verified the runtime config, train IDs `[1,2,3,4,5,6,7,9]`, eight
paths, eight GPUs, and waterz availability; training advanced through step 420
without an exception. No other job may be submitted. R62 must first beat R58
on paired crop MEC08/MEC10 metrics before a new full-block validation can be
considered; MEC11/12 remain frozen.

### R62 Strict-Accept Crop Result Rejected

R62 completed successfully: UUID `e0dead92-7a8d-466e-90bc-b42c4822efa5`, task
`mec8nm-r62-topoaccept-8a44b36c`, step `12000` in `1133.85s`. Its TOS prefix
is
`tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r62_seed1993_topology_accept_stage1`;
all 19 standard train/val objects uploaded, with no test-crop output. Final
task log:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r62_topology_accept_20260711T082534Z.log`.
The independently downloaded metrics, records, summary, and affinity stats are
under
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r62_topology_accept_20260711/`.

All 40 waterz configurations have paired MEC08/MEC10 crop rows. The
non-duplicate Pareto choices are zero-bias temperature `1.15/.50` and
zero-bias temperature `1.00/.50`; the fixed balanced protocol selects
temperature `1.15`, threshold `.50`:

- mean ARAND/VOI: `0.029136764797/0.352282465789`;
- balanced geometric value: `0.101313233823`;
- MEC08: `0.052355928557/0.497306034002`;
- MEC10: `0.005917601036/0.207258897576`.

The strict-accept correction improved on R61 by `21.8240%` ARAND,
`5.5783%` VOI, and `14.0843%` balanced, but it remains strictly dominated by
the same-code R58 crop winner (`0.022546003667/0.311183450751`, balanced
`0.083761227436`). R62 is still `29.2325%` worse in ARAND, `13.2073%` worse
in VOI, and `20.9548%` worse in balanced. The gain is highly asymmetric:
MEC10 improves sharply, while MEC08 degrades severely relative to R58. The
affinity diagnostic agrees with this warning but is not causal proof: at the
selected calibration R62 MEC08 p05 `z/y/x` is
`.063719/.155971/.135943` with Dice/IoU `.96616346/.93454183`, below R58's
`.119049/.219758/.197723` and `.97928067/.95940243`.

Reject R62 before full-block validation. Retain
`simple_sampling_require_eligible` as a default-off correctness control, but
do not treat it as a demonstrated quality improvement or submit a probability
or attempt-count sweep. All MEC tasks are terminal (`Running=0`, `Queueing=0`);
keep capacity idle. MEC11/12, deployment, and 2PB inference remain frozen
because no candidate has cleared paired crop selection followed by full-block
MEC08/MEC10 validation.

### R59 Global-Normalization Full-Block Validation Rejected

The initial broad global-normalization task UUID
`dfe11836-34ba-4e3e-bf96-c8dcd3d609f2` was intentionally stopped because the
six calibration variants, 27 thresholds, and three regions were not a
proportionate diagnostic. Its retained partial log is
`outputs/live_task_logs/mec8nm_r59_globalnorm_stopped_20260711T0852Z.log`;
it never became a selection artifact and did not read MEC11/MEC12.

The lean formal replacement UUID `a972c171-2b37-4283-a15c-b370fa956753`
(`mec8nm-r59-globalhigh-4bb3d3b9`) succeeded in `33m32s`. It evaluated only
MEC08/MEC10 using R59 DPP, `global` normalization, chunk `(32,512,512)`, halo
`(8,64,64)`, waterz, `min_boundary=4`, zero affinity bias, temperatures
`1.00/1.15/1.20`, thresholds `.55--.95`, and `region=full`. All result objects
were uploaded successfully to
`tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r59_seed1789_fullblock_val_global_highfull`.
The independently downloaded result is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r59_globalhigh_20260711/`,
and its final task log is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r59_globalhigh_20260711T093001Z.log`.

The official record has 54 rows, exactly 27 paired MEC08/MEC10 settings; every
row is `region=full`, `backend=waterz`, `inference_normalization=global`, and
`min_boundary=4`. The complete Pareto set is `t=1.20` at `.85/.90/.95`.
Fixed balanced selection uses `t=1.20, threshold=.85`:

- mean ARAND/VOI: `0.392960605259/1.382466356900`;
- balanced geometric value: `0.737058217753`;
- worst-block geometric value: `0.846541532551`;
- mean inference plus postprocess: `66.581s`;
- MEC08: `0.441996426221/1.621353757227`, `n_pred=219`;
- MEC10: `0.343924784297/1.143578956573`, `n_pred=577`.

The R57 full-block anchor is ARAND/VOI
`0.252279795600/1.062235821000`, balanced `0.517668461277`; the global result
is worse by `55.7638%`, `30.1468%`, and `42.3804%`, respectively. Reject R59:
global normalization does not repair the full-block failure. MEC11/MEC12,
deployment, and 2PB inference remain frozen.

R63 is the next controlled training follow-up. It holds the R59 architecture,
all-eight-block training set, loss, optimizer, and seed fixed, and changes only
photometric augmentation: intensity `p=.75`, gain `.55--1.35`, bias
`-.18--.18`, plus `p=.50` 3x3 XY mean blur. This directly tests the measured
MEC08 photometric shift. It is crop-screen-only until its paired MEC08/MEC10
metrics justify another full-block validation.

### R63 Photometric-Domain Run Active

The initial R63 task UUID `b4115cf1-ef43-4659-8616-d5853838b5b3`
(`mec8nm-r63-photodomain-6b28e34f`) failed before training or data download:
the R57-derived stage bundle omitted the root `third_party/waterz` source
needed by the val-only waterz preflight. The retained terminal log is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r63_photodomain_waterz_bundle_failure_20260711T094120Z.log`.
It is a packaging failure, not a model result, and did not read any MEC block.

The repaired bundle adds the same root WaterZ source used by the successful
R59 full-block bundle. Static preflight found `third_party/waterz` and the
bundled Boost headers, and `setup.py --name` returned `waterz`. The corrected
run UUID `2bb42bb7-cbbf-47a3-b280-a0422c5e8682`
(`mec8nm-r63-photo-wzfix-b9c874d0`) is active on one 8-GPU Shanghai changliu
`med-model` pack with the unchanged R63 configuration and output prefix.

Startup evidence is complete: the pod built/imported WaterZ `0.8`, reported
`cuda=true` and `gpu_count=8`, rewrote the runtime config with exactly train
IDs `[1,2,3,4,5,6,7,9]`, downloaded eight train paths, and loaded 214 R57
pretrained keys. The distributed trainer reports `world_size=8`, dataset size
32768, and had advanced through step 220 without exception. The task command
sets `EVAL_SPLITS=val` and `EVAL_BACKENDS=waterz`; after training it may only
screen MEC08/MEC10 crops. MEC11/MEC12, deployment, and 2PB inference remain
frozen.

### R63 Crop Recheck Rejected

R63 training completed in corrected UUID `2bb42bb7-cbbf-47a3-b280-a0422c5e8682`
at step 12000 (`1256s`), but its original embedded crop output is invalid.
The bundle combined the current evaluator, which calls the watershed helper
with `seed_nms_radius=0`, with an older staged `postprocess.py` that lacked
that argument. All 80 waterz rows failed, while the old stage launcher did not
pass `--fail-on-backend-error` and therefore uploaded empty records. Those
artifacts are excluded from every comparison.

Recovery reused the completed R63 final checkpoint only. Recheck UUID
`361a2b49-7337-45b9-ae05-d19649168bc3` failed before evaluation because the
crop CLI has no `--seed-nms-radius` option; its log is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r63_crop_recheck_cli_failure_20260711T101642Z.log`.
Corrected recheck UUID `f69d5bad-4d2f-4ddf-87b7-dcf731489e5e` succeeded in
`309s` with current `postprocess.py`, `--fail-on-backend-error`, fixed
MEC08/MEC10 samples, and an in-task assertion of exactly 80 waterz rows. Its
complete log and independently downloaded evidence are
`/volume/med-train/users/dchen02/code/dbMiM/outputs/live_task_logs/r63_photodomain_recheck_20260711T102507Z.log`
and
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r63_photodomain_recheck_20260711/`.

The valid summary has 80 records, zero failures, waterz available, and only
MEC08/MEC10. After removing 16 duplicate `pred`/zero-bias rows, 32 paired
calibrations remain. The two-point Pareto frontier selects zero bias,
`t=1.15`, `.50`:

- mean ARAND/VOI: `0.024314198449/0.293735246416`;
- balanced geometric value: `0.084509982090`;
- worst-block geometric value: `0.129345245600`;
- MEC08: `0.042331074359/0.395222488738`, `n_pred=357`;
- MEC10: `0.006297322539/0.192248004094`, `n_pred=151`.

The R58 crop winner remains zero bias, `t=1.15`, `.50`, with
`0.022546003667/0.311183450751` and balanced `0.083761227436`. R63 improves
mean VOI `5.6070%`, but ARAND is `7.8426%` higher, primary balance is
`0.8939%` higher, and worst-block balance is `23.1032%` higher. Its MEC10
gain masks an unacceptable MEC08 regression: `+183.1844%` ARAND and
`+54.3067%` VOI. Reject R63 before any full-block work; MEC11/MEC12,
deployment, and 2PB inference remain frozen.

The runner now has two protections: the stage1 launcher passes
`--fail-on-backend-error`, and `scripts/run_mec8nm_crop_eval_siflow.sh`
performs checkpoint-only MEC08/MEC10 evaluation, asserts 80 waterz metric
rows, and has no test-split path. Future crop screens must retain these
guards.

### R64 Bright-Gamma Follow-Up Prepared

R63's symmetric gain/bias and XY blur shifted the normalized train distribution
away from MEC08 despite reducing some gradients. The 128-crop audit at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_aug_distribution_audit_20260711T1028Z.txt`
measures MEC08 mean/std/p05/p50 `.618153/.263203/.127728/.681904` and
z/y/x gradient `.199492/.075573/.084305`. R58 default augmentation has
first-ten-statistic L2 distance `0.038494` to this target; R63 strong
augmentation rises to `0.064802` and increases one-saturation to `.043720`
versus MEC08 `.011196`.

The deterministic gamma simulation at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_gamma_candidate_audit_20260711T1030Z.txt`
selects a single targeted augmentation: probability `.90`, exponent
`.75--.85`. Keeping R59's ordinary intensity/noise policy, it reduces the
MEC08 L2 to `0.010121` and gives mean/std/p05/p50
`.612423/.268897/.129770/.661763`. This raises low/mid tones without the
strong symmetric gain/bias/blur that damaged R63.

`augment_gamma_probability` is now a default-`.35`, validated config
parameter. A 40-seed test proves default augmentation is bitwise unchanged;
the R64 real-HDF5 smoke passes with `.90`, `(.75,.85)`, image
`(1,32,160,160)`, and label `(32,160,160)`. New config
`configs/finetune_mec8nm_r64_seed1789_brightgamma_stage1.yaml` is R59 plus
only the output path and these two gamma fields. R64 remains a val-crop-only
controlled hypothesis; no test or full-block task is justified before its
paired MEC08/MEC10 crop result.

### R64 Bright-Gamma Crop Result Rejected

R64 SiFlow UUID `f4e357c7-a607-4404-94ee-77ebd403dcd5`
(`mec8nm-r64-brightgamma-13903c64`) completed successfully in `27m05s` after
12000 steps on the fixed eight MEC01--07/MEC09 training paths and one
eight-GPU `med-model` pack. Its WaterZ crop evaluation was validation-only:
the downloaded summary has zero failures, the CSV has exactly 80 MEC08/MEC10
`waterz` rows, and task logs contain no MEC11/MEC12 path. Evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r64_brightgamma_20260711/`.

After removing the 16 duplicate `pred`/zero-bias rows, 32 paired calibration
settings remain. The only Pareto point is bias `(-.20,-.45,-.45)`,
temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.021321377783/0.267121674862`, balanced `0.075467888162`, and worst-block
balance `0.108898647791`. MEC08 is
`0.035600267251/0.333113102975` with Dice/IoU
`.976117487142/.953349295436` and `n_pred=216`; MEC10 is
`0.007042488314/0.201130246749` with Dice/IoU
`.989626908612/.979466612926` and `n_pred=101`.

Versus the fixed R58 crop winner, R64 improves mean ARAND `5.4317%`, mean VOI
`14.1594%`, and primary balance `9.9012%`, but that average conceals an
unacceptable MEC08 failure: ARAND is `138.1569%` higher, VOI is `30.0573%`
higher, and worst-block balance is `3.6434%` higher. The strong MEC10 gain
does not compensate. Reject R64 before full-block validation. MEC11/MEC12,
deployment, and 2PB inference remain frozen; no new training run should be
submitted without a separately audited MEC08-preserving hypothesis.

### R64 Intermediate-Bias Crop Recheck Rejected

The default R64 grid showed that zero bias at `t=1.15,.50` preserves M08 but
breaks M10, while `(-.20,-.45,-.45)` produces the opposite failure. To test
whether that was merely a coarse calibration grid, the validation-only crop
runner was generalized to accept environment-provided fixed grids and derive
the expected metric row count. The standard grid still asserts 80 rows; this
five-bias, one-temperature scan correctly asserts 96 rows.

Checkpoint-only SiFlow UUID `5f7a7ad8-bb8d-49ef-81cd-3b4dbc58dd99`
(`mec8nm-r64-biasmid-dcec396f`) completed in `6m36s` without training. It
evaluated only M08/M10, fixed `t=1.15`, the usual eight thresholds, and five
global z/y/x biases linearly spanning zero to `(-.20,-.45,-.45)`. WaterZ was
available, the in-task assertion reported 96 rows, failures were zero, and no
MEC11/MEC12 path occurred. Evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r64_intermediate_bias_recheck_20260711/`.

The 96 rows yield 48 paired settings and two Pareto points. Fixed selection
still chooses the original `(-.20,-.45,-.45), t=1.15, .50` point exactly, so
the rejected M08 result is unchanged. The only new Pareto point,
`(-.05,-.1125,-.1125), t=1.15, .50`, has mean ARAND/VOI
`0.020017610135/0.295218562188`, balance `0.076873728169`, worst-block
balance `0.112877655965`, and MEC08 `0.032764371052/0.388878675428`.
It is inferior on the primary balance and still violates the M08 guard. The
other intermediate biases are non-Pareto and worse. Dense fixed global
calibration therefore cannot rescue R64; reject it conclusively before
full-block validation. MEC11/MEC12, deployment, and 2PB inference remain
frozen.

### R65 Balanced-Gamma Dose Control Running

R64's full bright-gamma dose and its dense fixed-bias recheck are both
negative. The 64-crop validation-only audit at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_gamma_dose_audit_20260711T1137Z.txt`
finds that historical R59 gamma is closest to M10 and full R64 bright gamma is
closest to M08. The tested midpoint `p=.50`, exponent `.75--.85` minimizes the
maximum first-ten-statistic distance across M08/M10: `.016754/.025294`, versus
the R59 worst `.037129` and R64 worst `.035862`. This justifies a single dose
control, but does not itself establish segmentation quality.

New config `configs/finetune_mec8nm_r65_seed1789_balancedgamma_stage1.yaml`
holds R64 fixed except for output path and gamma probability `.50`; the gamma
range, seed, model, loss, optimizer, train blocks, ordinary intensity/noise
policy, and crop shape do not change. YAML delta, HDF5 smoke, stage bundle
syntax, Python compilation, WaterZ setup, and staged-file hashes passed.

The only active SiFlow task is UUID `b449842a-53b8-4781-93cb-797a65dd94a1`
(`mec8nm-r65-gammamid-b5ff05e0`) on one 8-GPU
`cn-shanghai/changliu/med-model` pack. Startup confirms WaterZ, CUDA 8 GPUs,
the eight fixed training IDs, 214 loaded R57 keys, and val-only WaterZ crop
evaluation. It may only screen M08/M10. Do not submit another task or access
M11/M12, deployment, or 2PB inference until the same paired crop gate is
applied to R65.

### R65 Crop Gate and Full-Block Validation Rejected

R65 training UUID `b449842a-53b8-4781-93cb-797a65dd94a1`
(`mec8nm-r65-gammamid-b5ff05e0`) completed in `26m46s` with WaterZ, 8 GPUs,
80 valid M08/M10 rows, zero failures, and no test path. Its only crop Pareto
point is `(-.20,-.45,-.45), t=1.00, .50`: mean ARAND/VOI
`0.014667305322/0.278149880175`, balance `0.063872601465`, worst-block
balance `0.088341945947`. MEC08 is
`0.021810413186/0.357824464262` and MEC10 is
`0.007524197458/0.198475296088`. Relative to R58, crop mean ARAND/VOI improve
`34.9450%/10.6155%`, primary balance improves `23.7444%`, and worst-block
balance improves `15.9213%`; M08 ARAND/VOI still regress `45.9062%/39.7053%`.

Checkpoint-only UUID `a7fe194b-2799-49f4-b056-8caec109bfb2`
(`mec8nm-r65-biasmid-3b192eed`) completed in `6m44s` with fixed `t=1.00`, five
global biases, eight thresholds, 96 valid M08/M10 rows, zero failures, and no
test path. Removing 16 exact `pred`/zero-bias rows leaves 40 paired settings;
its only Pareto point remains the same R65 crop selection. Intermediate global
bias cannot resolve the M08 tradeoff. Crop evidence is under
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r65_balancedgamma_20260711/`
and
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r65_intermediate_bias_recheck_20260711/`.

The blockwise runner now accepts a named checkpoint TOS URI, while new
`scripts/run_mec8nm_blockwise_val_checkpoint_siflow.sh` forcibly exports M08/M10
only. This enabled official full-block UUID
`6ed95f00-32ff-48de-a2ea-c32cec72d5ea`
(`mec8nm-r65-fullval-1798381e`) to evaluate only M08/M10 with global
normalization, chunk `(32,512,512)`, halo `(8,64,64)`, fixed bias
`(-.20,-.45,-.45)`, `t=1.00`, `min_boundary=4`, and thresholds `.35--.95`.
It succeeded in `16m33s` with exactly 26 zero-failure full-region waterz rows.
The sole full-block Pareto point is threshold `.90`: mean ARAND/VOI
`0.401543070259/1.443839745931`, balance `0.761422251147`, worst block
`0.849147790727`; MEC08 is `0.446298691940/1.615626448203` and MEC10 is
`0.356787448577/1.272053043658`. It is `59.1658%` worse in ARAND, `35.9246%`
worse in VOI, and `47.0869%` worse in primary balance than the R57 official
full-block anchor. Reject R65 before M11/M12, deployment, or 2PB inference.
The next hypothesis must address crop-to-full-block normalization/topology
mismatch, not gamma dose. Full-block evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r65_fullblock_global_20260711/`.

The narrow normalization follow-up then isolated the remaining question without
retraining. Checkpoint-only UUID `ab25730e-b922-4e22-a1f9-bce15ba46209`
(`mec8nm-r65-fullvol-de7e19fb`) completed successfully on one 8-GPU pack. Its
wrapper forcibly supplied only M08/M10 and ran WaterZ over full blocks with
`volume` normalization, chunk `(32,512,512)`, halo `(8,64,64)`, fixed bias
`(-.20,-.45,-.45)`, `t=1.00`, `min_boundary=4`, and 11 thresholds
`.12,.16,.18,.20,.22,.24,.30,.40,.50,.60,.65`. The final evidence has exactly
22 rows, zero failure-ledger entries, and no M11/M12 path.

Its sole Pareto point is `.65`: mean ARAND/VOI
`0.395251787736/1.420713606624`, balance `0.749359455054`, and worst block
`0.859067495726`. MEC08 is `0.448038919944/1.647171549973` (`n_pred=784`),
MEC10 is `0.342464655527/1.194255663275` (`n_pred=1502`). Relative to R65
global, this is only `1.5668%` better in ARAND, `1.6017%` better in VOI, and
`1.5842%` better in balance; relative to the official R57 full-block anchor it
is `56.6720%/33.7475%/44.7566%` worse. The R65 volume mechanism is therefore
formally rejected. Do not evaluate M11/M12, deploy, or launch 2PB inference
from it. Full evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r65_fullblock_volume_20260711/`.

### R57/R59 Member-Calibrated Logit Ensemble Rejected

`scripts/evaluate_cremi_segmentation.py` and the blockwise evaluator now
support a strict weighted logit ensemble. Each member applies its own learned
postprocessor, fixed member bias, and fixed member temperature before mixing;
the outer WaterZ calibration is neutral. R57 and R59 have matching 214-key
`unetr_aniso_em` main models and volume size `(32,160,160)`; R59 alone carries
four postprocessor keys. Stub arithmetic, real checkpoint construction, staged
compilation, shell syntax, and R57 asset-byte identity passed.

Smoke UUID `c06b4e23-d1fe-4e95-86f4-4ba3c4e75c5d` verified the remote path on
center M08/M10 crops: R57/R59 `.85/.15`, member biases
`(-.20,-.45,-.45)/(0,0,0)`, member temperatures `1.20/1.20`, neutral outer
calibration, WaterZ, CUDA 8, six expected records, and zero failures. It is an
integration result only, not a model-selection result.

Formal full-block UUID `ee52f7b1-5348-408f-82d7-61bc6e7d9372` held the same
ensemble fixed and forcibly used M08/M10 only. It completed with WaterZ,
`volume`, chunks `(32,512,512)`, halos `(8,64,64)`, `min_boundary=4`, 11
predeclared thresholds `.10--.70`, exactly 22 records, and zero failures. Its
only Pareto threshold `.70` has mean ARAND/VOI `0.394638978896/1.731286401767`,
balance `0.826579153965`, worst block `0.916424680291`; M08 is
`0.456917433662/1.838043665601`, M10 is `0.332360524129/1.624529137932`.
Versus the R57 official full-block anchor it is `56.4291%` worse in ARAND,
`62.9851%` worse in VOI, and `59.6735%` worse in balance. It is also worse in
VOI/balance than both R59 global and R65 volume. Reject this R57/R59
reweighting family. Do not run additional weights, M11/M12, deployment, or 2PB
inference. Evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r57r59_ensemble_smoke_a085_20260711/`
and
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r57r59_ensemble_fullblock_a085_20260711/`.

### R66 LSD Auxiliary-Head Crop Gate Rejected

R66 is a topology-aware auxiliary supervision test, not a new split or
post-processing family. It retains the R57 seed, public R57 initialization,
MEC01--07/MEC09 training IDs, crop shape, affinity objective, optimizer, and
ordinary augmentation. The only model/loss delta expands `out_channels` from
3 to 7 and adds four local-shape-descriptor channels: foreground plus three
foreground-masked centroid offsets. `lsd_weight=.15`,
`lsd_fg_weight=.25`, and `lsd_offset_weight=1.0` keep this auxiliary objective
modest. The pretrained loader exactly copies the three R57 affinity bias/scale
channels and independently initializes the descriptor branch; static model,
HDF5 target/loss, staged-bundle, and val-only entrypoint checks passed.

SiFlow UUID `4f60e8f4-eda4-45b0-a21d-81fbd981ff68`
(`mec8nm-r66-lsd-e9960b6c`) ran one 8-GPU Shanghai `med-model` pack and
finished 5000 steps successfully. Runtime evidence confirms CUDA 8, WaterZ,
214 loaded R57 keys, the eight fixed train IDs, and the declared LSD loss.
The remote crop evaluator was hard-locked to M08/M10 and WaterZ. It emitted
exactly 80 rows, finite ARAND/VOI, and an empty failure list; no full-block or
frozen-split path was executed.

After collapsing the 16 exact `pred` / zero-bias, temperature-1.0 duplicates,
32 paired calibration settings remain. R66 has two Pareto points, neither of
which clears the R58 crop guard. Its best balance is bias
`(-.20,-.45,-.45)`, temperature `1.15`, threshold `.22`: mean ARAND/VOI
`0.049656091996/0.411782761303`, balance `0.142994834444`, worst-block balance
`0.173345275549`; MEC08 is `0.032558920140/0.373421472399` (`n_pred=809`) and
MEC10 is `0.066753263853/0.450144050207` (`n_pred=193`). The R58 anchor is
`0.022546003667/0.311183450751`, balance `0.083761227436`, with MEC08
`0.014948239287/0.256127966754`. R66 is `120.2434%/32.3280%/70.7172%` worse
in mean ARAND/VOI/balance and regresses M08 `117.8111%/45.7949%`. Reject R66
before full-block validation. Do not fetch its checkpoint for further testing,
run M11/M12, deploy, or schedule 2PB inference. The detailed local report is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r66_lsd_crop_report_20260711.md`.

### R67 Soft-Negative R58 Control Running

The boundary audit at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_boundary_widening_audit_20260711.json`
samples 16 deterministic allowed-block crops per volume. Historical widened
training targets have affinity-positive fractions `.788879/.852044/.853948`
(z/y/x), whereas M08/M10 raw validation labels are
`.960546/.969814/.968969`. R58--R65 use `weighted_mse`, `weight_alpha=1.0`,
which makes aggregate negative and positive loss mass equal; its per-negative
weights are about `3.74/5.76/5.85`. Existing full-block failures are dominated
by fragmentation, so more boundary pressure is not justified.

R67 is a one-variable R58 control: exactly the R58 configuration except
`weight_alpha: 1.0 -> 0.5`. Positive affinity weight stays one, while aggregate
negative loss mass becomes half the positive mass. No split, model, data,
augmentation, calibration grid, or post-processing change is included.

SiFlow UUID `94db3a0b-6084-4085-a6ad-1c05f3df07f0`
(`mec8nm-r67-softneg-4f000dc6`) is the sole active 8-GPU Shanghai `med-model`
task. Its entrypoint forces MEC01--07/MEC09 training, M08/M10 validation,
WaterZ, and 80 expected metric rows; it rejects any test split. Startup has
already proven CUDA 8, WaterZ, 214 loaded R57 keys, all eight train paths, and
remote `weighted_mse` with `weight_alpha=.5`. Do not perform full-block,
M11/M12, deployment, or 2PB work unless the completed paired crop result clears
the strict R58 gate. The plan is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r67_softneg_plan_20260711.md`.

### R67 Soft-Negative Crop Gate Rejected

R67 UUID `94db3a0b-6084-4085-a6ad-1c05f3df07f0`
(`mec8nm-r67-softneg-4f000dc6`) completed 12000 steps on one 8-GPU Shanghai
`med-model` pod. Runtime confirms CUDA 8, WaterZ, 214 loaded R57 keys, the
eight fixed train paths, and `weighted_mse` `weight_alpha=.5`. The task then
ran its hard-locked M08/M10 WaterZ crop evaluation successfully: 80 rows,
finite ARAND/VOI, empty failures, and no full-block or frozen-split path.

The downloaded CSV has 80 raw rows. The paired analyzer removes 16 exact
`pred` / zero-bias, temperature-1.0 duplicates, leaving 32 paired
configurations and no strict R58 gate pass. The best R67 point is bias
`(-.20,-.45,-.45)`, temperature `1.00`, threshold `.50`: mean ARAND/VOI
`0.052563021514/0.399776827409`, balance `0.144960263451`, worst-block balance
`0.210299685945`; M08 is `0.016472016743/0.300693537996` (`n_pred=154`) and
M10 is `0.088654026284/0.498860116822` (`n_pred=81`). Compared with R58,
R67 is `133.1368%/28.4698%/73.0637%` worse in mean ARAND/VOI/balance. M08
regresses `10.1937%/17.3997%`; M10 regresses `194.1040%/36.2117%`.

The soft-negative hypothesis is rejected: halving negative aggregate loss mass
reduces instance count but produces serious M10 merging and still violates the
M08 guard. Do not run R67 full-block validation, M11/M12, deployment, or 2PB
inference. Evidence is at
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r67_softneg_crop_report_20260711.md`
and
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r67_softneg_20260711/paired_analysis_vs_r58.json`.

### R68 Multiscale-Affinity Crop Gate Rejected

R68 UUID `b8939e9c-ffa2-48e5-bed9-a7b173bda89f`
(`mec8nm-r68-msaff-b43f40ff`) completed 12000 steps on one 8-GPU Shanghai
`med-model` pod. Runtime confirms CUDA 8, WaterZ, 214 inherited R57 keys, the
fixed train IDs, six affinity offsets, and `bcar_channels=3`. Its M08/M10-only
WaterZ evaluator produced exactly 80 finite metric rows without errors.

No point passes the strict R58 crop gate. R68's best balance is zero bias,
temperature `1.00`, threshold `.50`: mean ARAND/VOI
`0.029857502936/0.358199176768`, balance `0.103416309023`, M08
`0.052169534998/0.490800733464` (`n_pred=428`), and M10
`0.007545470874/0.225597620072` (`n_pred=165`). Relative to R58, mean
ARAND/VOI regress `32.4292%/15.1087%`, while M08 regresses
`249.0012%/91.6232%` despite an M10 improvement.

The uncalibrated affinity statistics show all three near-neighbor means and
fifth percentiles falling on both blocks; affinity Dice falls from
`.979281/.989033` (R58 M08/M10) to `.968275/.986410`. The two-voxel auxiliary
loss creates a global lower-affinity shift whose topology effect is
block-dependent, not an isolated axis defect. Reject R68 before full-block,
M11/M12, deployment, or 2PB inference, and do not blind-sweep its auxiliary
weight. Detailed evidence is
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r68_multiscale_affinity_crop_report_20260712.md`.

### R69 No-Widen R58 Control Queueing

R69 is a one-variable R58 control: `data.widen_border: true -> false`. It keeps
R58's seed, R57 checkpoint, MEC01--07/MEC09 paths, model, crop sampling,
augmentation, weighted-MSE `weight_alpha=1.0`, boundary Dice, BCAR, membrane
weighting, optimizer, and 12000-step schedule. Unlike R67, it retains full
negative supervision on genuine neighboring instances and removes only the
artificial one-pixel zero-label band.

Local and staged-bundle checks prove the config differs from R58 only in
`output_dir` and `data.widen_border`; the entrypoint hard locks M08/M10 `val`,
WaterZ, 80 metric rows, the eight allowed train IDs, and eight GPUs. SiFlow UUID
`ae98ae06-08b2-4fac-b5bf-936196b79b0f`
(`mec8nm-r69-nowiden-30543597`) is the sole submitted 8-GPU Shanghai
`med-model` wave. It is currently queueing because an 8-card node cannot yet be
scheduled; one medium-priority task is ahead. Do not duplicate the wave or run
full-block, M11/M12, deployment, or 2PB work unless its paired crop result
passes the strict R58 gate. Plan:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r69_nowiden_plan_20260712.md`.

### R69 No-Widen Crop Gate Rejected

R69 completed 12000 steps and its M08/M10-only WaterZ crop sweep successfully.
Runtime confirms CUDA 8, 214 R57 keys, eight allowed train paths, 32768 training
samples, and the unchanged R58 loss. The evaluator returned exactly 80 finite
rows; duplicate collapse leaves 32 paired settings and zero strict gate passes.

Best balance is bias `(-.20,-.45,-.45)`, temperature `1.00`, threshold `.50`:
mean ARAND/VOI `0.036164090089/0.396930915975`, M08
`0.045930476087/0.448332741337` (`n_pred=234`), and M10
`0.026397704092/0.345529090612` (`n_pred=106`). Against R58, mean ARAND/VOI
regress `60.4013%/27.5553%` and M08 regresses `207.2635%/75.0425%`.
Removing the artificial border band helps M10 slightly but strongly harms M08.
Reject R69 before full-block, M11/M12, deployment, or 2PB inference. Report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r69_nowiden_crop_report_20260713.md`.

### R70 FullEM-R33 Initialization Control Queueing

R70 tests broader external unlabeled EM pretraining without changing MEC
supervision. It replaces only the R58 PublicEM-R57 initialization with the
shape-compatible FullEM mixed-mask R33 CREMI checkpoint. Both checkpoints have
the same 214 model keys and tensor shapes. R33's self-supervised lineage used
CREMI raw plus the inherited approximately 485.8 GB FullEM mirror collection,
including FAFB, FIB25, Kasthuri, MitoEM, and MB-MOC assets, for 160000 mixed
edge/random masking steps. No external labels are used in R70.

All MEC settings remain R58-identical: seed 1789, MEC01--07/MEC09 supervision,
M08/M10 validation, model, augmentation, radius-1 border widening, losses,
optimizer, and 12000 steps. The bundle contains a byte-verified R33 checkpoint
and hard locks WaterZ, 80 rows, and eight GPUs. SiFlow UUID
`d49b444f-fcd6-44c3-8250-7a2eb176f799`
(`mec8nm-r70-fullem-4d74bb4`) is queueing first in Shanghai `med-model` with
zero tasks ahead. Full-block, M11/M12, deployment, and 2PB inference remain
forbidden before a strict R58 crop-gate pass. Plan:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r70_fullem_r33_plan_20260713.md`.

### R70 FullEM-R33 Crop Gate Rejected

R70 completed successfully with exactly 80 M08/M10 WaterZ rows and no errors,
but has zero strict gate passes. Its best mean ARAND/VOI is
`0.053447533042/0.487475984541`; M08 is
`0.041557721757/0.460674361248` and M10 is
`0.065337344326/0.514277607833`. Versus R58, mean ARAND/VOI regress
`137.0599%/56.6523%`, and M08 regresses `178.0108%/79.8610%`. Reject R70:
broader FullEM R33 initialization does not automatically improve MEC transfer.
Report: `/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r70_fullem_r33_crop_report_20260713.md`.

### R71 EMA Control Running

R71 returns to the strongest R58/PublicEM-R57 data and model recipe and changes
only `train.ema_decay=0.999`; saved and evaluated model weights are EMA. EMA is
disabled by default for every prior config. SiFlow UUID
`41bc1e3b-6b6e-4fdf-9de9-6af7ab993381`
(`mec8nm-r71-ema-4ad0d84a`) is running on one 8-GPU Shanghai `med-model` pod.
The entrypoint fixes MEC01--07/MEC09, M08/M10, WaterZ, and 80 rows. Require a
remote `ema_config` startup record and a strict R58 crop-gate pass before any
full-block work. Plan:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r71_ema_plan_20260713.md`.

### R71 EMA Crop Gate Rejected; Current Absolute MEC Result

R71 completed with remote EMA confirmation, 12000 steps, exactly 80 finite
M08/M10 WaterZ rows, and no errors. Its best aggregate point is uncalibrated,
temperature `1.00`, threshold `.50`: mean ARAND/VOI
`0.020969504209/0.299190914044`, improving R58 by `6.9924%/3.8538%`.
However, M08 regresses to `0.037575210047/0.433248168057`, while M10 improves
to `0.004363798372/0.165133660030`. There are zero strict gate passes. Reject
R71 before full-block, M11/M12, deployment, or 2PB inference.

The selected anchors remain R58 crop (`0.022546003667/0.311183450751`) and R57
M08/M10 full-block (`0.252279795612/1.062235821039`). Released dbMiM CREMI
A/B/C full-volume results (`R48 .178252/.962154`, `R57 .178248/.964617` in
ARAND/VOI order) are a different benchmark and cannot establish MEC SOTA. The
local MEC data card has no same-protocol published ARAND/VOI baseline. Compact
comparison:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_absolute_results_and_publication_comparison_20260713.md`.

### R72 Photometric Consistency Crop Gate Rejected

R72 UUID `ff377784-06ce-4b4e-a4f0-bb7a7b8d0a2f`
(`mec8nm-r72-consistency-90145185`) completed 12000 steps on one 8-GPU
Shanghai `med-model` pod. It held R58 fixed except for a weight-`.05`
stopped-teacher photometric affinity-consistency term. Startup verified CUDA 8,
WaterZ, 214 R57 keys, and train IDs M01--07/M09. The task returned exactly 80
finite M08/M10 WaterZ rows with no failures and did not evaluate M11/M12.

The frozen paired analyzer found zero strict gate passes. Best mean ARAND/VOI
is `0.028005094678/0.369409206168`; M08 is
`0.042332990281/0.473893471718`. These regress R58 by
`24.2131%/18.7111%` on the mean and `183.1972%/85.0222%` on M08. Reject R72
before full-block, M11/M12, deployment, or 2PB inference. Do not sweep its
consistency weight/ranges. The anchors remain R58 crop and R57 M08/M10
full-block. Versionable report:
`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/evidence/mec8nm_r72_consistency_crop_report_20260713.md`.

The MEC-only versionable workspace is now
`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/`. It contains the configs,
scripts, runtime snapshot, split/manifest contract, skill/state, compact
evidence, and workspace verifier. Large HDF5/checkpoint/bundle/log assets stay
outside that folder and are mapped in `mec8nm/ASSETS.md`.

### R73 Decoder-Aware MEC Domain Pretraining Rejected

R73 tested a new raw-only MEC adaptation family. Starting from the exact
PublicEM-R57 checkpoint, `DecoderAwareDBMIM3DMAE` adapted the shared encoder,
convolutional decoder, and affinity head for 4000 steps on M01--07/M09 raw
crops only. Its objective combined 75% masked reconstruction with a weight-`.1`
raw-gradient affinity proxy; no instance labels or M08/M10 data entered this
stage. The saved checkpoint records 219 model keys, all 214 shared R57 keys,
and `global_step=4000`. The following supervised stage loaded all 214 shared
keys and otherwise matched R58 exactly for 12000 steps.

Two initial SiFlow records exposed launcher-only readonly-variable failures.
UUID `4a011e80-1075-4bba-a117-5c5ec9ca9f9d` stopped before pretraining; UUID
`2e267c00-7ee4-41bc-8613-5e2ebed5ecb1` completed all 4000 pretraining steps
but stopped before checkpoint verification. The latter checkpoint was fetched,
verified as SHA256
`32ba0c3aa14a59cb3fa285bb0c5875f96574a1e1bd65da2c3751eeeecf3b2262`,
and reused without recomputation. Successful continuation UUID
`0a62d570-a2b6-4208-bfde-71df64eabc59`
(`mec8nm-r73-decaware-domain-r2-f8b87367`) completed in 25m07s. It asserted
exactly 80 finite M08/M10 WaterZ rows, an empty failure ledger, and no M11/M12
access.

The paired analyzer finds zero strict R58 gate passes. Best balance is zero
bias, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.055075340982/0.486239101228`, M08
`0.037673258094/0.443902435184` (`n_pred=267`), and M10
`0.072477423870/0.528575767271` (`n_pred=113`). Mean ARAND/VOI regress
`144.2798%/56.2548%` versus R58, while M08 regresses
`152.0247%/73.3128%`. Reject R73 before full-block, M11/M12, deployment, or
2PB inference. Do not repeat the raw-gradient pseudo-affinity objective. If
domain-aligned raw pretraining is retried, preserve the R57 decoder/head and
isolate adaptation to encoder parameters. Evidence:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r73_decoderaware_domain_20260713/paired_analysis_vs_r58.json`.

### R74 Encoder-Only MEC Domain Adaptation Rejected

R74 isolated the R73 failure mechanism by preventing raw-only adaptation from
touching the R57 decoder or affinity head. Standard masked reconstruction ran
for 4000 steps on raw M01--07/M09 after loading exactly 77 shared R57 encoder
tensors. An audited exporter replaced those 77 tensors in R57 and preserved
all 137 decoder/head tensors bit-identically. The resulting hybrid SHA256 was
`9195891cabe131f555575c3231c7e02f481d5f984daca832cb7f40a88315fca0`;
the R58-style supervised stage loaded all 214 tensors and ran for 12000 steps.

SiFlow UUID `9d0d3cb6-113a-46da-a3d6-81b1c7d8fa78`
(`mec8nm-r74-encoderadapt-4741d965`) succeeded in 26m45s. The remote assertion
and TOS evidence contain exactly 80 finite M08/M10 WaterZ rows and no failures.
The paired analyzer finds zero strict gate passes. Best balance is zero bias,
temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.035560839373/0.359082802791`; M08
`0.037953939399/0.356534733043`; M10
`0.033167739347/0.361630872539`. Mean ARAND/VOI regress
`57.7257%/15.3926%`, and M08 regresses `153.9024%/39.2018%` versus R58.

The M08-specific negative-bias point improves M08 by `1.4179%/6.5262%`, but
the identical setting regresses M10 by `119.9936%/39.4416%`; it is not a
paired promotion. Reject R74 before full-block, M11/M12, deployment, or 2PB
inference. Do not sweep the short raw-adaptation schedule or learning rate.
The selected anchors remain R58 crop and R57 M08/M10 full-block. Detailed
report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r74_encoderadapt_crop_report_20260713.md`.

### R75 Halo-Context Consistency Rejected

R75 returned to the exact R57 initializer and unchanged R58 supervised recipe.
Its only delta was a weight-`.05` stopped-teacher consistency term: the teacher
saw the ordinary `32x160x160` crop, while the student saw the same crop with
only the outer `4x16x16` z/y/x halo reflect-reconstructed from the byte-
identical central `24x128x128` core. The probability-MSE term applied to the
three affinity channels only inside that core; R72 photometric consistency was
disabled.

SiFlow UUID `1dde395e-27a4-46e8-b5a2-5be9199049ed`
(`mec8nm-r75-halo-consistency-e68c2276`) succeeded in 42m25s on one 8-GPU
Shanghai `med-model` pod. Runtime logs verified CUDA 8, WaterZ, 214 R57 keys,
eight fixed training paths, and the declared halo config. Training reached
12000 steps in 2059 seconds. The final assertion and independent TOS fetch
contain exactly 80 finite M08/M10 WaterZ rows and no failures.

There are zero strict R58 gate passes. The only Pareto/best-balance point is
bias `(-.20,-.45,-.45)`, temperature `1.00`, threshold `.50`: mean
ARAND/VOI `0.036896625527/0.387556128221`; M08
`0.040070609288/0.377633169630`; M10
`0.033722641766/0.397479086812`. Relative to R58, mean ARAND/VOI regress
`63.6504%/24.5427%`, and M08 regresses `168.0624%/47.4393%`.

Reject R75 before full-block, M11/M12, deployment, or 2PB inference. Do not
sweep its halo width, reflection mode, or consistency weight. The selected
anchors remain R58 crop and R57 M08/M10 full-block. Detailed report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r75_halo_consistency_crop_report_20260713.md`.

### R76 R58/R71 Checkpoint Soup Rejected

R76 evaluated whether small weight-space movement from R58 final toward the
R71 EMA mechanism signal could preserve M08 while borrowing aggregate gains.
Both sources are seed1789, step12000, `unetr_aniso_em`, with identical 214-key
topology and non-floating state. R71-to-R58 relative L2 distance is `1.8140%`
overall (`0.1749%` encoder, `2.5505%` decoder/head). The bounded nonzero alphas
were `.010,.025,.050,.100,.200`; alpha zero remained the frozen R58 anchor.

UUID `8597f63a-4517-4100-9fce-17eef686fbf9` failed after soup construction on
a readonly shell assignment, and UUID
`b507e2de-241c-4a1c-ae6c-37c35a474e05` failed on a missing evaluator import.
Neither reached inference. Successful UUID
`51a8f0f0-3873-4676-9e45-17371f9d1077`
(`mec8nm-r76-checkpoint-soup-r3-8db4ebe2`) completed in 26m01s. All five
candidates independently produced exactly 80 finite M08/M10 WaterZ rows and
empty failure ledgers.

Every alpha has zero strict R58 gate passes. Least-regressed alpha `.050`, zero
bias, temperature `1.15`, threshold `.50`, gives mean ARAND/VOI
`0.030034391192/0.338540905377`; M08
`0.035741292818/0.342101735986`; M10
`0.024327489565/0.334980074768`. Mean ARAND/VOI regress
`33.2138%/8.7914%`, while M08 regresses `139.1004%/33.5667%`.

Reject cross-run R58/R71 linear soup before full-block or frozen splits. Do
not expand the alpha grid. The selected anchors remain R58 crop and R57
M08/M10 full-block. The next bounded checkpoint-only test may compare R58's
own step8k/step10k checkpoints for early stopping; it must not interpolate
independently trained runs. Detailed report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r76_checkpoint_soup_crop_report_20260713.md`.

### R77 R58 Intermediate Checkpoints Rejected; Current Stop Point

R77 tested discrete early stopping on the exact R58 trajectory without new
training or weight interpolation. Audited step8000 and step10000 checkpoints
record seed1789, their exact global steps, no EMA, and the same 214-key
`unetr_aniso_em` topology as final. SiFlow UUID
`756ad839-ab71-463b-84e5-94cba285ee58`
(`mec8nm-r77-intermediate-ckpts-2f54fe03`) succeeded in 11m08s. Both candidates
independently returned exactly 80 finite M08/M10 WaterZ rows and empty failure
ledgers.

Neither candidate passes the strict R58 gate. Step8000's best point is zero
bias, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.025127315227/0.308270133998`; M08
`0.043286870594/0.405479642617`; M10
`0.006967759861/0.211060625378`. Mean VOI improves `0.9362%`, but mean ARAND
regresses `11.4491%` and M08 regresses `189.5784%/58.3114%`. Step10000 is
worse at mean `0.030151783802/0.368098632039`, with M08
`0.052432854055/0.499382871464`.

Reject R77 before full-block or frozen splits and do not sweep earlier R58
checkpoints. The selected MEC anchors remain R58 step12000 crop
(`0.022546003667/0.311183450751`) and R57 full-block
(`0.252279795612/1.062235821039`). R73--R77 close raw-domain adaptation,
reflect-halo consistency, cross-run soup, and early-stopping variants. A new
GPU wave now requires a materially new method, such as supervised consistency
between real adjacent overlapping HDF5 windows with shared augmentation and
coordinate-aware overlap matching; it must not be another parameter sweep.
Detailed report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r77_intermediate_checkpoints_crop_report_20260713.md`.

### R78 Real Paired-Overlap Consistency Rejected

R78 kept R58 fixed and added a weight-`.05` stopped-teacher loss between real
same-HDF5 crops at signed `8x32x32` offsets. One shared geometry/intensity
augmentation preserved a coordinate-aligned `24x128x128` overlap; inconsistent
affinity targets from crop-edge widening were excluded. The first submission
`cf5a12ba-a390-4f0e-9082-3743dcb5b812` failed before data/training on a
readonly shell assignment. Repaired UUID
`5b34b36a-1b10-416f-a744-76eed2a1874f`
(`mec8nm-r78-real-overlap-r2-cbd8a536`) succeeded. It verified CUDA 8,
WaterZ, 214 R57 keys, M01--07/M09 only, dataset size 32768, and trained 12000
steps in 2115 seconds.

The independent fetch has exactly 80 finite M08/M10 WaterZ rows and no
failures. Best balance is bias `(-.20,-.45,-.45)`, temperature `1.00`,
threshold `.50`: mean ARAND/VOI `0.021307430975/0.254415142584`; M08
`0.037685313511/0.346308913501`; M10
`0.004929548440/0.162521371668`. Mean metrics improve `5.4935%/18.2427%`, but
M08 regresses `152.1054%/35.2093%`. Thirteen of 32 settings improve both mean
metrics, while none preserves both M08 metrics; strict gate passes are zero.

Reject R78 before full-block or frozen splits. Do not sweep its shift, weight,
or teacher direction. R58 crop and R57 M08/M10 full-block remain selected.
Generic real-context invariance joins EMA as a mechanism that strongly helps
M10 while harming M08; the next method must preserve low-boundary/simple-volume
topology using training-only signals. Detailed report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r78_paired_overlap_crop_report_20260713.md`.

### R79 Topology-Routed Dual-Teacher Distillation Rejected

R79 preserved the complete R58 student line and added only a weight-`.10`
training-time probability distillation term. Raw training-label crop
foreground, boundary, and unique-instance statistics softly routed simple
crops toward frozen R58 and complex crops toward frozen R71 EMA. Statistics
were measured before boundary widening/augmentation; sampling and the main
supervised loss were unchanged. Both teachers loaded all 214 keys and were
frozen. A 1024-crop training-only audit measured route mean `.110561` with
139 crops at route >= `.5`.

The first local TOS upload failed through inherited proxy before any SiFlow
task existed. The byte-identical bundle was retried direct. UUID
`4c50adc6-ee6c-4ccf-bd28-e2341af5e770`
(`mec8nm-r79-topology-teacher-1a465cfa`) succeeded with CUDA 8, WaterZ, the
three expected 214-key loads, 12000 training steps in 1755 seconds, exactly
80 finite M08/M10 WaterZ rows, and no failures.

There are zero strict R58 gate passes. Best balance is zero bias, temperature
`1.00`, threshold `.50`: mean ARAND/VOI
`0.062021381189/0.412646966283`; M08
`0.035591466186/0.337509064870`; M10
`0.088451296192/0.487784867696`. Mean ARAND/VOI regress
`175.0881%/32.6057%`; M08 regresses `138.0980%/31.7736%`. At R58's exact
temperature-1.15 setting, mean is also negative at
`0.059614581022/0.441990110322`.

Reject R79 before full-block or frozen splits. Do not sweep its distillation
weight, route thresholds/widths, or teacher pair. R58 crop and R57 full-block
remain selected. Crop-level routing between R58 and R71 EMA is closed; a new
wave requires a materially different spatial or representation-level method,
not another teacher blend. Detailed report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r79_topology_teacher_crop_report_20260713.md`.

### R80 Decoder-Feature Instance Embedding Rejected

R80 kept the full R58 recipe fixed and added a weight-`.02` discriminative
loss to the existing 32-channel final decoder feature map. L2-normalized
features were sampled at z/y/x stride `2/4/4`; background and instances with
fewer than eight sampled voxels were ignored, and at most 64 instances per
crop were used. Equal within-instance variance and between-centroid distance
terms used margins `.20/1.0`. The change added no model parameters, checkpoint
keys, teacher, affinity output, crop sampling, or inference-time operation.

Six focused tests and all 37 MEC tests passed, including bit-identical default
logits and unchanged state keys. A one-step CPU CLI smoke run completed. A real
R57-initialized crop scale audit measured embedding loss `.682809`, contributing
`.013656` after weighting. The staged R57 checkpoint loaded all 214 keys.

SiFlow UUID `c4ced6cd-7387-4d3f-baf4-7ca2bf3504aa`
(`mec8nm-r80-instance-embedding-9edb4c3d`) ran as the only MEC wave on one
8-GPU Shanghai `med-model` pod and succeeded in 28m33s. Startup verified CUDA
8, WaterZ, dataset size 32768, train IDs M01--07/M09, and the exact embedding
configuration. Training reached step 12000 in 1344 logged seconds with no
runtime errors. The final embedding loss remained finite at `.161571`, with
variance/distance `.070016/.091554`.

The remote assertion and independent TOS fetch contain exactly 80 finite
WaterZ rows, 40 each for M08/M10, and an empty failure list. The paired analyzer
finds zero strict R58 gate passes. Best balance is bias
`(-.20,-.45,-.45)`, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.018971582484/0.264526010140`; M08
`0.033522118658/0.371451721207`; M10
`0.004421046309/0.157600299074`. The means improve
`15.8539%/14.9935%`, while M08 regresses `124.2546%/45.0258%`. At the exact
R58 zero-bias temperature-1.15 setting, mean metrics regress to
`0.062431049101/0.418866062355`.

Reject R80 before full-block, M11/M12, deployment, or 2PB inference. Do not
sweep embedding weight, feature stride, margins, instance filters/cap, or
normalization. R80 joins R71/R78 as a mechanism that improves aggregate/M10
while failing M08 preservation; that signal cannot override the frozen gate.
R58 remains the crop anchor and R57 remains the M08/M10 full-block anchor.
Detailed report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r80_instance_embedding_crop_report_20260714.md`.

### R81 Instance-Pair Boundary-Leak CVaR Rejected

R81 kept the complete R58 line and added a weight-`.05`, parameter-free loss
on the original pre-widening labels. For every true adjacent nonzero instance
pair, it selected the highest 10% predicted boundary affinities and penalized
squared excess above `.50`; interfaces needed four edges and at most 64 pairs
were retained. Per-pair then per-crop normalization protected sparse/simple
interfaces without changing sampling, global negative weights, the model,
outputs, ordinary loss, teachers, EMA, or inference.

Eight focused tests and all 45 MEC tests passed. A CPU CLI smoke run completed
loss/backward/checkpoint. An eight-block real R57 audit measured mean raw leak
loss about `.067`, or `.0034` after weighting. The bundle loaded all 214 R57
keys and added no state key.

SiFlow UUID `76ac0302-57de-4868-abb7-25315cf1c41d`
(`mec8nm-r81-boundary-leak-295cbbc2`) ran as the only MEC wave on one 8-GPU
Shanghai `med-model` pod and succeeded in 32m15s. Training reached step 12000
in 1559 logged seconds with no errors. The remote assertion and independent
fetch contain exactly 80 finite WaterZ rows, 40 each for M08/M10, and an empty
failure list.

The paired analyzer finds zero strict R58 gate passes. Best balance is bias
`(-.20,-.45,-.45)`, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.028377941932/0.352231169545`; M08
`0.032977813947/0.375969227906`; M10
`0.023778069917/0.328493111184`. Mean metrics regress
`25.8668%/13.1908%`, M08 regresses `120.6134%/46.7896%`, and M10 improves
`21.1178%/10.3063%`. M08 still has only 202 predictions and VOI merge
`.207795`, versus R58's 242 and `.081985`.

Reject R81 before full-block, M11/M12, deployment, or 2PB inference. Do not
sweep leak weight, tail fraction, maximum affinity, minimum edges, or pair cap.
Local voxel-interface cannot-link supervision is insufficient for WaterZ's
region-level merge ordering. R58 remains the crop anchor and R57 remains the
M08/M10 full-block anchor. Detailed report:
`/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_r81_boundary_leak_crop_report_20260714.md`.

### Pretraining and R58 Stem-Micro Post-processing Audit

Pretraining has a controlled downstream signal on CREMI. Same-seed 12k R45
PublicEM dbMiM reaches VOI/ARAND `0.986481/0.186187`, versus R47 PublicEM
plain MAE `1.043065/0.190743`, improving 5.42%/2.39%. Older R17/R23 also shows
a 2.35% VOI gain. There is no strict R58 scratch-versus-PublicEM MEC run, but
R70's initializer-only replacement of PublicEM-R57 with FullEM-R33 regresses
MEC crop ARAND/VOI by 137.06%/56.65%. Keep PublicEM dbMiM initialization.

The external CREMI-trained Stem-Micro scorer uses the exact R58 checkpoint
(SHA256 `4715c6679808eac5e073159202be1cca77307ab90202bd683f8ab97cacc55e09`),
zero bias, temperature 1.15, and fixed alpha 0.07 without MEC scorer training.
Hzzs26 passes the crop gate. Hzzs27 full M08/M10 changes paired same-R58
WaterZ `ARAND/VOI/time=0.418966/1.536809/70.800s` to
`0.091720/0.590987/17.803s`, improving 78.11%/61.54% and 3.98x in latency.

Hzzs28/hzzs29 are the one designated M11/M12 replay. Against paired same-R58
WaterZ `0.019254/0.279893/68.807s`, Stem-Micro reaches
`0.016160/0.334732/17.840s`: 16.07% better ARAND and 3.86x lower latency, but
19.59% worse VOI. M11 regresses in both quality metrics and M12 improves both.
M11/M12 were historically inspected and must not be reopened for selection.
Stem-Micro is a strong validation/runtime result, not a uniform WaterZ quality
replacement; R57 + calibrated WaterZ remains the conservative MEC quality
line. Detailed audit:
`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/evidence/mec8nm_pretraining_stem_micro_audit_20260714.md`.
