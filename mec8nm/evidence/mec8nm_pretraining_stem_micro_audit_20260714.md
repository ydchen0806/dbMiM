# MEC Pretraining and Stem-Micro Audit

Updated: 2026-07-14

## Answer

The PublicEM dbMiM pretraining line is useful, but the strength of the evidence
depends on the question:

- On CREMI, controlled same-seed comparisons show that the dbMiM objective is
  better than plain MAE.
- On MEC, PublicEM-R57 is the strongest tested initializer lineage, and swapping
  only to FullEM-R33 is strongly negative.
- A strict MEC R58 scratch-versus-PublicEM run does not exist, so the isolated
  causal gain from loading PublicEM weights on MEC is not measured directly.

Stem-Micro is also useful, but it is not a uniform replacement for WaterZ. It
strongly improves the frozen R58 setting on M08/M10 validation. On the single
paired M11/M12 replay it improves ARAND and latency but regresses VOI, with
opposite block-level behavior. The conservative MEC quality default therefore
remains the established R57 + calibrated WaterZ line.

## Pretraining Evidence

Lower VOI and ARAND are better.

| Controlled comparison | VOI | ARAND | Interpretation |
|---|---:|---:|---|
| R45 PublicEM dbMiM, seed 309, 12k | 0.986481 | 0.186187 | dbMiM |
| R47 PublicEM plain MAE, seed 309, 12k | 1.043065 | 0.190743 | same downstream recipe |
| Relative dbMiM gain | 5.42% | 2.39% | controlled objective gain |
| R17 PublicEM dbMiM | 1.002919 | -- | older comparison |
| R23 PublicEM plain MAE | 1.027073 | -- | older comparison |
| Relative dbMiM gain | 2.35% | -- | independently consistent VOI signal |

The R45/R47 configs hold the seed, PublicEM data family, anisotropic UNETR,
crop, mask ratio, finetuning data, loss, and 12k schedule fixed; the main change
is dbMiM-specific pretraining versus plain MAE. This supports loading the dbMiM
checkpoint rather than the plain-MAE control.

The MEC initializer control is R70 versus the frozen R58 lineage. R70 changes
the initializer from PublicEM-R57 to FullEM-R33 while retaining the R58 seed and
supervised recipe. PublicEM-R57/R58 is `ARAND/VOI=0.022546/0.311183` on the
M08/M10 crop anchor; FullEM-R33/R70 is `0.053448/0.487476`, a 137.06% ARAND
and 56.65% VOI regression. More external data is therefore not automatically a
better initializer for MEC.

## Frozen Same-Affinity Post-processing Test

R58 checkpoint SHA256:

`4715c6679808eac5e073159202be1cca77307ab90202bd683f8ab97cacc55e09`

The learned route uses the externally trained CREMI query scorer
`hzx_w64_morph.pt`, zero affinity bias, temperature 1.15, and structural stop
alpha 0.07. The scorer is not trained on MEC labels. The WaterZ comparator uses
the same R58 affinities and the predeclared crop threshold 0.50.

| Split / extent | R58 post-process | VOI | ARAND | Time (s) |
|---|---|---:|---:|---:|
| M08/M10 crop | WaterZ | 0.311183 | 0.022546 | 1.293 |
| M08/M10 crop | Stem-Micro | 0.284585 | 0.013019 | 1.799 |
| M08/M10 full | WaterZ | 1.536809 | 0.418966 | 70.800 |
| M08/M10 full | Stem-Micro | 0.590987 | 0.091720 | 17.803 |
| M11/M12 full | WaterZ | **0.279893** | 0.019254 | 68.807 |
| M11/M12 full | Stem-Micro | 0.334732 | **0.016160** | **17.840** |

On M08/M10 full validation, Stem-Micro reduces VOI by 61.54%, ARAND by
78.11%, and latency by 3.98x. It passed the predeclared crop gate separately on
both M08 and M10 before full evaluation.

On M11/M12, Stem-Micro is 3.86x faster and lowers mean ARAND by 16.07%, but
raises mean VOI by 19.59%. M11 regresses from WaterZ
`VOI/ARAND=0.282348/0.015713` to `0.402241/0.018388`; M12 improves from
`0.277437/0.022795` to `0.267223/0.013932`. This is a real speed/ARAND trade-off,
not a four-metric quality gain.

M11/M12 are designated test blocks but are historically non-pristine because
older dbMiM WaterZ and slabwise studies had already inspected them. Hzzs28/29
must not be used to select another scorer, alpha, calibration, threshold, or
executor.

## Current Decision

1. Keep PublicEM dbMiM initialization for the R57/R58 affinity line. Do not
   replace it with FullEM-R33, plain MAE, or MEC raw-only adaptation based on
   the current evidence.
2. Keep R58 + Stem-Micro as a strong validation and runtime research result,
   not the final MEC quality default.
3. Keep R57 + calibrated WaterZ as the conservative deployment-quality line.
   Its partition-equivalent M11/M12 full-block anchors are about
   `VOI/ARAND=0.224035/0.012460` (VOI-priority) and
   `0.228193/0.012404` (ARAND-priority), both better than R58 + Stem-Micro.
4. Any next learned post-processor must be selected without M08/M10/M11/M12 and
   evaluated on a new untouched labeled domain or newly held-out blocks.

## Evidence

- R45 dbMiM config:
  `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r16_seed309_r45q.yaml`
- R47 plain-MAE config:
  `configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_plainmae_seed309_r47q.yaml`
- MEC master state:
  `evidence/mec8nm_stage1_state_20260710.md`
- Stem-Micro report:
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/docs/dbmim_transfer/stem_micro_mec_r58_hzzs26_hzzs27_20260714.md`
- Machine-readable hzzs26--hzzs29 evidence:
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/paper_tmi/data/cross_dataset_hzzs26_20260714/`
  through
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/paper_tmi/data/cross_dataset_hzzs29_20260714/`
