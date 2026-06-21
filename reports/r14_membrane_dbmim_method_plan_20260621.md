# R14 Membrane-Aware dbMiM Plan

Date: 2026-06-21

## Goal

R12 fixed the CREMI A/B/C official waterz protocol. R13 moved fine-tuning from
BCE to SuperHuman-style weighted MSE and added Boundary-Calibrated Affinity
Ranking (BCAR). R14 turns this into a cleaner paper-level method:

**Membrane-aware Anisotropic dbMiM (MA-dbMiM)**.

The purpose is to make pretraining itself EM-specific instead of hoping a
generic masked reconstruction objective transfers to neuron boundaries.

## Method

Standard dbMiM masks ViT patches and reconstructs raw intensity with MSE plus a
small gradient structure loss. R14 changes the pretraining objective only:

```text
edge(x) = normalized(|dz x| * wz + |dy x| * wy + |dx x| * wx)
L_pixel = mean_masked((pred - x)^2 * (1 + lambda_edge * edge_patch))
L_struct = weighted_grad_l1(pred, x; wz, wy, wx)
L_pretrain = L_pixel + lambda_struct L_struct
```

The default R14 setting is EM-anisotropic:

- membrane patch weight: `lambda_edge = 1.25`
- membrane axis weights: `(z, y, x) = (0.25, 1.0, 1.0)`
- structure axis weights: `(z, y, x) = (0.5, 1.0, 1.0)`
- structure weight: `0.2`

This uses raw EM images only, so it can consume all available unlabeled EM
pretraining data. The objective should push the backbone to encode membranes,
in-plane continuity, and conservative z context before supervised affinity
fine-tuning.

## Paper Claim To Test

BCAR addresses the supervised agglomeration objective. MA-dbMiM addresses the
pretraining objective. The strongest claim needs both:

> EM-specific masked pretraining learns membrane-biased anisotropic structure;
> BCAR converts that representation into better waterz merge ordering.

The paper table should report full-volume CREMI A/B/C official waterz VOI and
ARAND, not sample-A-only diagnostics.

## Experiment Matrix

### Running Baseline

R13 is already running or queued:

| run | pretrain | finetune loss | BCAR |
|---|---|---|---|
| `em-shwmse-allpretrained-r13` | CREMI dbMiM r6 | SH-MSE | no |
| `em-shwmse-scratch-r13` | none | SH-MSE | no |
| `em-shwmse-bcar-allpretrained-r13` | CREMI dbMiM r6 | SH-MSE | yes |
| `em-shwmse-bcar-scratch-r13` | none | SH-MSE | yes |

### Quick R14 Screen

These use CREMI dbMiM r6 and only 12k steps, for fast BCAR component selection:

| run | purpose |
|---|---|
| `em-shwmse-bcar-rank-allpretrained-r14q` | BCAR ranking term only |
| `em-shwmse-bcar-calib-allpretrained-r14q` | z/xy calibration term only |

### Full R14 Method

| run | pretrain | finetune loss | BCAR |
|---|---|---|---|
| `pretrain-em-membrane-r14` | all EM + MA-dbMiM | pretraining only | n/a |
| `em-shwmse-mempretrained-r14` | MA-dbMiM r14 | SH-MSE | no |
| `em-shwmse-bcar-mempretrained-r14` | MA-dbMiM r14 | SH-MSE | yes |

The key comparisons are:

- `em-shwmse-mempretrained-r14` vs `em-shwmse-allpretrained-r13`
  tests whether membrane-aware pretraining improves over CREMI-only dbMiM.
- `em-shwmse-bcar-mempretrained-r14` vs `em-shwmse-bcar-allpretrained-r13`
  tests whether MA-dbMiM and BCAR are complementary.
- `em-shwmse-bcar-mempretrained-r14` vs `em-shwmse-scratch-r13`
  is the primary paper claim.

## Success Criteria

Primary metric: best A/B/C mean `voi_sum` under official waterz sweep.

Secondary metric: `adapted_rand_error` at the best-VOI threshold and the best
ARAND row under the same sweep.

The result is paper-worthy if MA-dbMiM + BCAR improves best VOI and does not
trade away ARAND relative to the scratch and CREMI-only pretrained baselines.
If MA-dbMiM improves only ARAND, it is still useful but should be framed as
merge stability rather than universal segmentation quality.

## 2026-06-21 Run Update

The first `pretrain-em-membrane-r14` SiFlow attempt failed before training
because `tos://agi-data/users/dchen02/dbmim/assets/em_pretrain_data/all` listed
successfully but contained no HDF5 files. Local `data/EM_pretrain_data` also
contains only Hugging Face API manifests, not downloaded EM volumes.

The submitter now tries three levels in order:

1. `em_pretrain_data/all`
2. group prefixes: `fafb`, `fib25`, `kasthuri`, `mitoem`, `mb_moc`
3. CREMI-only fallback with an explicit stdout marker:
   `em_pretrain_data_status=missing_offline_tos_fallback_to_cremi_only`

The rerun UUID is `76cc096f-95db-4d44-afb8-b6c69c4eebb2`. If the fallback
marker appears, the checkpoint is still a valid MA-dbMiM membrane-objective
pretrain, but it must be described as CREMI-only until the gated/all-EM HDF5
assets are prepared and uploaded.

An additional paper-oriented fine-tuning ablation was added:

**Membrane-Aware Weighted Supervision (MAWS)**. During supervised affinity
fine-tuning, the main pointwise affinity loss is spatially reweighted by a
normalized anisotropic raw-EM membrane proxy:

```text
w = normalize(1 + lambda_maws * edge(raw))
L_aff = mean(w * L_affinity)
```

The quick screen stage is
`finetune-cremi-unetr-aniso-em-shwmse-maws-bcar-rank-allpretrained-r14q`,
UUID `38b18ca3-d4c8-4fd2-94d9-632f590d92ce`. It uses the same 12k-step setup as
R14q rank-only and only adds MAWS, so the comparison isolates whether membrane
supervision improves waterz VOI/ARAND beyond BCAR ranking.

Two additional 12k-step MAWS controls were submitted:

| run | UUID | purpose |
|---|---|---|
| `em-shwmse-maws-allpretrained-r14q` | `2f63a6dc-c7a6-4e8b-97de-34ab80985b40` | MAWS-only, isolates membrane-weighted supervision without BCAR. |
| `em-shwmse-maws15-bcar-rank-allpretrained-r14q` | `30cb7b4a-ac52-402a-9085-c97749ab5f2b` | Stronger membrane weight (`lambda_maws=1.5`) plus BCAR rank, tests over-focus on membranes. |

Together with existing R14q rank and calibration runs, this gives the quick
ablation grid:

| run | MAWS | BCAR rank | BCAR calib |
|---|---:|---:|---:|
| `em-shwmse-bcar-rank-allpretrained-r14q` | no | yes | no |
| `em-shwmse-bcar-calib-allpretrained-r14q` | no | no | yes |
| `em-shwmse-maws-allpretrained-r14q` | yes, 0.75 | no | no |
| `em-shwmse-maws-bcar-rank-allpretrained-r14q` | yes, 0.75 | yes | no |
| `em-shwmse-maws15-bcar-rank-allpretrained-r14q` | yes, 1.5 | yes | no |

## 2026-06-21 Full R14 Submissions

The 120k-step MA-dbMiM pretraining completed and synchronized
`tos://agi-data/users/dchen02/dbmim/outputs/pretrain_em_membrane_dbmim_r14/pretrained_latest.pt`
plus 2k-step intermediate checkpoints. This run used the CREMI-only fallback
path because the all-EM TOS prefix still contains manifests but no HDF5 volumes.

The quick R14q A/B/C sweep favored MAWS+BCAR-rank with `lambda_maws=0.75`:

| run | best VOI | ARAND at best VOI | note |
|---|---:|---:|---|
| `bcar-rank-allpretrained-r14q` | 1.0818 | 0.1965 | 12k quick screen |
| `maws-allpretrained-r14q` | 1.0745 | 0.2011 | MAWS-only |
| `maws-bcar-rank-allpretrained-r14q` | 1.0407 | 0.1929 | best full A/B/C quick result |
| `maws15-bcar-rank-allpretrained-r14q` | 1.0441 | 0.1973 | stronger MAWS was not better |

The `bcar-calib-allpretrained-r14q` fallback summary currently contains only
sample A (`n=1`) and is therefore not used as an A/B/C conclusion.

Three full 40k-step MA-dbMiM fine-tuning jobs were submitted on cpt-train:

| run | UUID | purpose |
|---|---|---|
| `em-shwmse-mempretrained-r14` | `f4c00499-19a5-4fb2-99a8-99adf54bad4d` | Tests MA-dbMiM pretraining without BCAR/MAWS. |
| `em-shwmse-bcar-mempretrained-r14` | `d23472e9-567f-4bd3-9e04-24f450dbab85` | Tests MA-dbMiM plus original BCAR rank+calibration. |
| `em-shwmse-maws-bcar-rank-mempretrained-r14` | `9494b4fa-f6e6-410c-90ba-052bb8e70d01` | Strongest current method: MA-dbMiM + MAWS + BCAR rank. |
