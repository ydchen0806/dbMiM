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

