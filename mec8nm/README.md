# MEC 8 nm Myelin Instance Segmentation

This directory is the isolated, versionable MEC experiment workspace. It
contains the MEC configs, launch/evaluation scripts, runtime source snapshot,
skill/state handoff, split contract, and compact evidence needed to continue
the project without mixing MEC results with CREMI A/B/C results.

Large HDF5 volumes, checkpoints, WaterZ sources, SiFlow bundles, and complete
logs are intentionally not duplicated here. Their local and TOS locations are
listed in `ASSETS.md`.

The original repository layout is preserved. Root-level `dbmim/`, `configs/`,
`scripts/`, and training entrypoints remain available for historical commands;
this directory is a self-contained MEC snapshot. Shared runtime files must stay
byte-identical and are checked by `scripts/verify_workspace.py`. The canonical
repository skill is `../dbmim-neuron-seg-skill/`; `skill/` is its portable MEC
mirror, and the installed Codex copy is outside Git under
`~/.codex/skills/dbmim-neuron-seg/`.

## Frozen protocol

- Task: myelin instance segmentation. Label `0` is background.
- Supervised train blocks: M01--M07 and M09 only.
- Official validation blocks: M08 and M10 only.
- Designated test blocks: M11 and M12. The R58 + Stem-Micro candidate consumed
  one paired replay after passing both gates; older dbMiM experiments had
  already inspected these blocks, so they are historically non-pristine and
  must not be reused for model or post-processing selection.
- Headline backend: WaterZ. Lower ARAND and VOI are better.
- A crop result is not a full-block result and must be labeled as such.
- M07/M09 were later consumed as hzzs30--hzzs32 post-processing diagnostics.
  No existing MEC block remains untouched for another selection decision.

## Current anchors

| Scope | Run | Mean ARAND | Mean VOI | Status |
|---|---|---:|---:|---|
| M08/M10 center crop | R58 | 0.022546003667 | 0.311183450751 | frozen crop gate |
| M08/M10 full block | R57 | 0.252279795612 | 1.062235821039 | deployment-scale anchor |
| M08/M10 full block | R58 + LoCoAgg | 0.091720333957 | 0.590987031259 | strongest same-affinity validation/runtime result |
| M11/M12 full block | R58 + LoCoAgg | 0.016160111808 | 0.334732313606 | mixed paired replay; VOI worse than same-R58 WaterZ |
| M11/M12 full block | R58 + WaterZ crop-frozen | 0.019254036555 | 0.279892831006 | paired replay reference |
| M08/M10 center crop | R71 EMA | 0.020969504209 | 0.299190914044 | rejected: M08 regressed |
| M08/M10 center crop | R72 consistency | 0.028005094678 | 0.369409206168 | rejected: mean and M08 regressed |
| M08/M10 center crop | R73 decoder-aware domain SSL | 0.055075340982 | 0.486239101228 | rejected: mean and M08 regressed |
| M08/M10 center crop | R74 encoder-only domain SSL | 0.035560839373 | 0.359082802791 | rejected: mean and M08 regressed |
| M08/M10 center crop | R75 halo consistency | 0.036896625527 | 0.387556128221 | rejected: mean and M08 regressed |
| M08/M10 center crop | R76 R58/R71 soup alpha .05 | 0.030034391192 | 0.338540905377 | rejected: mean and M08 regressed |
| M08/M10 center crop | R77 R58 step8k | 0.025127315227 | 0.308270133998 | rejected: ARAND and M08 regressed |
| M08/M10 center crop | R78 real overlap consistency | 0.021307430975 | 0.254415142584 | rejected: M08 regressed |
| M08/M10 center crop | R79 topology-routed teachers | 0.062021381189 | 0.412646966283 | rejected: mean, M08, and M10 regressed |
| M08/M10 center crop | R80 decoder-feature instance embedding | 0.018971582484 | 0.264526010140 | rejected: M08 regressed |
| M08/M10 center crop | R81 instance-pair boundary leak | 0.028377941932 | 0.352231169545 | rejected: mean and M08 regressed |

The R58 crop operating point is zero bias, temperature `1.15`, threshold
`0.50`. A new crop candidate passes only if mean ARAND, mean VOI, M08 ARAND,
and M08 VOI all do not regress versus R58. Exactly 80 finite WaterZ rows for
M08/M10 and an empty failure ledger are required before analysis.

Hzzs32 replaces the heuristic fragment-count stop with exact signed local VI
utility and materially repairs pooled-scorer under-merging on consumed M07/M09,
but fails the predeclared M07 merge-VI guard. Hzzs33 reduces AxonEM split VOI
but creates severe false merges and fails both transfer gates. Neither is a
promoted checkpoint. The next admissible learning hypothesis is component-
rescored signed utility on new untouched labels, not another threshold, alpha,
domain-weight, seed, or static pooled-scorer sweep.

## Layout

```text
mec8nm/
  configs/          MEC configs and controlled experiment definitions
  data/MEC_8nm/     versioned manifest and frozen split contract only
  dbmim/            runtime source snapshot used by MEC jobs
  evidence/         compact reports, anchor CSV/JSON, and diagnostics
  scripts/          MEC launchers plus the two required evaluators
  skill/            portable mirror of the repository skill/state/UI metadata
  train_finetune.py MEC finetuning entrypoint snapshot
  train_pretrain.py MEC raw-only pretraining entrypoint snapshot
```

Legacy copies remain at the repository root because historical SiFlow bundles
and recorded commands refer to those paths. Apply a shared runtime change to
both locations in one commit and run the workspace verifier before merging.
Keep large volumes, checkpoints, generated outputs, credentials, and wheel
caches outside Git.

## Start here

```bash
cd /volume/med-train/users/dchen02/code/dbMiM/mec8nm
tail -n 320 skill/STATE.md
python scripts/verify_workspace.py
python -m pytest -q tests
python scripts/analyze_mec8nm_paired_crop.py \
  --metrics evidence/mec8nm_remote_results/r58_balanced_crop_20260711/cremi_segmentation_metrics.csv \
  --baseline-metrics evidence/mec8nm_remote_results/r58_balanced_crop_20260711/cremi_segmentation_metrics.csv \
  --output /tmp/mec8nm_r58_selfcheck.json
```

Rejected families R66--R81 must not be repeated: LSD auxiliary heads,
soft-negative weighting, fixed two-voxel affinity auxiliaries, border-width
changes, direct FullEM-R33 initialization, EMA checkpoint selection, and the
R72 stopped-teacher photometric-consistency recipe or weight/range sweeps, and
the R73 masked-reconstruction plus raw-gradient pseudo-affinity objective.
Earlier rejected calibration, gamma-dose, topology-sampling, logit-mixing, and
DPP families are documented in `skill/STATE.md`. R74 additionally rules out
short MEC masked-reconstruction adaptation even when all R57 decoder/head
tensors are preserved.
R75 additionally rules out reflect-reconstructed halo consistency and its
width/mode/weight sweeps.
R76 rules out linear checkpoint soup across the independently trained R58 and
R71 runs, including expansion of its alpha grid.
R77 rules out selecting or sweeping earlier checkpoints from the R58 run.
R78 rules out real paired-overlap consistency and shift/weight/teacher sweeps;
it improves M10 and both mean metrics but has zero M08-preserving settings.
R79 rules out crop-level topology routing between R58 and R71 EMA, including
route-threshold, width, distillation-weight, and teacher-pair sweeps.
R80 rules out generic decoder-feature instance embedding and sweeps of its
weight, stride, margins, instance filters/cap, or normalization; it improves
M10 and both means but has zero M08-preserving settings.
R81 rules out local instance-pair boundary-leak CVaR and sweeps of its weight,
tail fraction, affinity ceiling, minimum interface size, or pair cap; it still
leaves M08 strongly merge-biased.

Another GPU wave requires a materially new region/hierarchy-aware training
signal, or should preserve R58 and gate an independent post-processing delta.
Do not repeat simple-region sampling, teacher blending, generic consistency,
global instance embedding, or local boundary focal/CVaR variants. Run only one
MEC wave at a time unless the user explicitly changes the cap.

The detailed pretraining and post-processing decision audit is
`evidence/mec8nm_pretraining_stem_micro_audit_20260714.md`. Its key conclusion
is that PublicEM dbMiM initialization has controlled CREMI evidence and remains
the best tested MEC lineage, while Stem-Micro is a strong validation/runtime
result but not a uniform M11/M12 quality gain. Do not reopen M11/M12.

## PB-scale status

Blockwise affinity inference and local segmentation are available, but the
pipeline is not ready for PB-scale globally coherent instance segmentation.
It has not yet demonstrated volume-independent host memory, stable cross-block
`uint64` IDs, resumable manifests, failure recovery, or PB storage and compute
cost. Do not run global WaterZ or one dense global RAG over PB data.

Advance in order: single-block parity, `2x2x2` seam/stable-ID parity,
`1--10 TB` resumability and throughput, `100 TB` recovery and cost, then PB.
The near-term large-volume baseline is local WaterZ or MWS with deterministic
overlap reconciliation. Do not deploy the current learned LoCoAgg checkpoint
globally while the AxonEM false-merge failure remains open.

## Merge discipline

Create an integration branch from an up-to-date `origin/main`, stage only the
MEC runtime/config/evidence/skill assets, run the verifier and tests, then merge
the branch back with a merge commit. Never use a destructive reset to make the
mirrors agree. The verifier is the contract that the original root entrypoints
and the isolated workspace remain compatible.
