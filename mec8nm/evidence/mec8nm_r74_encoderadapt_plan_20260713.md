# MEC 8 nm R74 Encoder-Only Domain-Adaptation Plan

R74 is the mechanistic follow-up to R73. It tests whether short MEC raw-domain
adaptation is useful when it cannot overwrite the R57 convolutional decoder or
affinity head.

## Controlled hypothesis

Load the 77 encoder parameters shared by PublicEM-R57 and the standard DBMIM
masked autoencoder. Adapt those parameters for 4000 steps on raw M01--M07/M09
crops using masked reconstruction, structure, and membrane weighting only.
Merge the adapted 77-key encoder into the untouched 137-key R57 decoder/head,
then run the unchanged R58 supervised recipe for 12000 steps.

The hybrid exporter fails closed unless the adapted checkpoint records step
4000, exactly the complete 77-key R57 encoder lineage, at least one changed
encoder tensor, all compatible shapes, and bit-identical preservation of every
non-encoder R57 tensor.

## Frozen protocol

- Raw pretraining: M01--M07/M09 only; no label path.
- Supervised training: M01--M07/M09 only.
- Validation: M08/M10 center crop, WaterZ only, exactly 80 finite rows.
- M11/M12 absent from the remote download and evaluation path.
- Exactly one 8-GPU Shanghai `med-model` wave.
- Pass only if mean ARAND, mean VOI, M08 ARAND, and M08 VOI all do not regress
  versus R58. A crop failure forbids full-block work.

## Outcome

Completed and rejected. SiFlow UUID
`9d0d3cb6-113a-46da-a3d6-81b1c7d8fa78` returned exactly 80 finite M08/M10
WaterZ rows, but the paired analyzer found zero strict R58 gate passes. Best
mean ARAND/VOI was `0.035560839373/0.359082802791`, with M08
`0.037953939399/0.356534733043`. No full-block task was submitted.
