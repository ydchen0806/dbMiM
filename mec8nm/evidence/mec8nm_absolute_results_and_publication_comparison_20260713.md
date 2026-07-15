# MEC8nm Absolute Results and Publication Comparison

## Comparable MEC Results

- Selected crop anchor R58, M08/M10 center `64x512x512`: mean ARAND/VOI
  `0.022546003667/0.311183450751`.
- Best aggregate diagnostic R71 EMA: `0.020969504209/0.299190914044`, but it is
  rejected because M08 is `0.037575210047/0.433248168057` versus R58
  `0.014948239287/0.256127966754`.
- R72 photometric consistency is a strict negative: mean ARAND/VOI
  `0.028005094678/0.369409206168`, with M08
  `0.042332990281/0.473893471718`. It has zero crop-gate passes and receives no
  full-block or frozen-split follow-up.
- R73--R77 all fail the same strict gate. They test whole-network raw-domain
  pretraining, encoder-only raw adaptation, reflected-halo consistency,
  R58/R71 cross-run weight soup, and R58 intermediate checkpoints. The least
  regressed late candidate is R77 step8000 at mean ARAND/VOI
  `0.025127315227/0.308270133998`, but M08 is
  `0.043286870594/0.405479642617`; it is rejected.
- R78 real paired-overlap consistency improves mean ARAND/VOI to
  `0.021307430975/0.254415142584`, but M08 regresses to
  `0.037685313511/0.346308913501`. None of its 32 operating points preserves
  both M08 metrics, so it is rejected before full-block work.
- Reliable full-block anchor R57, complete M08/M10 volumes: mean ARAND/VOI
  `0.252279795612/1.062235821039`. This is the current honest deployment-scale
MEC result. No later candidate has passed the protocol to replace it.

As of R78, R58 step12000 remains the selected crop checkpoint. R57 remains the
only full-block anchor because no R58-successor candidate has cleared the crop
gate required to schedule new M08/M10 full-block validation.

## Published/Released dbMiM CREMI Results

The repository's released CREMI A/B/C full-volume results are R48
`ARAND=0.178252, VOI=0.962154` and R57
`ARAND=0.178248, VOI=0.964617`. These are published/released model results on a
different dataset, label taxonomy, sample set, and three-volume protocol. They
show that the architecture is competitive on CREMI, but they are not a valid
numerical leaderboard comparison for MEC.

The MEC data card identifies the data as EMNeXt/MEC myelin segmentation under
CC-BY-4.0 but provides no same-split, same-WaterZ ARAND/VOI publication table.
Therefore no defensible claim that `0.25228/1.06224` beats or trails the
published MEC SOTA can be made from local evidence. A paper comparison requires
the EMNeXt paper's exact split, metric implementation, foreground treatment,
and full-volume inference protocol.

## Interpretation

MEC crop scores near ARAND `.02` and VOI `.30` are strong local diagnostics but
are not deployment numbers. Full-block performance is roughly ARAND `.252` and
VOI `1.062`, exposing a major crop-to-full-volume topology gap. Future claims
must lead with full-block M08/M10 and reserve crop metrics for model screening.
