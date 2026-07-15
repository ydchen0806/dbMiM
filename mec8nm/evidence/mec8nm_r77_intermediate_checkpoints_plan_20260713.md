# MEC 8 nm R77 R58 Intermediate-Checkpoint Plan

R77 tests a narrow early-stopping hypothesis on the exact R58 training
trajectory. It performs no new training and no weight interpolation.

## Audited candidates

- Step 8000 SHA256:
  `7fc758d5afc18661241e49d16c4256889be08966eecafa33d23f562456a2c73f`.
- Step 10000 SHA256:
  `2aed3b67f565233fdce696d4341d6178a84524cbb42e181ac98ab833d9b1f130`.
- Both record seed1789, the exact step in their payload, `unetr_aniso_em`, no
  EMA, and the same 214-key topology as R58 final.
- Their relative L2 distances from R58 final are `5.3905%` and `3.4995%`.

Evaluate each discrete checkpoint independently on the frozen M08/M10 center-
crop WaterZ grid. Each must return exactly 80 finite rows and an empty failure
ledger. M11/M12 remain absent. Promote only a checkpoint whose mean ARAND,
mean VOI, M08 ARAND, and M08 VOI all do not regress versus R58 final. Otherwise
retain R58 step12000 and reject early stopping before full-block work.

## Outcome

Completed and rejected. SiFlow UUID
`756ad839-ab71-463b-84e5-94cba285ee58` returned two independently audited
80-row M08/M10 WaterZ results. Both have zero strict R58 gate passes. Step8000
has mean ARAND/VOI `0.025127315227/0.308270133998` but severely regresses M08;
step10000 is worse. R58 step12000 remains selected, and no full-block task was
submitted.
