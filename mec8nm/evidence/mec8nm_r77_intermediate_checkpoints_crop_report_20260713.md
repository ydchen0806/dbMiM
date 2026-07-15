# MEC8nm R77 R58 Intermediate-Checkpoint Crop Report

Canonical versionable report:

`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/evidence/mec8nm_r77_intermediate_checkpoints_crop_report_20260713.md`

R77 tested whether R58 step12000 had overfit by evaluating two discrete
checkpoints from the same training trajectory. Step8000 and step10000 record
the exact global step, seed1789, `unetr_aniso_em`, no EMA, and the same 214-key
topology as final. Their SHA256 values are
`7fc758d5afc18661241e49d16c4256889be08966eecafa33d23f562456a2c73f`
and
`2aed3b67f565233fdce696d4341d6178a84524cbb42e181ac98ab833d9b1f130`.
No weights were interpolated and no new training was performed.

SiFlow UUID `756ad839-ab71-463b-84e5-94cba285ee58`
(`mec8nm-r77-intermediate-ckpts-2f54fe03`) succeeded in 11m08s on one 8-GPU
Shanghai `med-model` pod. Both checkpoints independently returned exactly 80
finite M08/M10 WaterZ rows and empty failure ledgers. M11/M12 were absent.

Neither checkpoint passes the strict R58 gate. Step8000's best point is zero
bias, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.025127315227/0.308270133998`; M08
`0.043286870594/0.405479642617`; M10
`0.006967759861/0.211060625378`. Mean VOI improves `0.9362%`, but mean ARAND
regresses `11.4491%` and M08 ARAND/VOI regress `189.5784%/58.3114%`.

Step10000 is worse: mean ARAND/VOI
`0.030151783802/0.368098632039`, with M08
`0.052432854055/0.499382871464`. Reject both intermediate checkpoints before
full-block validation, M11/M12, deployment, or 2PB inference. Do not extend
the sweep to earlier R58 checkpoints. R58 step12000 remains the selected crop
anchor, and R57 remains the honest M08/M10 full-block anchor.
