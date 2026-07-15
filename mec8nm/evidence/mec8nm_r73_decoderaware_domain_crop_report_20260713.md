# MEC8nm R73 Decoder-Aware Domain-Pretraining Crop Report

Canonical versionable report:

`/volume/med-train/users/dchen02/code/dbMiM/mec8nm/evidence/mec8nm_r73_decoderaware_domain_crop_report_20260713.md`

R73 initialized the exact UNETR-EM encoder, decoder, and affinity head from
PublicEM-R57, then ran 4000 label-free steps on raw M01--M07/M09 crops using
masked reconstruction plus a weak raw-gradient affinity proxy. The resulting
checkpoint recorded `global_step=4000`, 219 model keys, and all 214 shared R57
keys. The supervised stage then loaded all 214 segmentation keys and ran the
unchanged R58 recipe for 12000 steps.

The successful continuation task was SiFlow UUID
`0a62d570-a2b6-4208-bfde-71df64eabc59`. It returned exactly 80 finite
M08/M10 WaterZ rows and an empty failure ledger. Two earlier launcher records
failed outside model computation because a readonly shell variable was reused
as a temporary environment assignment. The second record nevertheless
completed the 4000 pretraining steps; its byte-verified checkpoint was reused
by the successful task rather than recomputed.

The paired analyzer found zero strict R58 gate passes. R73's best-balance point
is zero bias, temperature `1.15`, threshold `.50`: mean ARAND/VOI
`0.055075340982/0.486239101228`; M08
`0.037673258094/0.443902435184`; M10
`0.072477423870/0.528575767271`. Relative to R58, mean ARAND/VOI regress
`144.2798%/56.2548%`, while M08 regresses `152.0247%/73.3128%`.

Reject R73 before full-block validation, M11/M12, deployment, or 2PB
inference. Do not repeat its raw-gradient pseudo-affinity objective. If MEC
domain pretraining is revisited, preserve the R57 decoder/head and isolate any
raw-only adaptation to the encoder so the pseudo target cannot overwrite the
segmentation calibration.
