# MEC8nm R78 Real Paired-Overlap Consistency Crop Report

R78 tested whether predictions for the same physical voxels become more robust
when trained across two real overlapping HDF5 windows. It kept the complete
R58 supervised recipe and read the second crop from the same training block at
a signed `8x32x32` offset. Shared geometry/intensity augmentation preserved a
coordinate-aligned `24x128x128` overlap. A weight-`.05` stopped-teacher loss
used only affinity targets that agreed exactly after alignment.

The first submission, UUID `cf5a12ba-a390-4f0e-9082-3743dcb5b812`, failed
before data download or training because the wrapper assigned an environment
value to a readonly shell variable. The repaired bundle passed static/hash
checks. UUID `5b34b36a-1b10-416f-a744-76eed2a1874f`
(`mec8nm-r78-real-overlap-r2-cbd8a536`) succeeded on one 8-GPU Shanghai
`med-model` pod. Startup verified CUDA 8, WaterZ, 214 R57 keys, the eight
allowed training paths, dataset size 32768, and the declared paired-overlap
configuration. Training reached step12000 in 2115 seconds.

The remote assertion and independent TOS fetch contain exactly 80 finite
WaterZ rows, samples exactly M08/M10, and an empty failure ledger. The paired
analyzer finds zero strict R58 gate passes. Best balance is bias
`(-.20,-.45,-.45)`, temperature `1.00`, threshold `.50`: mean ARAND/VOI
`0.021307430975/0.254415142584`; M08
`0.037685313511/0.346308913501`; M10
`0.004929548440/0.162521371668`.

Mean ARAND/VOI improve `5.4935%/18.2427%` versus R58, but M08 ARAND/VOI
regress `152.1054%/35.2093%`. Thirteen of 32 paired operating points improve
both mean metrics, but no point preserves both M08 metrics. At R58's exact
zero-bias, temperature-1.15, threshold-.50 setting, R78 still has M08
`0.036259477896/0.346591414479`.

Reject R78 before full-block validation, M11/M12, deployment, or 2PB
inference. Do not sweep overlap shift, consistency weight, or teacher direction.
R58 step12000 remains the crop anchor and R57 remains the M08/M10 full-block
anchor. The recurring M10 gain plus M08 regression means the next method must
explicitly preserve low-boundary/simple-volume topology using training-only
signals; more generic invariance is not sufficient.
