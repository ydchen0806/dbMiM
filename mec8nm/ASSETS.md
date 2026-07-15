# MEC External Assets

## Dataset

- Local HDF5 root:
  `/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/hdf5`
- Local source snapshot:
  `/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/hf_snapshot`
- TOS mirror:
  `tos://agi-data/users/dchen02/dbmim/assets/mec8nm_20260711/MEC_8nm`
- Versioned manifest: `data/MEC_8nm/mec8nm_manifest.json`
- Versioned split contract: `data/MEC_8nm/mec8nm_splits.json`

The HDF5 directory is about 20 GB and is not copied into this workspace.

## Frozen checkpoints and results

- PublicEM R57 initializer:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/hf_release/dbmim-neuron-segmentation/weights/publicem_dbmim_r57_seed777_long20k/finetuned_latest.pt`
  SHA256:
  `9b67e337aba837ee6c26e5c4b80627ab3745fdb3fe152190d0164c6e9a2fe2ba`
- R58 training/result prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r58_seed1789_balanced_stage1`
- R58 crop evidence:
  `evidence/mec8nm_remote_results/r58_balanced_crop_20260711/`
- R57 official full-block anchor and protocol history:
  `skill/STATE.md`
- R71 paired rejection evidence:
  `evidence/mec8nm_remote_results/r71_ema_20260713/paired_analysis_vs_r58.json`
- R72 paired rejection evidence:
  `evidence/mec8nm_remote_results/r72_consistency_20260713/paired_analysis_vs_r58.json`
- R73 paired rejection evidence:
  `evidence/mec8nm_remote_results/r73_decoderaware_domain_20260713/paired_analysis_vs_r58.json`
- R74 paired rejection evidence:
  `evidence/mec8nm_remote_results/r74_encoderadapt_20260713/paired_analysis_vs_r58.json`
- R74 full remote prefix, including the 214-key finetuned checkpoint and
  audited encoder-adaptation checkpoints:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r74_seed1789_encoderadapt_stage1`
- R75 paired rejection evidence:
  `evidence/mec8nm_remote_results/r75_halo_consistency_20260713/paired_analysis_vs_r58.json`
- R75 full remote prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r75_seed1789_halo_consistency_stage1`
- R76 paired rejection evidence:
  `evidence/mec8nm_remote_results/r76_checkpoint_soup_20260713/`
- R76 full remote prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r76_r58_r71_soup_val_crop`
- R77 paired rejection evidence:
  `evidence/mec8nm_remote_results/r77_intermediate_checkpoints_20260713/`
- R77 full remote prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/eval_mec8nm_r77_r58_intermediate_val_crop`
- R78 paired rejection evidence:
  `evidence/mec8nm_remote_results/r78_paired_overlap_20260713/`
- R78 full remote prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r78_seed1789_paired_overlap_stage1`
- R79 paired rejection evidence:
  `evidence/mec8nm_remote_results/r79_topology_teacher_20260713/`
- R79 full remote prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r79_seed1789_topology_teacher_stage1`
- R80 paired rejection evidence:
  `evidence/mec8nm_remote_results/r80_instance_embedding_20260714/paired_analysis_vs_r58.json`
- R80 full remote prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r80_seed1789_instance_embedding_stage1`
- R81 paired rejection evidence:
  `evidence/mec8nm_remote_results/r81_boundary_leak_20260714/paired_analysis_vs_r58.json`
- R81 full remote prefix:
  `tos://agi-data/users/dchen02/dbmim/mec8nm_runs/finetune_mec8nm_r81_seed1789_boundary_leak_stage1`
- Pretraining and Stem-Micro decision audit:
  `evidence/mec8nm_pretraining_stem_micro_audit_20260714.md`
- Stem-Micro hzzs26--hzzs29 report and machine-readable evidence:
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/docs/dbmim_transfer/stem_micro_mec_r58_hzzs26_hzzs27_20260714.md`
  and
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/paper_tmi/data/cross_dataset_hzzs26_20260714/`
  through
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/paper_tmi/data/cross_dataset_hzzs29_20260714/`
- LoCoAgg hzzs32 signed-VI semantic-stop diagnostic:
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/docs/dbmim_transfer/locagg_signed_vi_semantic_hzzs32_plan_20260714.md`
  and
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/paper_tmi/data/signed_vi_semantic_hzzs32_20260714/`
- LoCoAgg hzzs33 AxonEM transfer diagnostic:
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/docs/dbmim_transfer/locagg_axonem_signed_vi_hzzs33_plan_20260714.md`
  and
  `/volume/med-train/users/dchen02/code/nnEM-Seg-diff-postprocess/paper_tmi/data/axonem_signed_vi_hzzs33_20260714/`
- R73 4000-step raw-only pretraining checkpoint (large, not copied here):
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/mec8nm_remote_results/r73_decoderaware_domain_20260713/pretrain/pretrained_latest.pt`
  with SHA256
  `32ba0c3aa14a59cb3fa285bb0c5875f96574a1e1bd65da2c3751eeeecf3b2262`.

## Runtime dependencies

- Offline wheels:
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/wheelhouse_dbmim`
  and
  `/volume/med-train/users/dchen02/code/dbMiM/outputs/wheelhouse_superhuman_eval`
- WaterZ source is bundled by the SiFlow staging workflow. Do not assume a
  changliu node can reach GitHub, Hugging Face, or local `/volume` paths.
- Never print or commit SiFlow/TOS credentials.

## Historical evidence retained outside this workspace

Complete task logs, saved checkpoints, staged bundles, and large intermediate
segmentations remain under the root `outputs/` tree and the recorded TOS
prefixes. The exhaustive mapping is in `skill/STATE.md`. These assets are not
part of the small versionable MEC workspace and must not be interpreted as new
validation evidence merely because they are locally accessible.
