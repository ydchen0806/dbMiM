#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/volume/med-train/users/dchen02/code/dbMiM")
TOS = Path("/volume/med-train/users/dchen02/bin/tosutil")
CONF = Path("/volume/med-train/users/dchen02/secrets/tosutil_dchen02.conf")
SIFLOW_PY = Path("/volume/med-train/users/dchen02/envs/siflow-sdk-20260523/bin/python")
SIFLOW_ENV = Path("/volume/med-train/users/dchen02/secrets/siflow_env_dchen02.sh")
BASE = "tos://agi-data/users/dchen02/dbmim/outputs"
GROUP_KEYS = [
    "affinity_variant",
    "calibration_bias_z",
    "calibration_bias_y",
    "calibration_bias_x",
    "calibration_temperature",
    "backend",
    "threshold",
    "seed_distance",
    "boundary_threshold",
    "min_boundary",
    "score_mode",
    "rag_quantile",
    "waterz_scoring",
    "z_threshold",
    "xy_threshold",
]
METRIC_KEYS = [
    "adapted_rand_error",
    "rand_fscore",
    "rand_precision",
    "rand_recall",
    "voi_split",
    "voi_merge",
    "voi_sum",
    "affinity_dice",
    "affinity_iou",
    "inference_sec",
    "postprocess_sec",
    "metrics_sec",
]

RUNS = {
    "r9": [
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_allpretrained_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_allpretrained_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_encoderlr_allpretrained_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_encoderlr_allpretrained_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_freezeenc_allpretrained_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_freezeenc_allpretrained_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_ignore_allpretrained_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_ignore_allpretrained_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_ignore_scratch_r9", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_ignore_scratch_r9"),
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_allpretrained_r10", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_allpretrained_r10"),
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_scratch_r10", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_scratch_r10"),
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_ignore_allpretrained_r10", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_ignore_allpretrained_r10"),
    ],
    "r11": [
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_pure_allpretrained_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_pure_allpretrained_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_shwmse_pure_scratch_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_pure_scratch_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_shwmse_mix_allpretrained_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_shwmse_mix_allpretrained_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_shwmse_mix_scratch_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_shwmse_mix_scratch_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_aug_encoderlr_allpretrained_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_aug_encoderlr_allpretrained_r11"),
        ("finetune_cremi_real_unetr_aniso_superhuman_bce_aug_scratch_r11", "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_aug_scratch_r11"),
    ],
    "r12": [
        ("finetune_cremi_real_unetr_aniso_em_bce_encoderlr_allpretrained_r12", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_allpretrained_r12"),
        ("finetune_cremi_real_unetr_aniso_em_bce_encoderlr_scratch_r12", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_scratch_r12"),
    ],
    "r13": [
        ("finetune_cremi_real_unetr_aniso_em_shwmse_allpretrained_r13", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_allpretrained_r13"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_scratch_r13", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_scratch_r13"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_allpretrained_r13", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_allpretrained_r13"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_scratch_r13", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_scratch_r13"),
    ],
    "r14q": [
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_rank_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_rank_allpretrained_r14q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_calib_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_calib_allpretrained_r14q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_allpretrained_r14q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_allpretrained_r14q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws15_bcar_rank_allpretrained_r14q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws15_bcar_rank_allpretrained_r14q"),
    ],
    "r14": [
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_rank_allpretrained_r14", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_rank_allpretrained_r14"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_calib_allpretrained_r14", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_calib_allpretrained_r14"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_mempretrained_r14", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_mempretrained_r14"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_bcar_mempretrained_r14", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_mempretrained_r14"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_mempretrained_r14", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_mempretrained_r14"),
    ],
    "r15q": [
        ("finetune_cremi_real_unetr_aniso_em_shwmse_longaff_mempretrained_r15q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_mempretrained_r15q_arch_bench"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_longaff_lsd_mempretrained_r15q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_lsd_mempretrained_r15q_arch_bench"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_longaff_bcar2_mempretrained_r15q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_bcar2_mempretrained_r15q_arch_bench"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_allpretrained_r14q", "eval_cremi_arch_explore_postprocess_r15q"),
    ],
    "r16q": [
        ("finetune_cremi_real_unetr_aniso_em_shwmse_longaff_publicem_r16q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_publicem_r16q_arch_bench"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_longaff_scratch_r16q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_scratch_r16q_arch_bench"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_publicem_r16q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_publicem_r16q_arch_bench"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_scratch_r16q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_scratch_r16q_arch_bench"),
    ],
    "r16q_waterz": [
        ("finetune_cremi_real_unetr_aniso_em_shwmse_longaff_publicem_r16q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_publicem_r16q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_longaff_scratch_r16q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_scratch_r16q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_publicem_r16q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_publicem_r16q"),
        ("finetune_cremi_real_unetr_aniso_em_shwmse_maws_bcar_rank_scratch_r16q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_scratch_r16q"),
    ],
    "r17q": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_plainmae_r23q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_plainmae_r23q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_bcar_rank_publicem_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_bcar_rank_publicem_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_bcar_rank_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_bcar_rank_scratch_r17q"),
    ],
    "r23_plainmae": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_plainmae_r23q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_plainmae_r23q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r17q"),
    ],
    "r24_dbmim_vs_mae": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_decoderaware_r24q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_decoderaware_r24q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_plainmae_r23q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_plainmae_r23q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r17q"),
    ],
    "r26_encoderonly": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_decoderaware_encoderonly_r26q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_decoderaware_encoderonly_r26q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_decoderaware_r24q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_decoderaware_r24q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_plainmae_r23q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_plainmae_r23q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r17q"),
    ],
    "r29_edgemask_vs_mae": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_edgemask_r29q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_edgemask_r29q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_plainmae_r23q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_plainmae_r23q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r17q"),
    ],
    "r25_early3k": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_decoderaware_r24q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_decoderaware_r24q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_plainmae_r23q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_plainmae_r23q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_scratch_r17q"),
    ],
    "r26_encoderonly_early3k": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_decoderaware_encoderonly_r26q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_decoderaware_encoderonly_r26q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_decoderaware_r24q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_decoderaware_r24q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_plainmae_r23q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_plainmae_r23q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_publicem_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_early3k_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_scratch_r17q"),
    ],
    "r20q": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_r20q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fullem_r20q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_plainmae_r23q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fullem_plainmae_r23q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r17q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_bcar_rank_fullem_r20q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_bcar_rank_fullem_r20q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_bcar_rank_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_bcar_rank_scratch_r17q"),
    ],
    "r23_plainmae_full": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_r20q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fullem_r20q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_plainmae_r23q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fullem_plainmae_r23q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r17q"),
    ],
    "r20q_dpp": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_r20q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fullem_r20q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_dpp_fullem_r20q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_dpp_fullem_r20q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r17q"),
    ],
    "r20q_learned_calib": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_r20q", "eval_cremi_learned_affinity_calibration_r20q"),
    ],
    "r27_fast_learned_postprocess": [
        (
            "finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q",
            "eval_cremi_fast_learned_postprocess_r17q/publicem",
        ),
        (
            "finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q",
            "eval_cremi_fast_learned_postprocess_r17q/scratch",
        ),
    ],
    "r28_fast_learned_postprocess_screen": [
        (
            "finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q",
            "eval_cremi_fast_learned_postprocess_r28q/publicem",
        ),
        (
            "finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r17q",
            "eval_cremi_fast_learned_postprocess_r28q/scratch",
        ),
    ],
    "r21q": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_scratch_r21q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r21q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_decoderaware_r21q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_decoderaware_r21q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_dpp_scratch_r21q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_dpp_scratch_r21q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_decoderaware_dpp_r21q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_decoderaware_dpp_r21q"),
    ],
    "r18q": [
        ("finetune_cremi_real_unetr_aniso_em_mse_longaff_publicem_r18q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_longaff_publicem_r18q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_longaff_scratch_r18q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_longaff_scratch_r18q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_longaff_bcar_rank_publicem_r18q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_longaff_bcar_rank_publicem_r18q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_longaff_bcar_rank_scratch_r18q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_longaff_bcar_rank_scratch_r18q"),
    ],
    "r19q": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_context48_publicem_r19q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_context48_publicem_r19q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_context48_scratch_r19q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_context48_scratch_r19q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_fs48_publicem_r19q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fs48_publicem_r19q"),
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_fs48_scratch_r19q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fs48_scratch_r19q"),
    ],
    "r17q_fine": [
        ("finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r17q", "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_r17q_fine"),
    ],
}

SIFLOW_UUIDS = {
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_allpretrained_r9": "99ab7d58-8886-430e-86fa-92c9d4a0fcae",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_encoderlr_allpretrained_r9": "33638465-d404-4229-b150-cdbf401e8159",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_freezeenc_allpretrained_r9": "7ad21665-e4a3-4766-903e-d89088429919",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_ignore_allpretrained_r9": "37fe0514-76a0-4d74-adb5-c51e4b1d7dcc",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_ignore_scratch_r9": "885406a6-6572-46a0-8bbe-da607132b26a",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_allpretrained_r10": "6c7675a1-7661-4363-9c65-90e1f2e7129e",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_scratch_r10": "af35134e-a7ea-4f27-bfeb-4777dd48ae5b",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_ignore_allpretrained_r10": "20239e88-03f5-4269-a348-da79dd2adbb4",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_pure_allpretrained_r11": "f09b5b01-ef57-4490-8c5b-4184757cbd01",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_shwmse_pure_scratch_r11": "a3eedbdf-70d0-496f-9148-1bc6f082a53d",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_shwmse_mix_allpretrained_r11": "71106220-8efa-48af-b512-c3141e76831d",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_shwmse_mix_scratch_r11": "bc6d23ed-c43d-4d9b-a61b-8ce955a1e64e",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_aug_encoderlr_allpretrained_r11": "f2e6d5da-66c6-4ae2-a66f-23cb139f3d0b",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_bce_aug_scratch_r11": "bd71349d-b066-4cd2-90c9-5e3145ebdb1f",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_allpretrained_r12": "70c97464-7179-41b8-80f7-9ccf4f94ff25",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_bce_encoderlr_scratch_r12": "bfc6ca92-13fa-4d1e-b234-fab4b8e3c14c",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_allpretrained_r13": "4be6ecf8-5053-4693-bac8-7b5f15aa2df9",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_scratch_r13": "e0a7eb92-bb68-49e6-bcc3-5e992347b98f",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_allpretrained_r13": "64c9f7f4-5f0f-4658-b4a6-73acb32e4e5e",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_scratch_r13": "28c140b6-589d-4a62-9ffb-ddea4b0eb2c7",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_rank_allpretrained_r14q": "fa076c76-f3bf-4eac-91ac-c8f4a1677062",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_calib_allpretrained_r14q": "cb95420a-2482-48a6-a3bc-9cb86c51c8d3",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_allpretrained_r14q": "38b18ca3-d4c8-4fd2-94d9-632f590d92ce",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_allpretrained_r14q": "2f63a6dc-c7a6-4e8b-97de-34ab80985b40",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws15_bcar_rank_allpretrained_r14q": "30cb7b4a-ac52-402a-9085-c97749ab5f2b",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_mempretrained_r14": "f4c00499-19a5-4fb2-99a8-99adf54bad4d",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_bcar_mempretrained_r14": "d23472e9-567f-4bd3-9e04-24f450dbab85",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_mempretrained_r14": "9494b4fa-f6e6-410c-90ba-052bb8e70d01",
    "eval_cremi_arch_explore_postprocess_r15q": "bcac3b16-9896-4114-84bd-a70f854e2a8e",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_mempretrained_r15q_arch_bench": "73a40fd4-287b-4bfc-98b6-c57aa6a38c1a",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_lsd_mempretrained_r15q_arch_bench": "08fad37f-4257-4f4f-9e9b-ad86c4b7f93f",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_bcar2_mempretrained_r15q_arch_bench": "cb58f241-4fac-490f-b958-1ca6376bfb14",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_publicem_r16q_arch_bench": "11df9593-273a-4456-8da4-6f844b1d8292",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_scratch_r16q_arch_bench": "198e39cc-c2aa-4ab1-82f3-edcc15e6917f",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_publicem_r16q_arch_bench": "b8e5f3dc-9047-48a2-9b90-dd439f0265c7",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_scratch_r16q_arch_bench": "997ba3ec-77f6-41f0-851b-83f770427cfd",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_publicem_r16q": "d820080c-2c13-4278-980d-155add949017",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_longaff_scratch_r16q": "418b6d25-06df-4820-8594-53e7a55e9b9c",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_publicem_r16q": "a9456991-13c5-41cc-af72-f3f4427b3f26",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_shwmse_maws_bcar_rank_scratch_r16q": "b51f6415-2bae-4e34-8b95-50994f092496",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_r17q": "68013c0c-b712-4c93-9b46-984c69f812ac",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_scratch_r17q": "1d218802-03e8-4121-8101-09637a775089",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_bcar_rank_publicem_r17q": "8f9e4a5e-729f-42b5-8516-d0c0784fb8cb",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_bcar_rank_scratch_r17q": "d208dea6-9b2c-4dc3-ae94-a5104ad38d39",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_longaff_publicem_r18q": "d35064aa-e7d7-4819-95b2-777e53c94c50",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_longaff_scratch_r18q": "b1d1d975-409d-4204-85d2-03fb47f7068c",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_longaff_bcar_rank_publicem_r18q": "deac4066-3261-4739-8887-f3d3c4629aae",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_longaff_bcar_rank_scratch_r18q": "dffb90bf-1526-4a8a-8a65-ee7d36065824",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_r17q_fine": "4503d96c-9b52-4974-8e5e-7ee08bc21362",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_context48_publicem_r19q": "e9e01802-c98e-466b-b3cf-f5cf1b0edbbd",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_context48_scratch_r19q": "6655b114-66b7-4a66-8efc-d55ca6d2dfcc",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fs48_publicem_r19q": "77d049b8-76a1-4369-84c7-d02bc361851e",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fs48_scratch_r19q": "2ade7cb2-667c-4410-b25d-cba312fc112e",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_fullem_r20q": "0e29a6b1-26bb-45c2-813d-db8efb266d21",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_bcar_rank_fullem_r20q": "eb647c53-f408-4a5f-99c9-ead3d7b1f2df",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_dpp_fullem_r20q": "e52d773c-abae-481f-9da7-c3b34cab38a8",
    "eval_cremi_learned_affinity_calibration_r20q": "db054938-053d-4d25-9849-d616d66ff57e",
    "eval_cremi_fast_learned_postprocess_r17q/publicem": "b86d2af4-09ca-414e-a493-42e1d9c039e1",
    "eval_cremi_fast_learned_postprocess_r17q/scratch": "b86d2af4-09ca-414e-a493-42e1d9c039e1",
    "eval_cremi_fast_learned_postprocess_r28q/publicem": "a957727f-8dc3-4b4c-a66a-975957e03ed6",
    "eval_cremi_fast_learned_postprocess_r28q/scratch": "a957727f-8dc3-4b4c-a66a-975957e03ed6",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_decoderaware_r24q": "628faa9d-4e5a-4b19-98bd-555bef604302",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_publicem_decoderaware_encoderonly_r26q": "faba607c-b53c-4e45-9de2-798eb8462612",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_decoderaware_r24q": "191db4dd-9949-4e08-8b79-838b34c19755",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_plainmae_r23q": "d650fd4d-4447-44dc-b25e-64c3cb6b65d0",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_r17q": "2b6dda20-de13-4ca4-af18-05af44a9988f",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_scratch_r17q": "ffb9c6a4-935a-4dc4-aee9-87cf62d36c89",
    "eval_cremi_unetr_aniso_superhuman_calibration_official_abc_em_mse_maws_early3k_publicem_decoderaware_encoderonly_r26q": "992d7ef4-dcca-43d5-9a01-1ba7ca9477cd",
}


def tos_cp(src: str, dst: Path, timeout: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [str(TOS), "cp", src, str(tmp), "-f", "-bt=fns", f"-conf={CONF}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


def tos_exists(uri: str, timeout: int) -> bool:
    try:
        proc = subprocess.run(
            [str(TOS), "ls", uri, "-bt=fns", f"-conf={CONF}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != 0:
        return False
    return "File number is: 0" not in proc.stdout


def read_last_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip().endswith("}"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows[-1] if rows else None


def _load_env_exports(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            env[key] = value
    return env


def parse_metric_rows_from_siflow(uuid: str, region: str, cluster: str) -> list[dict]:
    code = r"""
import json
import sys
from siflow import SiFlow

uuid, region, cluster = sys.argv[1:4]
client = SiFlow(region=region, cluster=cluster)
try:
    logs = client.tasks.query_logs(uuid, limit=2000, sort_order="asc")
except Exception:
    logs = client.tasks.query_logs(uuid, limit=2000, sort_order="desc")
print(json.dumps([str(item.content) for item in logs.logs]))
"""
    env = os.environ.copy()
    env.update(_load_env_exports(SIFLOW_ENV))
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)
    proc = subprocess.run(
        [str(SIFLOW_PY), "-c", code, uuid, region, cluster],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-500:] or proc.stdout.strip()[-500:])
    rows = []
    contents = json.loads(proc.stdout)
    if contents is None:
        contents = []
    for content in contents:
        for line in str(content).splitlines():
            start = line.find("{'sample'")
            if start < 0 or "voi_sum" not in line or "adapted_rand_error" not in line:
                continue
            try:
                row = ast.literal_eval(line[start:].strip())
            except Exception:
                continue
            if isinstance(row, dict) and "voi_sum" in row:
                rows.append(row)
    dedup = []
    seen = set()
    for row in rows:
        key = (row.get("sample"), *(row.get(name) for name in GROUP_KEYS))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
    return dedup


def _mean_metric(rows: list[dict], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return sum(values) / len(values) if values else float("nan")


def aggregate_metric_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[object, ...], list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(name, "") for name in GROUP_KEYS), []).append(row)
    per_backend_threshold = []
    for key_tuple, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        mean_row = dict(zip(GROUP_KEYS, key_tuple))
        mean_row["n"] = len(group_rows)
        for metric in METRIC_KEYS:
            mean_row[metric] = _mean_metric(group_rows, metric)
        per_backend_threshold.append(mean_row)

    threshold_grouped: dict[float, list[dict]] = {}
    for row in rows:
        threshold_grouped.setdefault(float(row.get("threshold", 0.0)), []).append(row)
    per_threshold = []
    for threshold, group_rows in sorted(threshold_grouped.items()):
        mean_row = {"threshold": threshold, "n": len(group_rows)}
        for metric in METRIC_KEYS:
            mean_row[metric] = _mean_metric(group_rows, metric)
        per_threshold.append(mean_row)
    return per_backend_threshold, per_threshold


def write_siflow_fallback_summary(
    eval_name: str,
    path: Path,
    *,
    region: str,
    cluster: str,
) -> bool:
    uuid = SIFLOW_UUIDS.get(eval_name)
    if not uuid:
        return False
    try:
        rows = parse_metric_rows_from_siflow(uuid, region, cluster)
    except Exception as exc:
        print("SIFLOW_FALLBACK_ERROR", eval_name, type(exc).__name__, str(exc)[:160])
        return False
    if not rows:
        return False
    per_backend_threshold, per_threshold = aggregate_metric_rows(rows)
    if not per_backend_threshold:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "siflow_stdout_fallback",
        "uuid": uuid,
        "num_records": len(rows),
        "sample_names": sorted({str(row.get("sample")) for row in rows}),
        "records": rows,
        "per_backend_threshold": per_backend_threshold,
        "per_threshold": per_threshold,
        "best_by_voi_sum": min(per_backend_threshold, key=lambda row: row.get("voi_sum", float("inf"))),
        "best_by_adapted_rand": min(
            per_backend_threshold,
            key=lambda row: row.get("adapted_rand_error", float("inf")),
        ),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return True


def _is_complete_summary(eval_name: str, summary: dict) -> bool:
    """Return True only when a summary covers the intended evaluation scope."""

    samples = {str(name) for name in summary.get("sample_names", [])}
    if not samples:
        records = summary.get("records") or []
        samples = {str(row.get("sample") or row.get("sample_name")) for row in records if isinstance(row, dict)}
    if not samples:
        sample_records = summary.get("sample_records") or []
        samples = {
            str(row.get("sample") or row.get("sample_name"))
            for row in sample_records
            if isinstance(row, dict)
        }
    if "official_abc" in eval_name:
        expected = {
            "sample_A_20160501.hdf",
            "sample_B_20160501.hdf",
            "sample_C_20160501.hdf",
        }
        if not expected.issubset(samples):
            return False
        try:
            num_records = int(summary.get("num_records") or 0)
        except (TypeError, ValueError):
            num_records = 0
        return num_records >= 60
    return bool(samples)


def _is_stale_summary(eval_name: str, summary: dict) -> bool:
    """Detect local stdout-fallback summaries from superseded SiFlow UUIDs."""

    if summary.get("source") != "siflow_stdout_fallback":
        return False
    expected_uuid = SIFLOW_UUIDS.get(eval_name)
    actual_uuid = summary.get("uuid")
    return bool(expected_uuid and actual_uuid and expected_uuid != actual_uuid)


def summarize(group: str) -> int:
    root = ROOT / "outputs" / "tos_fetch" / group
    done = 0
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {group}")
    for train, eval_name in RUNS[group]:
        train_log = root / train / "finetune_log.jsonl"
        rec = read_last_jsonl(train_log)
        if rec:
            print(
                "LOG",
                train,
                "step",
                rec.get("step"),
                "loss",
                rec.get("train_loss"),
                "main",
                rec.get("train_main_loss"),
                "valid",
                rec.get("train_valid_fraction"),
            )
        summary_path = root / eval_name / "cremi_segmentation_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                by_voi = summary.get("best_by_voi_sum", {})
                by_rand = summary.get("best_by_adapted_rand", {})
                stale = _is_stale_summary(eval_name, summary)
                complete = (not stale) and _is_complete_summary(eval_name, summary)
                if complete:
                    done += 1
                print(
                    "SUMMARY" if complete else ("STALE" if stale else "PARTIAL"),
                    eval_name,
                    "source",
                    summary.get("source", "tos"),
                    "uuid",
                    summary.get("uuid", ""),
                    "records",
                    summary.get("num_records", "?"),
                    "samples",
                    ",".join(summary.get("sample_names", [])),
                    "VOI",
                    by_voi.get("voi_sum"),
                    "ARAND@VOI",
                    by_voi.get("adapted_rand_error"),
                    "bestARAND",
                    by_rand.get("adapted_rand_error"),
                    "VOI@ARAND",
                    by_rand.get("voi_sum"),
                )
            except Exception as exc:
                print("SUMMARY_ERROR", eval_name, type(exc).__name__, str(exc)[:160])
    print(f"{group}_done_summaries={done}/{len(RUNS[group])}", flush=True)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(RUNS), required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--logs", action="store_true", help="Also download finetune logs; summaries are always checked.")
    parser.add_argument(
        "--siflow-fallback",
        action="store_true",
        help="If TOS summary download is missing or slow, rebuild summaries from SiFlow stdout logs.",
    )
    parser.add_argument("--siflow-region", default="cn-shanghai")
    parser.add_argument("--siflow-cluster", default="changliu")
    args = parser.parse_args()

    while True:
        for train, eval_name in RUNS[args.group]:
            root = ROOT / "outputs" / "tos_fetch" / args.group
            if args.logs:
                log_uri = f"{BASE}/{train}/finetune_log.jsonl"
                if tos_exists(log_uri, args.timeout):
                    tos_cp(log_uri, root / train / "finetune_log.jsonl", args.timeout)
            summary_uri = f"{BASE}/{eval_name}/cremi_segmentation_summary.json"
            if tos_exists(summary_uri, args.timeout):
                tos_cp(summary_uri, root / eval_name / "cremi_segmentation_summary.json", args.timeout)
            summary_path = root / eval_name / "cremi_segmentation_summary.json"
            if args.siflow_fallback:
                # SiFlow stdout fallback can be partial while post-processing is
                # still running. Rebuild it every poll until the canonical TOS
                # summary appears, so later waterz/RAG rows are not hidden by an
                # early graph-CC-only snapshot.
                write_siflow_fallback_summary(
                    eval_name,
                    summary_path,
                    region=args.siflow_region,
                    cluster=args.siflow_cluster,
                )
        done = summarize(args.group)
        if args.once or done >= len(RUNS[args.group]):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
