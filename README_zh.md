# dbMiM 神经元分割

[![English README](https://img.shields.io/badge/README-English-blue)](README.md)
[![Hugging Face 权重](https://img.shields.io/badge/HuggingFace-weights-yellow)](https://huggingface.co/cyd0806/dbmim-neuron-segmentation)

这个仓库是当前 dbMiM 神经元分割实验的清理版代码，主要面向 CREMI
复现、预训练收益验证、UNETR 结构改进和 VOI/ARAND 评测。当前维护的主线是：

- 在未标注 EM 体数据上做 dbMiM / MAE 风格自监督预训练；
- 在 CREMI 上用各向异性 3D UNETR 做 affinity 微调；
- 在 CREMI A/B/C 全体积上用 VOI 和 adapted Rand error (ARAND) 评测；
- 用 waterz 后处理，并做 calibration / threshold sweep。

旧的私有集群提交器、历史实验报告、缓存字节码、旧 dataloader、旧模型和临时脚本已经从 Git 中删除。大数据、权重、报告和本地实验输出会被 `.gitignore` 忽略。

## 当前方法

当前最好方法不是最早的简单复现，而是经过消融后稳定有效的一条主线：

1. **各向异性 UNETR backbone**：使用 `UNETRAnisotropicAffinityNet`，
   输入 crop 为 `32x160x160`，`patch_size=(4,16,16)`，Transformer hidden
   states 作为 UNETR skip，decoder 分阶段上采样，并在最后 decoder 前加入
   z-only 的各向异性 transition。
2. **EM 体数据上的 dbMiM 预训练**：先训练 ViT/MAE encoder，目标包括 masked
   reconstruction、membrane-aware weighting 和轻量 structure consistency loss，
   再把 encoder 权重加载到 UNETR。
3. **MSE + MAWS 微调**：CREMI instance label 被转换成 z/y/x 三通道最近邻
   affinity。当前获胜微调方案使用纯 MSE，加 membrane-aware spatial weighting
   (MAWS)，通道权重为 `[1.35, 1.0, 1.0]`，并保证图像和 label 的几何增强同步。
4. **official-style waterz 评测**：在 CREMI A/B/C 全体积上评测，使用
   `ignore_label=0`，CREMI 风格 XY boundary ignore distance `1`，z 方向为 `0`，
   并做 logit calibration bias 与 waterz threshold sweep。

当前最好的受控结果是 **R48**：它使用 R16 线的 publicEM dbMiM 预训练
encoder，固定微调 seed 为 `309`，并把 MSE+MAWS 微调延长到 20k step。在
CREMI A/B/C 全体积评测口径下，R48 达到 `VOI=0.962154`、
`ARAND=0.178252`，优于同 seed 的 plain MAE 控制组，也优于早期 12k 微调。

full-data 方向最新有价值的正结果是：**fullEM 数据上的固定 mixed
edge/random masking (R33)** 相比 scratch、旧 fullEM dbMiM、纯 edge mask 和
fullEM plain MAE 都有提升。R33 仍是当前最好的 fullEM recipe，而 R48 是当前
全局最好 checkpoint。

当前推荐的 fullEM R33 主线**没有使用强化学习 mask policy**，而是固定的
mixed edge/random masking。RL-style decision module 作为消融继续保留，并在
R51/R52 线做稳定化：R51 是不健康的 policy collapse，R52 改成 constrained
adaptive prior sampling。R52/R53 下游微调已经完成但为负结果，所以当前主线
仍然是 R48 风格 publicEM dbMiM 预训练加更长 MSE+MAWS 微调。

## 权重

训练好的预训练权重和微调权重已经上传到：

**https://huggingface.co/cyd0806/dbmim-neuron-segmentation**

| 模型 | HF 路径 | 用途 |
|---|---|---|
| PublicEM dbMiM R48 微调 | `weights/publicem_dbmim_r48_seed309_long20k/finetuned_latest.pt` | 当前最好分割权重；等 TOS artifact 可拉回后同步到 HF |
| PublicEM dbMiM R17 预训练 | `weights/publicem_dbmim_r17/pretrained_latest.pt` | ViT/dbMiM encoder 初始化权重 |
| PublicEM dbMiM R17 微调 | `weights/publicem_dbmim_r17/finetuned_latest.pt` | 较早 publicEM 分割权重 |
| FullEM mixed-mask dbMiM R33 预训练 | `weights/fullem_mixedmask_dbmim_r33/pretrained_latest.pt` | 推荐的 full-data dbMiM 预训练权重 |
| FullEM mixed-mask dbMiM R33 微调 | `weights/fullem_mixedmask_dbmim_r33/finetuned_latest.pt` | 推荐的 full-data 分割权重 |

预训练 checkpoint 包含 masked-image-modeling 的 encoder/decoder 状态。微调时只把兼容的 encoder 前缀加载到各向异性 UNETR：`pos_embed`、`patch_embed`、`encoder_blocks`、`norm`。微调 checkpoint 是完整 affinity 分割网络。

## 数据

### 有标注 CREMI 数据

微调和评测使用公开的 CREMI 2016 training volumes：

```text
data/CREMI/sample_A_20160501.hdf
data/CREMI/sample_B_20160501.hdf
data/CREMI/sample_C_20160501.hdf
```

raw key 是 `volumes/raw`，instance label key 是
`volumes/labels/neuron_ids`。

### 预训练数据

使用过两套未标注 EM 预训练数据。

| 名称 | 内容 | 配置示例 |
|---|---|---|
| publicEM | CREMI raw + public ISBI 2012 + SNEMI3D raw volumes | `configs/pretrain_public_em_membrane_r16.yaml`, `configs/pretrain_public_em_plain_mae_r23.yaml` |
| fullEM | CREMI raw + `cyd0806/EM_pretrain_data` 的 FAFB、FIB-25、Kasthuri、MitoEM、MB-MOC | `configs/pretrain_em_full_mixedmask_dbmim_r33.yaml`, `configs/pretrain_em_full_plain_mae_r23.yaml` |

本仓库没有使用 CREMI challenge hidden test labels。

### 训练和评测划分

本文档中的结果是 **public labeled CREMI A/B/C training volumes 上的 official-style validation**，不是 challenge server hidden-test 结果。

评测口径如下：

- 监督微调时从 CREMI A/B/C 中随机采样 crop；
- 评测时对 A/B/C 三个体数据做 full-volume sliding-window inference；
- metric 计算时启用 CREMI-style boundary ignore：`xy=1`，`z=0`；
- sweep calibration bias 和 waterz threshold；
- 报告 A/B/C 聚合后的 `voi_sum` 和 `adapted_rand_error`。

这个划分很小，但适合当前目标：在同一各向异性 UNETR、同一微调 recipe、同一后处理下，对比 dbMiM 预训练、scratch 和 plain MAE 控制组。

## 结果

VOI 和 ARAND 都是越低越好。`ARAND at best VOI` 是 VOI 最优阈值对应的
ARAND；`Best ARAND` 是单独按 ARAND 选出的最优阈值，因为 VOI 和 ARAND 有时会偏好不同后处理点。

### PublicEM 预训练

| 实验 | VOI | ARAND at best VOI | Best ARAND | 结论 |
|---|---:|---:|---:|---|
| R48 publicEM dbMiM, seed309, 20k finetune | **0.962154** | **0.178252** | **0.178252** | 当前全局最好结果 |
| R45 publicEM dbMiM, seed309, 12k finetune | 0.986481 | 0.186187 | 0.186187 | 强 same-seed dbMiM 结果 |
| R17 publicEM random-mask dbMiM | 1.002919 | 0.188832 | 0.188832 | 较早 publicEM dbMiM 结果 |
| R23 publicEM random-mask plain MAE | 1.027073 | 0.192763 | 0.189247 | matched MAE baseline |
| R47 publicEM plain MAE, seed309 | 1.043065 | 0.190743 | 0.190743 | R45/R48 的 same-seed MAE 控制组 |
| R29 publicEM pure edge-mask dbMiM | 1.033564 | 0.186827 | **0.186827** | publicEM 最好 ARAND，但 VOI 较差 |
| R32 publicEM fixed mixed-mask dbMiM | 1.046538 | 0.206256 | 0.193183 | 相比 R17/R23 为负 |
| R34 publicEM adaptive mixed dbMiM | 1.067471 | 0.205437 | 0.200604 | adaptive 为负 |
| R30 publicEM pure edge-mask plain MAE | 1.077594 | 0.203182 | 0.198562 | edge-mask MAE 控制组 |
| R17 scratch UNETR | 1.095164 | 0.213401 | 0.210442 | scratch 控制组 |

关键差值：

- R48 dbMiM 相比 same-seed publicEM plain MAE R47：VOI 降低 `0.0809`，ARAND 降低 `0.0125`。
- R16 seed309 线从 R45 的 12k 微调延长到 R48 的 20k 微调后，VOI 降低
  `0.0243`，ARAND 降低 `0.0079`。
- R17 dbMiM 相比 matched publicEM plain MAE R23：VOI 降低 `0.0242`，best ARAND 约降低 `0.0004`。
- R29 edge-mask dbMiM 相比同 mask 的 plain MAE R30：VOI 降低 `0.0440`，best ARAND 降低 `0.0117`，但 VOI 不如 R17/R23。

### FullEM 预训练

| 实验 | VOI | ARAND at best VOI | Best ARAND | 结论 |
|---|---:|---:|---:|---|
| R33 fullEM fixed mixed-mask dbMiM | **1.039372** | **0.191216** | **0.190932** | 最好 fullEM 结果 |
| R31 fullEM pure edge-mask dbMiM | 1.055438 | 0.195125 | 0.195125 | 正收益，但弱于 R33 |
| R20 fullEM old dbMiM | 1.085331 | 0.195722 | 0.195722 | 旧 fullEM baseline |
| R35 fullEM adaptive mixed dbMiM | 1.089639 | 0.205551 | 0.205551 | 弱于 R33/R31/R20 |
| R17 scratch UNETR | 1.095164 | 0.213401 | 0.210442 | scratch 控制组 |
| R23 fullEM plain MAE | 1.440684 | 0.281216 | 0.281216 | fullEM MAE 明显为负 |

关键差值：

- R33 fullEM mixed-mask dbMiM 相比 fullEM plain MAE R23：VOI 降低 `0.4013`，best ARAND 降低 `0.0903`。
- R33 相比 scratch：VOI 降低 `0.0558`，best ARAND 降低 `0.0195`。
- R33 相比旧 fullEM R20：VOI 约降低 `0.0460`。
- R33 仍然略差于 publicEM R17 的最好 VOI (`1.039372` vs `1.002919`)，所以 fullEM recipe 是当前最好 fullEM 方案，但还不是全局最好 checkpoint。

### Adaptive Masking

R34/R35 测试了每个 crop 自适应选择 mask ratio 和 edge fraction 的 mixed masking
policy。这个方向目前没有带来提升。40k step 之后，policy 基本收敛到
`sampled_mask_ratio=0.75`；R34 平均 `edge_fraction=0.4456`，R35 平均
`edge_fraction=0.3322`。因此当前 adaptive policy 作为负消融保留，不作为推荐方法。

新的 R51/R52 线没有简单加大 reward，而是改 policy 稳定性。R51 仍然 collapse
到无信息策略；R52 限制 mask-ratio / edge-fraction bins，使用较小 edge-proxy
reward、KL-to-prior 正则、advantage clip/normalize，并在 warmup 后冻结 policy。
R52 的预训练诊断比 R51 更健康，但下游微调是负结果：R52 为 `VOI=1.056275`、
`ARAND=0.194687`；`membrane_weight=1.0` 的 R53 为 `VOI=1.080240`、
`ARAND=0.209107`。它们都差于 R48，因此只保留为负消融。

## 训练策略

### 预训练

代表性命令：

```bash
python train_pretrain.py \
  --config configs/pretrain_em_full_mixedmask_dbmim_r33.yaml
```

主要设置：

| 设置 | 值 |
|---|---|
| Crop | `32x160x160` |
| Patch size | `4x16x16` |
| Encoder | ViT, `embed_dim=192`, `depth=6`, `heads=6` |
| Mask ratio | `0.75` |
| R33 mask strategy | `edge_random_mix`, `edge_mask_fraction=0.5`, `edge_mask_power=1.25` |
| dbMiM loss | reconstruction + structure loss `0.2` + membrane weighting `1.35` |
| Batch size | 每张 GPU 2 |
| Schedule | 160k optimizer steps, AdamW, lr `1.5e-4`, weight decay `0.05`, AMP |

plain MAE 控制组使用 `architecture: plain_mae`，`structure_weight: 0.0`，
`membrane_weight: 0.0`，其它数据、crop、模型大小、mask ratio 和 schedule 保持匹配。

### 微调

代表性命令：

```bash
python train_finetune.py \
  --config configs/finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_mixedmask_r33q.yaml
```

主要设置：

| 设置 | 值 |
|---|---|
| Backbone | `unetr_aniso_em` |
| 输出 | 3 个 affinity channel：z、y、x |
| Crop | `32x160x160` |
| Loss | MSE + MAWS，当前获胜 recipe 不用 BCE/Dice |
| Label 处理 | 图像/label 几何增强同步，2D border widening radius 1 |
| Batch size | 每张 GPU 2 |
| Schedule | 标准消融为 12k optimizer steps；R48 使用 20k steps。lr `8e-5`, encoder lr `1e-5`, weight decay `0.01`, AMP |
| 预训练加载前缀 | `pos_embed`, `patch_embed`, `encoder_blocks`, `norm` |

### 评测

代表性 full-volume 命令：

```bash
python scripts/evaluate_cremi_segmentation.py \
  --config configs/finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_mixedmask_r33q.yaml \
  --checkpoint outputs/finetune_cremi_real_unetr_aniso_em_mse_maws_fullem_mixedmask_r33q/finetuned_latest.pt \
  --data-dir data/CREMI \
  --output-dir outputs/eval_cremi_r33_waterz_abc \
  --crop-size 0 0 0 \
  --stride 16 80 80 \
  --backends waterz \
  --thresholds 0.35 0.40 0.45 0.50 0.55 \
  --calibration-biases -0.50 -1.00 -1.00 -0.25 -0.50 -0.50 0.0 0.0 0.0 \
  --metric-backend skimage \
  --ignore-label 0 \
  --cremi-boundary-ignore-distance-xy 1 \
  --cremi-boundary-ignore-distance-z 0 \
  --max-samples 0 \
  --device cuda
```

评测会输出：

```text
cremi_segmentation_records.json
cremi_segmentation_metrics.csv
cremi_segmentation_summary.json
```

主表使用 `best_by_voi_sum` 作为 VOI 汇报点，同时检查 `best_by_adapted_rand`
以避免只看单一阈值。

## 快速开始

安装依赖：

```bash
pip install -r requirements-dbMIM.txt
```

运行 synthetic smoke test：

```bash
bash scripts/run_smoke.sh
```

编译当前维护入口：

```bash
python -m py_compile \
  dbmim/*.py \
  train_pretrain.py \
  train_finetune.py \
  scripts/download_data.py \
  scripts/inspect_hdf5.py \
  scripts/prepare_public_em_pretrain_data.py \
  scripts/prepare_em_pretrain_data.py \
  scripts/evaluate_cremi_segmentation.py \
  scripts/evaluate_cremi_blockwise_scale.py
```

用 `huggingface_hub` 下载权重：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="cyd0806/dbmim-neuron-segmentation",
    local_dir="outputs/hf_weights",
    allow_patterns=["weights/**", "configs/**"],
)
```

## 仓库结构

```text
dbmim/                         核心 dataset、model、metric、postprocess、utils
configs/                       当前保留的 smoke、推荐方法和 matched ablation 配置
scripts/download_data.py       CREMI 下载辅助脚本
scripts/prepare_*_data.py      publicEM / fullEM 预训练数据准备脚本
scripts/evaluate_*.py          VOI/ARAND 和 blockwise-scale 评测脚本
train_pretrain.py              dbMiM / MAE 预训练入口
train_finetune.py              affinity 微调入口
requirements-dbMIM.txt         Python 依赖
```

## 引用

```bibtex
@inproceedings{chen2023self,
  title={Self-supervised neuron segmentation with multi-agent reinforcement learning},
  author={Chen, Yinda and Huang, Wei and Zhou, Shenglong and Chen, Qi and Xiong, Zhiwei},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={609--617},
  year={2023}
}
```

## 数据和密钥说明

外部 EM 数据请遵守其原始 license 和访问规则。不要把下载的数据、生成的
checkpoint、TOS 凭证、Hugging Face token、GitHub token 或集群凭证提交到仓库。
