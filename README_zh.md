# dbMiM 神经元分割

[English](README.md) | [Hugging Face 权重](https://huggingface.co/cyd0806/dbmim-neuron-segmentation)

这是 dbMiM 预训练和 CREMI 神经元分割的官方实现仓库。当前维护流程是：

1. 准备未标注 EM 体数据；
2. 运行 dbMiM / MAE 风格 masked-image 预训练；
3. 在 CREMI 上微调各向异性 3D UNETR affinity 网络；
4. 在 CREMI A/B/C 全体积上计算 VOI 和 adapted Rand error (ARAND)；
5. 使用 waterz 作为参考实例分割后处理。

可学习 / 可微后处理作为独立方法维护在：
https://github.com/ydchen0806/nnEM-Seg-diff-postprocess。

## 方法

分割模型为 `UNETRAnisotropicAffinityNet`。

- 输入 crop：`32 x 160 x 160`
- Patch size：`(4, 16, 16)`
- 输出：z/y/x 最近邻 affinity logits
- Backbone：由 dbMiM 预训练初始化的 ViT encoder
- Decoder：UNETR 分阶段上采样，并加入各向异性 z transition
- 微调损失：MSE + membrane-aware spatial weighting (MAWS)
- 评测：CREMI A/B/C 全体积推理，`ignore_label=0`，boundary ignore 为 `xy=1, z=0`

dbMiM 预训练对 3D ViT patch 做 mask，并重建 EM voxel；训练中使用
membrane-aware weighting 和轻量 structure-consistency term。plain MAE 控制组使用
相同数据、模型大小、crop、mask ratio 和 schedule，只关闭 dbMiM-specific term。

## 结果

VOI 和 ARAND 都是越低越好。

| Run | 权重 | VOI | ARAND | 说明 |
|---|---|---:|---:|---|
| R48 | `weights/publicem_dbmim_r48_seed309_long20k/finetuned_latest.pt` | **0.962154** | 0.178252 | VOI 最好 |
| R57 | `weights/publicem_dbmim_r57_seed777_long20k/finetuned_latest.pt` | 0.964617 | **0.178248** | repeat sweep 中 ARAND 最好 |
| R33 | `weights/fullem_mixedmask_dbmim_r33/finetuned_latest.pt` | 1.039372 | 0.205380 | fullEM 最好 recipe |

这里的验证集是公开 CREMI A/B/C training volumes，不使用 hidden challenge label。

## 权重

权重托管在 Hugging Face：

https://huggingface.co/cyd0806/dbmim-neuron-segmentation

| 文件 | 用途 |
|---|---|
| `weights/publicem_dbmim_r48_seed309_long20k/finetuned_latest.pt` | 推荐分割权重 |
| `weights/publicem_dbmim_r57_seed777_long20k/finetuned_latest.pt` | ARAND 最好 repeat 权重 |
| `weights/publicem_dbmim_r17/pretrained_latest.pt` | PublicEM dbMiM encoder 预训练权重 |
| `weights/publicem_dbmim_r17/finetuned_latest.pt` | 早期 publicEM 微调权重 |
| `weights/fullem_mixedmask_dbmim_r33/pretrained_latest.pt` | FullEM mixed-mask 预训练权重 |
| `weights/fullem_mixedmask_dbmim_r33/finetuned_latest.pt` | FullEM mixed-mask 微调权重 |

下载示例：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="cyd0806/dbmim-neuron-segmentation",
    local_dir="weights/dbmim-neuron-segmentation",
)
```

## 环境

```bash
git clone https://github.com/ydchen0806/dbMiM.git
cd dbMiM
python -m pip install -r requirements-dbMIM.txt
```

waterz 对训练不是必需的，但复现实例分割指标需要安装 waterz。

## 数据

CREMI 微调和评测默认读取：

```text
data/CREMI/sample_A_20160501.hdf
data/CREMI/sample_B_20160501.hdf
data/CREMI/sample_C_20160501.hdf
```

准备 publicEM 预训练数据：

```bash
python scripts/prepare_public_em_pretrain_data.py \
  --target-dir data/EM_pretrain_data
```

准备 fullEM 预训练数据：

```bash
HF_TOKEN=<your_token> python scripts/prepare_em_pretrain_data.py \
  --target-dir data/EM_pretrain_data
```

## 预训练

PublicEM dbMiM：

```bash
python train_pretrain.py \
  --config configs/pretrain_public_em_membrane_r16.yaml
```

FullEM mixed-mask：

```bash
python train_pretrain.py \
  --config configs/pretrain_em_full_mixedmask_dbmim_r33.yaml
```

## 微调

推荐 R48 微调：

```bash
python train_finetune.py \
  --config configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r16_seed309_long20k_r48q.yaml
```

如果权重不在 `outputs/` 下，需要同步修改 config 中的预训练权重路径。

## 评测

CREMI A/B/C 全体积 waterz 评测：

```bash
python scripts/evaluate_cremi_segmentation.py \
  --config configs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r16_seed309_long20k_r48q.yaml \
  --checkpoint outputs/finetune_cremi_real_unetr_aniso_em_mse_maws_publicem_r16_seed309_long20k_r48q/finetuned_latest.pt \
  --data-dir data/CREMI \
  --output-dir outputs/eval_r48_cremi_abc \
  --crop-size 0 0 0 \
  --stride 16 80 80 \
  --backends waterz \
  --thresholds 0.16 0.18 0.20 0.22 0.24 \
  --calibration-biases -0.25 -0.50 -0.50 \
  --seed-method maxima_distance \
  --seed-distance 10 \
  --boundary-threshold 0.5 \
  --waterz-scoring hist_quantile \
  --batched-waterz \
  --metric-backend skimage \
  --ignore-label 0 \
  --cremi-boundary-ignore-distance-xy 1 \
  --cremi-boundary-ignore-distance-z 0 \
  --device cuda
```

`--batched-waterz` 会在同一次 waterz hierarchy 中评测多个 threshold。它保持
R48 的 VOI 不变（`0.962154`），并把 CREMI A/B/C threshold sweep 的后处理时间
从约 `75s` 降到约 `17s`。

结果写入：

```text
outputs/eval_r48_cremi_abc/cremi_segmentation_summary.json
```

## 目录

```text
dbmim/                         模型、数据、指标和工具
configs/                       预训练和微调配置
scripts/prepare_*_data.py       数据准备
scripts/evaluate_*.py           CREMI 评测
train_pretrain.py               dbMiM / MAE 预训练
train_finetune.py               CREMI affinity 微调
```

大数据、权重和生成结果不进入 Git。
