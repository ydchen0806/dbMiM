from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from dbmim.datasets import (
    EMVolumeDataset,
    SyntheticEMDataset,
    labels_to_affinities,
    labels_to_local_shape_descriptors,
)
from dbmim.metrics import AverageMeter, binary_iou_from_logits, dice_from_logits
from dbmim.models import (
    MAEBackboneAffinityNet,
    UNETRAffinityNet,
    UNETRAnisotropicAffinityNet,
    load_pretrained_backbone,
)
from dbmim.utils import (
    atomic_jsonl_append,
    count_parameters,
    device_from_config,
    ensure_dir,
    load_config,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
)


def setup_distributed(cfg: dict) -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = device_from_config(cfg)
    return distributed, rank, local_rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _loss_cfg(train_cfg: dict) -> dict:
    return train_cfg.get("loss", {}) if isinstance(train_cfg.get("loss", {}), dict) else {}


def _broadcast_loss_tensor(value, device: torch.device, *, channels: int, name: str) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        if len(value) != channels:
            raise ValueError(f"train.loss.{name} must have {channels} values, got {value}")
        tensor = torch.tensor([float(v) for v in value], device=device, dtype=torch.float32).view(1, channels, 1, 1, 1)
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32).view(1, 1, 1, 1, 1)
    return tensor


def _binary_ratio_weight(target: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """SuperHuman-style per-sample/channel binary class rebalancing."""

    target_bin = (target > 0.5).to(torch.float32)
    pos_fraction = target_bin.mean(dim=(2, 3, 4), keepdim=True).clamp(0.05, 0.99)
    pos_heavy = pos_fraction > 0.5
    pos_weight = alpha * (1.0 - pos_fraction) / pos_fraction.clamp_min(eps)
    neg_weight = alpha * pos_fraction / (1.0 - pos_fraction).clamp_min(eps)
    return torch.where(
        pos_heavy,
        target_bin + neg_weight * (1.0 - target_bin),
        pos_weight * target_bin + (1.0 - target_bin),
    )


def _weighted_mean(
    loss: torch.Tensor,
    channel_weight: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = channel_weight.expand_as(loss)
    if valid_mask is not None:
        weights = weights * valid_mask
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _superhuman_spatial_norm(logits: torch.Tensor, valid_mask: torch.Tensor | None, mode: str) -> torch.Tensor:
    """Normalization used by SuperHuman-style weighted affinity MSE."""

    if valid_mask is not None and mode in {"valid", "valid_spatial", "mask", "masked"}:
        return valid_mask.sum().clamp_min(1.0)
    bsz, _, depth, height, width = logits.shape
    return logits.new_tensor(float(bsz * depth * height * width)).clamp_min(1.0)


def make_affinity_valid_mask(labels: torch.Tensor, train_cfg: dict) -> torch.Tensor | None:
    cfg = _loss_cfg(train_cfg)
    if not bool(cfg.get("ignore_label_edges", train_cfg.get("ignore_label_edges", False))):
        return None
    ignore_label = int(cfg.get("ignore_label", train_cfg.get("ignore_label", 0)))
    if labels.ndim == 3:
        labels = labels.unsqueeze(0)
    labels = labels.long()
    valid_voxels = labels != ignore_label
    bsz, depth, height, width = labels.shape
    valid = labels.new_zeros((bsz, 3, depth, height, width), dtype=torch.float32)
    valid[:, 0, 1:] = valid_voxels[:, 1:] & valid_voxels[:, :-1]
    valid[:, 1, :, 1:] = valid_voxels[:, :, 1:] & valid_voxels[:, :, :-1]
    valid[:, 2, :, :, 1:] = valid_voxels[:, :, :, 1:] & valid_voxels[:, :, :, :-1]
    if bool(train_cfg.get("replicate_affinity_boundary", False)):
        valid[:, 0, 0] = valid_voxels[:, 0]
        valid[:, 1, :, 0] = valid_voxels[:, :, 0]
        valid[:, 2, :, :, 0] = valid_voxels[:, :, :, 0]
    return valid.float()


def affinity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    train_cfg: dict,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Configurable affinity objective.

    The default config is exactly the original BCE objective. Optional nested
    ``train.loss`` keys enable channel weighting, Dice regularization, and a
    focal factor for harder affinity edges.
    """

    cfg = _loss_cfg(train_cfg)
    channels = int(logits.shape[1])
    pos_weight_value = cfg.get("pos_weight", train_cfg.get("pos_weight", 1.0))
    neg_weight_value = cfg.get("neg_weight", train_cfg.get("neg_weight", 1.0))
    channel_weight_value = cfg.get("channel_weights", train_cfg.get("channel_weights", [1.0] * channels))
    pos_weight = _broadcast_loss_tensor(pos_weight_value, logits.device, channels=channels, name="pos_weight")
    neg_weight = _broadcast_loss_tensor(neg_weight_value, logits.device, channels=channels, name="neg_weight")
    channel_weight = _broadcast_loss_tensor(channel_weight_value, logits.device, channels=channels, name="channel_weights")

    loss_type = str(cfg.get("loss_type", train_cfg.get("loss_type", "bce"))).lower()
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=logits.device, dtype=torch.float32)

    if loss_type in {"mse", "weighted_mse", "superhuman_mse"}:
        prob_main = torch.sigmoid(logits)
        main = (prob_main - target).pow(2)
        if loss_type in {"weighted_mse", "superhuman_mse"}:
            main = main * _binary_ratio_weight(target, alpha=float(cfg.get("weight_alpha", 1.0)))
        bce_loss = _weighted_mean(main, channel_weight, valid_mask)
    elif loss_type in {"superhuman_weighted_mse", "superhuman_weighted_mse_loss", "superhuman_wmse"}:
        prob_main = torch.sigmoid(logits)
        weight = _binary_ratio_weight(target, alpha=float(cfg.get("weight_alpha", 1.0)))
        if valid_mask is not None:
            weight = weight * valid_mask
        main = (prob_main - target).pow(2) * weight * channel_weight
        norm_mode = str(cfg.get("superhuman_norm", "spatial")).lower()
        bce_loss = main.sum() / _superhuman_spatial_norm(logits, valid_mask, norm_mode)
    elif loss_type in {"bce", "weighted_bce"}:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = torch.where(target > 0.5, bce * pos_weight, bce * neg_weight)
        focal_gamma = float(cfg.get("focal_gamma", train_cfg.get("focal_gamma", 0.0)))
        pos_focal_gamma = float(cfg.get("pos_focal_gamma", train_cfg.get("pos_focal_gamma", focal_gamma)))
        neg_focal_gamma = float(cfg.get("neg_focal_gamma", train_cfg.get("neg_focal_gamma", focal_gamma)))
        if max(pos_focal_gamma, neg_focal_gamma) > 0.0:
            prob = torch.sigmoid(logits.detach())
            pos_pt = prob.clamp(1e-4, 1.0)
            neg_pt = (1.0 - prob).clamp(1e-4, 1.0)
            pos_factor = (1.0 - pos_pt).pow(pos_focal_gamma)
            neg_factor = (1.0 - neg_pt).pow(neg_focal_gamma)
            bce = bce * torch.where(target > 0.5, pos_factor, neg_factor)
        bce_loss = _weighted_mean(bce, channel_weight, valid_mask)
    elif loss_type in {"bce_mse", "mse_bce", "hybrid_bce_mse", "hybrid_mse_bce", "weighted_bce_mse"}:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = torch.where(target > 0.5, bce * pos_weight, bce * neg_weight)
        prob_main = torch.sigmoid(logits)
        mse = (prob_main - target).pow(2)
        if loss_type == "weighted_bce_mse":
            mse = mse * _binary_ratio_weight(target, alpha=float(cfg.get("weight_alpha", 1.0)))
        bce_main = _weighted_mean(bce, channel_weight, valid_mask)
        mse_main = _weighted_mean(mse, channel_weight, valid_mask)
        bce_loss = float(cfg.get("hybrid_bce_weight", 1.0)) * bce_main + float(
            cfg.get("hybrid_mse_weight", 1.0)
        ) * mse_main
    else:
        raise ValueError(f"unknown train.loss.loss_type: {loss_type}")

    dice_weight = float(cfg.get("dice_weight", train_cfg.get("dice_weight", 0.0)))
    boundary_dice_weight = float(cfg.get("boundary_dice_weight", train_cfg.get("boundary_dice_weight", 0.0)))
    dice_loss = logits.new_tensor(0.0)
    boundary_dice_loss = logits.new_tensor(0.0)
    if dice_weight > 0.0 or boundary_dice_weight > 0.0:
        prob = torch.sigmoid(logits)
    if dice_weight > 0.0:
        reduce_dims = (0, 2, 3, 4)
        smooth = float(cfg.get("dice_smooth", train_cfg.get("dice_smooth", 1.0)))
        dice_mask = 1.0 if valid_mask is None else valid_mask
        intersection = (prob * target * dice_mask).sum(dim=reduce_dims)
        denominator = ((prob + target) * dice_mask).sum(dim=reduce_dims)
        per_channel = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
        weights = channel_weight.view(channels)
        dice_loss = (per_channel * weights).sum() / weights.sum().clamp_min(1.0)
    if boundary_dice_weight > 0.0:
        reduce_dims = (0, 2, 3, 4)
        smooth = float(cfg.get("boundary_dice_smooth", train_cfg.get("boundary_dice_smooth", 1.0)))
        boundary_prob = 1.0 - prob
        boundary_target = 1.0 - target
        dice_mask = 1.0 if valid_mask is None else valid_mask
        intersection = (boundary_prob * boundary_target * dice_mask).sum(dim=reduce_dims)
        denominator = ((boundary_prob + boundary_target) * dice_mask).sum(dim=reduce_dims)
        per_channel = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
        weights = channel_weight.view(channels)
        boundary_dice_loss = (per_channel * weights).sum() / weights.sum().clamp_min(1.0)

    bce_weight = float(cfg.get("bce_weight", train_cfg.get("bce_weight", 1.0)))
    loss = bce_weight * bce_loss + dice_weight * dice_loss + boundary_dice_weight * boundary_dice_loss
    return loss, {
        "bce_loss": bce_loss.detach(),
        "main_loss": bce_loss.detach(),
        "dice_loss": dice_loss.detach(),
        "boundary_dice_loss": boundary_dice_loss.detach(),
        "valid_fraction": logits.new_tensor(1.0)
        if valid_mask is None
        else valid_mask.float().mean().detach(),
    }


def make_affinity_target(labels: torch.Tensor, train_cfg: dict) -> torch.Tensor:
    return labels_to_affinities(
        labels,
        replicate_boundary=bool(train_cfg.get("replicate_affinity_boundary", False)),
    )


def finetune_loss(
    logits: torch.Tensor,
    affinity_target: torch.Tensor,
    labels: torch.Tensor,
    train_cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Affinity loss plus optional LSD-style auxiliary supervision."""

    cfg = _loss_cfg(train_cfg)
    affinity_channels = int(affinity_target.shape[1])
    affinity_logits = logits[:, :affinity_channels]
    valid_mask = make_affinity_valid_mask(labels, train_cfg)
    if valid_mask is not None:
        valid_mask = valid_mask[:, :affinity_channels].to(logits.device)
    loss, parts = affinity_loss(affinity_logits, affinity_target, train_cfg, valid_mask=valid_mask)
    lsd_weight = float(cfg.get("lsd_weight", train_cfg.get("lsd_weight", 0.0)))
    lsd_loss = logits.new_tensor(0.0)
    lsd_fg_loss = logits.new_tensor(0.0)
    lsd_offset_loss = logits.new_tensor(0.0)
    if lsd_weight > 0.0:
        lsd_channels = int(cfg.get("lsd_channels", train_cfg.get("lsd_channels", 4)))
        lsd_logits = logits[:, affinity_channels : affinity_channels + lsd_channels]
        if int(lsd_logits.shape[1]) != lsd_channels:
            raise ValueError(
                f"lsd_weight>0 requires {affinity_channels + lsd_channels} model output channels, "
                f"got {int(logits.shape[1])}"
            )
        if lsd_channels != 4:
            raise ValueError("current LSD-style descriptor expects train.loss.lsd_channels=4")
        lsd_target = labels_to_local_shape_descriptors(labels).to(logits.device)
        foreground = lsd_target[:, :1]
        lsd_fg_loss = F.binary_cross_entropy_with_logits(lsd_logits[:, :1], foreground)
        foreground_voxels = foreground.sum().clamp_min(1.0)
        lsd_offset_loss = (
            F.smooth_l1_loss(
                torch.tanh(lsd_logits[:, 1:]) * foreground,
                lsd_target[:, 1:] * foreground,
                reduction="sum",
            )
            / (foreground_voxels * 3.0)
        )
        lsd_loss = float(cfg.get("lsd_fg_weight", 0.25)) * lsd_fg_loss + float(
            cfg.get("lsd_offset_weight", 1.0)
        ) * lsd_offset_loss
        loss = loss + lsd_weight * lsd_loss
    parts.update(
        {
            "lsd_loss": lsd_loss.detach(),
            "lsd_fg_loss": lsd_fg_loss.detach(),
            "lsd_offset_loss": lsd_offset_loss.detach(),
        }
    )
    return loss, parts


def build_dataset(cfg: dict) -> torch.utils.data.Dataset:
    data_cfg = cfg.get("data", {})
    volume_size = tuple(data_cfg.get("volume_size", cfg.get("model", {}).get("volume_size", [16, 64, 64])))
    if data_cfg.get("synthetic", False):
        return SyntheticEMDataset(
            length=int(data_cfg.get("synthetic_length", 128)),
            volume_size=volume_size,
            with_labels=True,
            seed=int(cfg.get("seed", 0)),
        )
    image_paths = data_cfg.get("image_paths") or data_cfg.get("train_images")
    label_paths = data_cfg.get("label_paths") or data_cfg.get("train_labels")
    if not image_paths or not label_paths:
        raise ValueError("data.image_paths and data.label_paths are required unless data.synthetic=true")
    return EMVolumeDataset(
        paths=image_paths,
        label_paths=label_paths,
        volume_size=volume_size,
        keys=data_cfg.get("image_keys") or data_cfg.get("keys"),
        label_keys=data_cfg.get("label_keys"),
        length_multiplier=int(data_cfg.get("length_multiplier", 1)),
        augment=bool(data_cfg.get("augment", True)),
        widen_border=bool(data_cfg.get("widen_border", False)),
        widen_border_radius=int(data_cfg.get("widen_border_radius", 1)),
        seed=int(cfg.get("seed", 0)),
    )


def build_affinity_model(cfg: dict) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    architecture = str(model_cfg.get("architecture", model_cfg.get("name", "mae_head"))).lower()
    common = {
        "in_channels": int(model_cfg.get("in_channels", 1)),
        "out_channels": int(model_cfg.get("out_channels", 3)),
        "volume_size": tuple(model_cfg.get("volume_size", [16, 64, 64])),
        "patch_size": tuple(model_cfg.get("patch_size", [4, 16, 16])),
        "embed_dim": int(model_cfg.get("embed_dim", 192)),
        "depth": int(model_cfg.get("depth", 6)),
        "num_heads": int(model_cfg.get("num_heads", 6)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
    }
    if architecture in {"unetr", "unetr_affinity", "unetr_affinity_net"}:
        return UNETRAffinityNet(
            **common,
            feature_size=int(model_cfg.get("feature_size", 32)),
            skip_indices=model_cfg.get("skip_indices"),
        )
    if architecture in {"unetr_aniso", "unetr_anisotropic", "paper_unetr", "unetr_dtrans"}:
        aniso_kwargs = {}
        if "use_dtrans" in model_cfg:
            aniso_kwargs["use_dtrans"] = bool(model_cfg["use_dtrans"])
        if "dtrans_stride_z" in model_cfg:
            aniso_kwargs["dtrans_stride_z"] = int(model_cfg["dtrans_stride_z"])
        return UNETRAnisotropicAffinityNet(
            **common,
            feature_size=int(model_cfg.get("feature_size", 32)),
            skip_indices=model_cfg.get("skip_indices"),
            **aniso_kwargs,
        )
    if architecture in {"linear_head", "transpose_head", "mae_head", "mae_backbone"}:
        return MAEBackboneAffinityNet(**common)
    raise ValueError(f"unknown model architecture: {architecture}")


def _strip_module_prefix(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def apply_param_freeze(model: torch.nn.Module, train_cfg: dict) -> list[str]:
    freeze_prefixes = [str(name) for name in train_cfg.get("freeze_param_prefixes", [])]
    if bool(train_cfg.get("freeze_encoder", False)):
        freeze_prefixes.extend(
            str(name)
            for name in train_cfg.get("encoder_param_prefixes", ["patch_embed", "pos_embed", "encoder_blocks", "norm"])
        )
    if not freeze_prefixes:
        return []
    prefixes = tuple(freeze_prefixes)
    frozen = []
    for name, param in model.named_parameters():
        match_name = _strip_module_prefix(name)
        if match_name.startswith(prefixes):
            param.requires_grad_(False)
            frozen.append(match_name)
    if not frozen:
        raise ValueError(f"freeze_param_prefixes matched no parameters: {freeze_prefixes}")
    return frozen


def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    encoder_lr_value = train_cfg.get("encoder_lr", None)
    if encoder_lr_value is None:
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            raise ValueError("optimizer received no trainable parameters")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    encoder_lr = float(encoder_lr_value)
    encoder_prefixes = tuple(
        str(name) for name in train_cfg.get("encoder_param_prefixes", ["patch_embed", "pos_embed", "encoder_blocks", "norm"])
    )
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match_name = _strip_module_prefix(name)
        if match_name.startswith(encoder_prefixes):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay, "name": "encoder"})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": lr, "weight_decay": weight_decay, "name": "decoder"})
    if not param_groups:
        raise ValueError("optimizer received no trainable parameters")
    return torch.optim.AdamW(param_groups)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, train_cfg: dict) -> dict[str, float]:
    model.eval()
    max_batches = int(train_cfg.get("eval_max_batches", 0))
    loss_m = AverageMeter()
    bce_m = AverageMeter()
    dice_loss_m = AverageMeter()
    boundary_dice_loss_m = AverageMeter()
    lsd_loss_m = AverageMeter()
    dice_m = AverageMeter()
    iou_m = AverageMeter()
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        target = make_affinity_target(labels, train_cfg).to(device)
        logits = model(image)
        loss, loss_parts = finetune_loss(logits, target, labels, train_cfg)
        affinity_logits = logits[:, : target.shape[1]]
        bsz = image.shape[0]
        loss_m.update(float(loss.detach().cpu()), bsz)
        bce_m.update(float(loss_parts["bce_loss"].cpu()), bsz)
        dice_loss_m.update(float(loss_parts["dice_loss"].cpu()), bsz)
        boundary_dice_loss_m.update(float(loss_parts["boundary_dice_loss"].cpu()), bsz)
        lsd_loss_m.update(float(loss_parts["lsd_loss"].cpu()), bsz)
        dice_m.update(dice_from_logits(affinity_logits.detach().cpu(), target.detach().cpu()), bsz)
        iou_m.update(binary_iou_from_logits(affinity_logits.detach().cpu(), target.detach().cpu()), bsz)
    return {
        "val_loss": loss_m.avg,
        "val_bce_loss": bce_m.avg,
        "val_dice_loss": dice_loss_m.avg,
        "val_boundary_dice_loss": boundary_dice_loss_m.avg,
        "val_lsd_loss": lsd_loss_m.avg,
        "val_dice": dice_m.avg,
        "val_iou": iou_m.avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="dbMiM affinity finetuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 0)))
    output_dir = ensure_dir(args.output_dir or cfg["output_dir"])
    log_path = output_dir / "finetune_log.jsonl"
    distributed, rank, local_rank, world_size, device = setup_distributed(cfg)
    dataset = build_dataset(cfg)

    train_cfg = cfg.get("train", {})
    val_fraction = float(train_cfg.get("val_fraction", 0.1))
    if val_fraction > 0.0 and len(dataset) > 1:
        val_len = min(len(dataset) - 1, max(1, int(round(len(dataset) * val_fraction))))
    else:
        val_len = 0
    train_len = len(dataset) - val_len
    if val_len > 0:
        train_set, val_set = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(int(cfg.get("seed", 0))),
        )
    else:
        train_set, val_set = dataset, None
    sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    loader = DataLoader(
        train_set,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=int(train_cfg.get("batch_size", 2)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 2)),
            pin_memory=(device.type == "cuda"),
        )
        if val_set is not None
        else None
    )

    model = build_affinity_model(cfg).to(device)

    pretrained = args.pretrained or cfg.get("pretrained", "")
    if pretrained:
        ckpt = load_checkpoint(pretrained, map_location="cpu")
        loaded = load_pretrained_backbone(model, ckpt)
        if is_main(rank):
            print(f"loaded_pretrained_keys={len(loaded)} from {pretrained}")

    frozen_params = apply_param_freeze(model, train_cfg)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = build_optimizer(model, train_cfg)
    start_epoch = 0
    global_step = 0
    best_score = -1.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_score = float(ckpt.get("best_score", -1.0))

    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    epochs = int(train_cfg.get("epochs", 100))
    max_steps = int(train_cfg.get("max_steps", 0))
    log_every = int(train_cfg.get("log_every", 20))
    eval_every = int(train_cfg.get("eval_every", 1))
    save_every = int(train_cfg.get("save_every", 1))
    save_steps = int(train_cfg.get("save_steps", 0))
    if is_main(rank):
        print(f"output_dir={output_dir}")
        print(
            f"dataset_size={len(dataset)} train={len(train_set)} "
            f"val={0 if val_set is None else len(val_set)} device={device} world_size={world_size}"
        )
        print(f"model_trainable_params={count_parameters(unwrap(model))}")
        print(f"loss_config={_loss_cfg(train_cfg)}")
        if frozen_params:
            print({"frozen_param_count": len(frozen_params), "frozen_param_prefix_sample": frozen_params[:12]})
        print(
            {
                "optimizer_param_groups": [
                    {
                        "name": group.get("name", f"group_{idx}"),
                        "lr": float(group["lr"]),
                        "weight_decay": float(group["weight_decay"]),
                        "num_tensors": len(group["params"]),
                        "num_parameters": int(sum(param.numel() for param in group["params"])),
                    }
                    for idx, group in enumerate(optimizer.param_groups)
                ]
            }
        )
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            target = make_affinity_target(labels, train_cfg).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(image)
                loss, loss_parts = finetune_loss(logits, target, labels, train_cfg)
            scaler.scale(loss).backward()
            if float(train_cfg.get("clip_grad", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["clip_grad"]))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if is_main(rank) and global_step % log_every == 0:
                payload = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float(loss.detach().cpu()),
                    "train_main_loss": float(loss_parts["main_loss"].cpu()),
                    "train_bce_loss": float(loss_parts["bce_loss"].cpu()),
                    "train_dice_loss": float(loss_parts["dice_loss"].cpu()),
                    "train_boundary_dice_loss": float(loss_parts["boundary_dice_loss"].cpu()),
                    "train_lsd_loss": float(loss_parts["lsd_loss"].cpu()),
                    "train_valid_fraction": float(loss_parts["valid_fraction"].cpu()),
                    "elapsed_sec": round(time.time() - t0, 2),
                }
                print(payload, flush=True)
                atomic_jsonl_append(log_path, payload)
            if save_steps and global_step % save_steps == 0:
                if distributed:
                    dist.barrier()
                if is_main(rank):
                    payload = {
                        "model": unwrap(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_score": best_score,
                        "config": cfg,
                        "last_eval": {},
                    }
                    save_checkpoint(output_dir / f"checkpoint_step_{global_step:08d}.pt", payload)
                    save_checkpoint(output_dir / "finetuned_latest.pt", payload)
                if distributed:
                    dist.barrier()
            if max_steps and global_step >= max_steps:
                break

        stats = {}
        should_eval = (
            eval_every > 0
            and val_loader is not None
            and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs)
        )
        if distributed and should_eval:
            dist.barrier()
        if is_main(rank) and should_eval:
            stats = evaluate(unwrap(model), val_loader, device, train_cfg)
            stats.update({"epoch": epoch, "step": global_step})
            print(stats, flush=True)
            atomic_jsonl_append(log_path, stats)
            if stats["val_dice"] > best_score:
                best_score = stats["val_dice"]
                save_checkpoint(
                    output_dir / "finetuned_best.pt",
                    {
                        "model": unwrap(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_score": best_score,
                        "config": cfg,
                    },
                )
        if distributed and should_eval:
            dist.barrier()

        should_save = (save_every > 0 and (epoch + 1) % save_every == 0) or epoch + 1 == epochs or (
            max_steps and global_step >= max_steps
        )
        if distributed and should_save:
            dist.barrier()
        if is_main(rank) and should_save:
            payload = {
                "model": unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_score": best_score,
                "config": cfg,
                "last_eval": stats,
            }
            save_checkpoint(output_dir / f"checkpoint_{epoch:04d}.pt", payload)
            save_checkpoint(output_dir / "finetuned_latest.pt", payload)
        if distributed and should_save:
            dist.barrier()
        if max_steps and global_step >= max_steps:
            break
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
