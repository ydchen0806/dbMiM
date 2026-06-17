from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from dbmim.datasets import EMVolumeDataset, SyntheticEMDataset, labels_to_affinities
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


def affinity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    train_cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Configurable affinity objective.

    The default config is exactly the original BCE objective. Optional nested
    ``train.loss`` keys enable channel weighting, Dice regularization, and a
    focal factor for harder affinity edges.
    """

    cfg = _loss_cfg(train_cfg)
    channels = int(logits.shape[1])
    pos_weight_value = cfg.get("pos_weight", train_cfg.get("pos_weight", 1.0))
    channel_weight_value = cfg.get("channel_weights", train_cfg.get("channel_weights", [1.0] * channels))
    pos_weight = _broadcast_loss_tensor(pos_weight_value, logits.device, channels=channels, name="pos_weight")
    channel_weight = _broadcast_loss_tensor(channel_weight_value, logits.device, channels=channels, name="channel_weights")

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = torch.where(target > 0.5, bce * pos_weight, bce)
    focal_gamma = float(cfg.get("focal_gamma", train_cfg.get("focal_gamma", 0.0)))
    if focal_gamma > 0.0:
        prob = torch.sigmoid(logits.detach())
        pt = torch.where(target > 0.5, prob, 1.0 - prob).clamp(1e-4, 1.0)
        bce = bce * (1.0 - pt).pow(focal_gamma)
    bce_loss = (bce * channel_weight).sum() / channel_weight.expand_as(bce).sum().clamp_min(1.0)

    dice_weight = float(cfg.get("dice_weight", train_cfg.get("dice_weight", 0.0)))
    dice_loss = logits.new_tensor(0.0)
    if dice_weight > 0.0:
        prob = torch.sigmoid(logits)
        reduce_dims = (0, 2, 3, 4)
        smooth = float(cfg.get("dice_smooth", train_cfg.get("dice_smooth", 1.0)))
        intersection = (prob * target).sum(dim=reduce_dims)
        denominator = (prob + target).sum(dim=reduce_dims)
        per_channel = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
        weights = channel_weight.view(channels)
        dice_loss = (per_channel * weights).sum() / weights.sum().clamp_min(1.0)

    bce_weight = float(cfg.get("bce_weight", train_cfg.get("bce_weight", 1.0)))
    loss = bce_weight * bce_loss + dice_weight * dice_loss
    return loss, {
        "bce_loss": bce_loss.detach(),
        "dice_loss": dice_loss.detach(),
    }


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


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, train_cfg: dict) -> dict[str, float]:
    model.eval()
    loss_m = AverageMeter()
    bce_m = AverageMeter()
    dice_loss_m = AverageMeter()
    dice_m = AverageMeter()
    iou_m = AverageMeter()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        target = labels_to_affinities(labels).to(device)
        logits = model(image)
        loss, loss_parts = affinity_loss(logits, target, train_cfg)
        bsz = image.shape[0]
        loss_m.update(float(loss.detach().cpu()), bsz)
        bce_m.update(float(loss_parts["bce_loss"].cpu()), bsz)
        dice_loss_m.update(float(loss_parts["dice_loss"].cpu()), bsz)
        dice_m.update(dice_from_logits(logits.detach().cpu(), target.detach().cpu()), bsz)
        iou_m.update(binary_iou_from_logits(logits.detach().cpu(), target.detach().cpu()), bsz)
    return {
        "val_loss": loss_m.avg,
        "val_bce_loss": bce_m.avg,
        "val_dice_loss": dice_loss_m.avg,
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
    val_len = max(1, int(round(len(dataset) * val_fraction))) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len
    if val_len > 0:
        train_set, val_set = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(int(cfg.get("seed", 0))),
        )
    else:
        train_set, val_set = dataset, dataset
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
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
    )

    model = build_affinity_model(cfg).to(device)

    pretrained = args.pretrained or cfg.get("pretrained", "")
    if pretrained:
        ckpt = load_checkpoint(pretrained, map_location="cpu")
        loaded = load_pretrained_backbone(model, ckpt)
        if is_main(rank):
            print(f"loaded_pretrained_keys={len(loaded)} from {pretrained}")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
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
    if is_main(rank):
        print(f"output_dir={output_dir}")
        print(f"dataset_size={len(dataset)} train={len(train_set)} val={len(val_set)} device={device} world_size={world_size}")
        print(f"model_trainable_params={count_parameters(unwrap(model))}")
        print(f"loss_config={_loss_cfg(train_cfg)}")
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            target = labels_to_affinities(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(image)
                loss, loss_parts = affinity_loss(logits, target, train_cfg)
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
                    "train_bce_loss": float(loss_parts["bce_loss"].cpu()),
                    "train_dice_loss": float(loss_parts["dice_loss"].cpu()),
                    "elapsed_sec": round(time.time() - t0, 2),
                }
                print(payload, flush=True)
                atomic_jsonl_append(log_path, payload)
            if max_steps and global_step >= max_steps:
                break

        stats = {}
        should_eval = (epoch + 1) % eval_every == 0 or epoch + 1 == epochs
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

        should_save = (epoch + 1) % save_every == 0 or epoch + 1 == epochs or (max_steps and global_step >= max_steps)
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
