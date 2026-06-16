from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dbmim.datasets import EMVolumeDataset, SyntheticEMDataset, labels_to_affinities
from dbmim.metrics import AverageMeter, binary_iou_from_logits, dice_from_logits
from dbmim.models import MAEBackboneAffinityNet, load_pretrained_backbone
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
        length_multiplier=int(data_cfg.get("length_multiplier", 1)),
        augment=bool(data_cfg.get("augment", True)),
        seed=int(cfg.get("seed", 0)),
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_m = AverageMeter()
    dice_m = AverageMeter()
    iou_m = AverageMeter()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        target = labels_to_affinities(labels).to(device)
        logits = model(image)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        bsz = image.shape[0]
        loss_m.update(float(loss.detach().cpu()), bsz)
        dice_m.update(dice_from_logits(logits.detach().cpu(), target.detach().cpu()), bsz)
        iou_m.update(binary_iou_from_logits(logits.detach().cpu(), target.detach().cpu()), bsz)
    return {"val_loss": loss_m.avg, "val_dice": dice_m.avg, "val_iou": iou_m.avg}


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
    device = device_from_config(cfg)
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
    loader = DataLoader(
        train_set,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
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

    model_cfg = cfg.get("model", {})
    model = MAEBackboneAffinityNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        volume_size=tuple(model_cfg.get("volume_size", [16, 64, 64])),
        patch_size=tuple(model_cfg.get("patch_size", [4, 16, 16])),
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 6)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    pretrained = args.pretrained or cfg.get("pretrained", "")
    if pretrained:
        ckpt = load_checkpoint(pretrained, map_location="cpu")
        loaded = load_pretrained_backbone(model, ckpt)
        print(f"loaded_pretrained_keys={len(loaded)} from {pretrained}")

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
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_score = float(ckpt.get("best_score", -1.0))

    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    epochs = int(train_cfg.get("epochs", 100))
    max_steps = int(train_cfg.get("max_steps", 0))
    log_every = int(train_cfg.get("log_every", 20))
    eval_every = int(train_cfg.get("eval_every", 1))
    pos_weight = torch.tensor(float(train_cfg.get("pos_weight", 1.0)), device=device)

    print(f"output_dir={output_dir}")
    print(f"dataset_size={len(dataset)} train={len(train_set)} val={len(val_set)} device={device}")
    print(f"model_trainable_params={count_parameters(model)}")
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            target = labels_to_affinities(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(image)
                loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            if float(train_cfg.get("clip_grad", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["clip_grad"]))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if global_step % log_every == 0:
                payload = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float(loss.detach().cpu()),
                    "elapsed_sec": round(time.time() - t0, 2),
                }
                print(payload, flush=True)
                atomic_jsonl_append(log_path, payload)
            if max_steps and global_step >= max_steps:
                break

        stats = {}
        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            stats = evaluate(model, val_loader, device)
            stats.update({"epoch": epoch, "step": global_step})
            print(stats, flush=True)
            atomic_jsonl_append(log_path, stats)
            if stats["val_dice"] > best_score:
                best_score = stats["val_dice"]
                save_checkpoint(
                    output_dir / "finetuned_best.pt",
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_score": best_score,
                        "config": cfg,
                    },
                )

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_score": best_score,
            "config": cfg,
            "last_eval": stats,
        }
        save_checkpoint(output_dir / f"checkpoint_{epoch:04d}.pt", payload)
        save_checkpoint(output_dir / "finetuned_latest.pt", payload)
        if max_steps and global_step >= max_steps:
            break


if __name__ == "__main__":
    main()
