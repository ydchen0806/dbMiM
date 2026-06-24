from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dbmim.datasets import EMVolumeDataset, SyntheticEMDataset
from dbmim.models import DBMIM3DMAE, DecoderAwareDBMIM3DMAE, DecisionModule
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


def build_dataset(cfg: dict) -> torch.utils.data.Dataset:
    data_cfg = cfg.get("data", {})
    volume_size = tuple(data_cfg.get("volume_size", cfg.get("model", {}).get("volume_size", [16, 64, 64])))
    if data_cfg.get("synthetic", False):
        return SyntheticEMDataset(
            length=int(data_cfg.get("synthetic_length", 128)),
            volume_size=volume_size,
            with_labels=False,
            seed=int(cfg.get("seed", 0)),
        )
    paths = data_cfg.get("train_paths") or data_cfg.get("paths")
    if not paths:
        raise ValueError("data.train_paths is required unless data.synthetic=true")
    return EMVolumeDataset(
        paths=paths,
        volume_size=volume_size,
        keys=data_cfg.get("image_keys") or data_cfg.get("keys"),
        length_multiplier=int(data_cfg.get("length_multiplier", 1)),
        augment=bool(data_cfg.get("augment", True)),
        augment_rotate_xy=bool(data_cfg.get("augment_rotate_xy", False)),
        augment_gamma=bool(data_cfg.get("augment_gamma", False)),
        augment_gamma_range=tuple(data_cfg.get("augment_gamma_range", [0.7, 1.5])),
        augment_noise_std=float(data_cfg.get("augment_noise_std", 0.03)),
        seed=int(cfg.get("seed", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="dbMiM pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 0)))
    output_dir = ensure_dir(args.output_dir or cfg["output_dir"])
    log_path = output_dir / "train_log.jsonl"
    distributed, rank, local_rank, world_size, device = setup_distributed(cfg)

    dataset = build_dataset(cfg)
    train_cfg = cfg.get("train", {})
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    model_cfg = cfg.get("model", {})
    architecture = str(model_cfg.get("architecture", model_cfg.get("name", "dbmim"))).lower()
    if architecture in {"decoder_aware", "decoder_aware_dbmim", "unetr_em_dbmim", "affinity_aware_dbmim"}:
        model = DecoderAwareDBMIM3DMAE(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("out_channels", 3)),
            volume_size=tuple(model_cfg.get("volume_size", [32, 160, 160])),
            patch_size=tuple(model_cfg.get("patch_size", [4, 16, 16])),
            embed_dim=int(model_cfg.get("embed_dim", 192)),
            depth=int(model_cfg.get("depth", 6)),
            num_heads=int(model_cfg.get("num_heads", 6)),
            feature_size=int(model_cfg.get("feature_size", 32)),
            decoder_dim=int(model_cfg.get("decoder_dim", model_cfg.get("embed_dim", 192))),
            mask_ratio=float(model_cfg.get("mask_ratio", 0.75)),
            structure_weight=float(model_cfg.get("structure_weight", 0.1)),
            structure_axis_weights=model_cfg.get("structure_axis_weights"),
            membrane_weight=float(model_cfg.get("membrane_weight", 0.0)),
            membrane_axis_weights=model_cfg.get("membrane_axis_weights"),
            membrane_clip=float(model_cfg.get("membrane_clip", 5.0)),
            affinity_weight=float(model_cfg.get("affinity_weight", 0.35)),
            affinity_temperature=float(model_cfg.get("affinity_temperature", 1.0)),
            affinity_axis_weights=model_cfg.get("affinity_axis_weights"),
            affinity_membrane_weight=float(model_cfg.get("affinity_membrane_weight", 0.0)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            skip_indices=model_cfg.get("skip_indices"),
            em_refine_depth=int(model_cfg.get("em_refine_depth", 2)),
            channel_bias_init=model_cfg.get("channel_bias_init"),
            use_dtrans=model_cfg.get("use_dtrans", None),
            dtrans_stride_z=model_cfg.get("dtrans_stride_z", None),
        ).to(device)
    else:
        model = DBMIM3DMAE(
            in_channels=int(model_cfg.get("in_channels", 1)),
            volume_size=tuple(model_cfg.get("volume_size", [16, 64, 64])),
            patch_size=tuple(model_cfg.get("patch_size", [4, 16, 16])),
            embed_dim=int(model_cfg.get("embed_dim", 192)),
            depth=int(model_cfg.get("depth", 6)),
            num_heads=int(model_cfg.get("num_heads", 6)),
            decoder_dim=int(model_cfg.get("decoder_dim", model_cfg.get("embed_dim", 192))),
            mask_ratio=float(model_cfg.get("mask_ratio", 0.75)),
            mask_strategy=str(model_cfg.get("mask_strategy", "random")),
            edge_mask_power=float(model_cfg.get("edge_mask_power", 1.0)),
            edge_mask_noise=float(model_cfg.get("edge_mask_noise", 0.05)),
            structure_weight=float(model_cfg.get("structure_weight", 0.1)),
            structure_axis_weights=model_cfg.get("structure_axis_weights"),
            membrane_weight=float(model_cfg.get("membrane_weight", 0.0)),
            membrane_axis_weights=model_cfg.get("membrane_axis_weights"),
            membrane_clip=float(model_cfg.get("membrane_clip", 5.0)),
        ).to(device)
    decision_cfg = cfg.get("decision", {})
    use_policy = bool(decision_cfg.get("enabled", True))
    decision_module = None
    if use_policy:
        decision_module = DecisionModule(
            embed_dim=int(model_cfg.get("embed_dim", 192)),
            hidden_dim=int(decision_cfg.get("hidden_dim", 256)),
            min_mask_ratio=float(decision_cfg.get("min_mask_ratio", 0.35)),
            max_mask_ratio=float(decision_cfg.get("max_mask_ratio", 0.90)),
            entropy_coef=float(decision_cfg.get("entropy_coef", 0.01)),
            value_coef=float(decision_cfg.get("value_coef", 0.5)),
            ratio_coef=float(decision_cfg.get("ratio_coef", 0.25)),
        ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        if decision_module is not None:
            decision_module = torch.nn.parallel.DistributedDataParallel(
                decision_module,
                device_ids=[local_rank],
                find_unused_parameters=True,
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1.5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    policy_optimizer = None
    if decision_module is not None:
        policy_optimizer = torch.optim.AdamW(
            decision_module.parameters(),
            lr=float(decision_cfg.get("lr", 5e-4)),
            weight_decay=float(decision_cfg.get("weight_decay", 1e-4)),
        )

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if decision_module is not None and ckpt.get("decision_module") is not None:
            unwrap(decision_module).load_state_dict(ckpt["decision_module"])
        if policy_optimizer is not None and ckpt.get("policy_optimizer") is not None:
            policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))

    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    epochs = int(train_cfg.get("epochs", 100))
    max_steps = int(train_cfg.get("max_steps", 0))
    log_every = int(train_cfg.get("log_every", 20))
    save_every = int(train_cfg.get("save_every", 1))
    save_steps = int(train_cfg.get("save_steps", 0))
    target_ratio = float(decision_cfg.get("target_mask_ratio", model_cfg.get("mask_ratio", 0.75)))
    freeze_policy_after = int(decision_cfg.get("freeze_after_steps", 0))
    use_frozen_policy_after_freeze = bool(decision_cfg.get("use_frozen_policy_after_freeze", False))
    deterministic_frozen_policy = bool(decision_cfg.get("deterministic_frozen_policy", True))

    if is_main(rank):
        print(f"output_dir={output_dir}")
        print(f"dataset_size={len(dataset)} batches={len(loader)} device={device} world_size={world_size}")
        print(f"model_trainable_params={count_parameters(unwrap(model))}")
        if decision_module is not None:
            print(f"decision_trainable_params={count_parameters(unwrap(decision_module))}")

    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            update_policy = (
                decision_module is not None
                and policy_optimizer is not None
                and (freeze_policy_after <= 0 or global_step < freeze_policy_after)
            )
            if decision_module is not None:
                decision_module.train(update_policy)
            optimizer.zero_grad(set_to_none=True)
            if update_policy:
                policy_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                active_decision_module = (
                    decision_module
                    if update_policy or (decision_module is not None and use_frozen_policy_after_freeze)
                    else None
                )
                deterministic_policy = (
                    deterministic_frozen_policy
                    and active_decision_module is not None
                    and not update_policy
                )
                out = model(
                    image,
                    decision_module=active_decision_module,
                    target_mask_ratio=target_ratio,
                    deterministic_policy=deterministic_policy,
                )
                loss = out.loss
                policy_loss = image.new_tensor(0.0)
                if (
                    decision_module is not None
                    and policy_optimizer is not None
                    and update_policy
                    and out.decision is not None
                ):
                    per_sample = (out.pred.detach() - out.target).pow(2).flatten(1).mean(dim=1)
                    policy_loss = unwrap(decision_module).policy_loss(out.decision, per_sample)
                    loss = loss + float(decision_cfg.get("policy_weight", 0.05)) * policy_loss
            scaler.scale(loss).backward()
            if float(train_cfg.get("clip_grad", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["clip_grad"]))
            scaler.step(optimizer)
            if update_policy:
                scaler.step(policy_optimizer)
            scaler.update()

            global_step += 1
            if is_main(rank) and global_step % log_every == 0:
                payload = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(out.loss.detach().cpu()),
                    "pixel_loss": float(out.pixel_loss.detach().cpu()),
                    "structure_loss": float(out.structure_loss.detach().cpu()),
                    "affinity_loss": float(out.affinity_loss.detach().cpu())
                    if out.affinity_loss is not None
                    else 0.0,
                    "membrane_weight_mean": float(out.membrane_weight_mean.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "mask_ratio": float(out.mask.float().mean().detach().cpu()),
                    "lr": optimizer.param_groups[0]["lr"],
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
                        "decision_module": unwrap(decision_module).state_dict() if decision_module is not None else None,
                        "optimizer": optimizer.state_dict(),
                        "policy_optimizer": policy_optimizer.state_dict() if policy_optimizer is not None else None,
                        "epoch": epoch,
                        "global_step": global_step,
                        "config": cfg,
                    }
                    save_checkpoint(output_dir / f"checkpoint_step_{global_step:08d}.pt", payload)
                    save_checkpoint(output_dir / "pretrained_latest.pt", payload)
                if distributed:
                    dist.barrier()
            if max_steps and global_step >= max_steps:
                break

        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs or (max_steps and global_step >= max_steps):
            if is_main(rank):
                payload = {
                    "model": unwrap(model).state_dict(),
                    "decision_module": unwrap(decision_module).state_dict() if decision_module is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "policy_optimizer": policy_optimizer.state_dict() if policy_optimizer is not None else None,
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": cfg,
                }
                save_checkpoint(output_dir / f"checkpoint_{epoch:04d}.pt", payload)
                save_checkpoint(output_dir / "pretrained_latest.pt", payload)
        if max_steps and global_step >= max_steps:
            break
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
