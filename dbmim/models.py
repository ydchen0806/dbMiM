from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value, value)


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 192,
        patch_size: tuple[int, int, int] = (4, 16, 16),
    ) -> None:
        super().__init__()
        self.patch_size = _triple(patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(x)
        grid = tuple(x.shape[-3:])
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, grid


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class DecisionModule(nn.Module):
    """Shared actor-critic mask policy used by patch agents.

    Each patch is treated as an agent. Agents share policy parameters and receive
    a global context feature, which keeps the implementation close to the paper
    while avoiding hand-written per-patch modules.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        min_mask_ratio: float = 0.35,
        max_mask_ratio: float = 0.90,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ratio_coef: float = 0.25,
    ) -> None:
        super().__init__()
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ratio_coef = ratio_coef
        self.feature = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)

    def _enforce_ratio(self, actions: torch.Tensor, mask_prob: torch.Tensor) -> torch.Tensor:
        bsz, num_patches = actions.shape
        min_count = max(1, int(round(self.min_mask_ratio * num_patches)))
        max_count = min(num_patches - 1, int(round(self.max_mask_ratio * num_patches)))
        fixed = actions.clone()
        for bidx in range(bsz):
            count = int(fixed[bidx].sum().item())
            if count < min_count:
                keep_idx = torch.nonzero(fixed[bidx] == 0, as_tuple=False).flatten()
                add_count = min(min_count - count, keep_idx.numel())
                if add_count > 0:
                    order = torch.argsort(mask_prob[bidx, keep_idx], descending=True)
                    fixed[bidx, keep_idx[order[:add_count]]] = 1
            elif count > max_count:
                mask_idx = torch.nonzero(fixed[bidx] == 1, as_tuple=False).flatten()
                drop_count = min(count - max_count, mask_idx.numel())
                if drop_count > 0:
                    order = torch.argsort(mask_prob[bidx, mask_idx], descending=False)
                    fixed[bidx, mask_idx[order[:drop_count]]] = 0
        return fixed

    def forward(
        self,
        patch_features: torch.Tensor,
        target_mask_ratio: float | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self.feature(patch_features)
        h = h + self.context(h.mean(dim=1, keepdim=True))
        h = torch.nan_to_num(h, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1).mean(dim=1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        probs = logits.softmax(dim=-1)
        actions = self._enforce_ratio(actions, probs[..., 1])
        mask = actions.bool()
        log_prob = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().mean(dim=1)
        mask_ratio = mask.float().mean(dim=1)
        target = target_mask_ratio if target_mask_ratio is not None else float(mask_ratio.mean().item())
        ratio_penalty = (mask_ratio - target).abs()
        return {
            "mask": mask,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "mask_ratio": mask_ratio,
            "ratio_penalty": ratio_penalty,
        }

    def policy_loss(
        self,
        decision: dict[str, torch.Tensor],
        reconstruction_loss: torch.Tensor,
    ) -> torch.Tensor:
        reward = -reconstruction_loss.detach() - self.ratio_coef * decision["ratio_penalty"].detach()
        advantage = reward - decision["value"]
        actor = -(decision["log_prob"] * advantage.detach()).mean()
        critic = F.mse_loss(decision["value"], reward)
        entropy = -decision["entropy"].mean()
        return actor + self.value_coef * critic + self.entropy_coef * entropy


@dataclass
class MAEOutput:
    loss: torch.Tensor
    pixel_loss: torch.Tensor
    structure_loss: torch.Tensor
    membrane_weight_mean: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    decision: dict[str, torch.Tensor] | None
    affinity_loss: torch.Tensor | None = None
    affinity_pred: torch.Tensor | None = None
    affinity_target: torch.Tensor | None = None


def _normalize_axis_weights(axis_weights: Sequence[float] | None) -> tuple[float, float, float]:
    if axis_weights is None:
        return (1.0, 1.0, 1.0)
    values = tuple(float(v) for v in axis_weights)
    if len(values) != 3:
        raise ValueError(f"axis weights must have 3 values, got {values}")
    if sum(max(v, 0.0) for v in values) <= 0.0:
        raise ValueError(f"at least one axis weight must be positive, got {values}")
    return values  # type: ignore[return-value]


def membrane_edge_map_3d(
    x: torch.Tensor,
    axis_weights: Sequence[float] | None = None,
    clip: float = 5.0,
) -> torch.Tensor:
    """Unsupervised EM membrane proxy from local anisotropic intensity gradients."""

    weights = _normalize_axis_weights(axis_weights)
    edge = x.new_zeros(x.shape)
    for dim, weight in zip((-3, -2, -1), weights):
        if weight <= 0.0:
            continue
        grad = x.diff(dim=dim).abs()
        pad_shape = list(grad.shape)
        pad_shape[dim] = 1
        grad = torch.cat([x.new_zeros(pad_shape), grad], dim=dim)
        edge = edge + float(weight) * grad
    reduce_dims = tuple(range(1, edge.ndim))
    edge = edge / edge.mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    if clip > 0:
        edge = edge.clamp(max=float(clip))
    return edge


class DBMIM3DMAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        volume_size: tuple[int, int, int] = (16, 64, 64),
        patch_size: tuple[int, int, int] = (4, 16, 16),
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        decoder_dim: int = 192,
        mask_ratio: float = 0.75,
        structure_weight: float = 0.1,
        structure_axis_weights: Sequence[float] | None = None,
        membrane_weight: float = 0.0,
        membrane_axis_weights: Sequence[float] | None = None,
        membrane_clip: float = 5.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.volume_size = _triple(volume_size)
        self.patch_size = _triple(patch_size)
        self.mask_ratio = mask_ratio
        self.structure_weight = structure_weight
        self.structure_axis_weights = _normalize_axis_weights(structure_axis_weights)
        self.membrane_weight = float(membrane_weight)
        self.membrane_axis_weights = _normalize_axis_weights(membrane_axis_weights)
        self.membrane_clip = float(membrane_clip)
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, self.patch_size)
        grid = tuple(v // p for v, p in zip(self.volume_size, self.patch_size))
        self.grid_size = grid
        self.num_patches = math.prod(grid)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        patch_dim = in_channels * math.prod(self.patch_size)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, patch_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        pz, py, px = self.patch_size
        bsz, ch, depth, height, width = x.shape
        x = x.reshape(bsz, ch, depth // pz, pz, height // py, py, width // px, px)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(bsz, -1, ch * pz * py * px)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        pz, py, px = self.patch_size
        gd, gh, gw = self.grid_size
        bsz = patches.shape[0]
        ch = self.in_channels
        x = patches.reshape(bsz, gd, gh, gw, ch, pz, py, px)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshape(bsz, ch, gd * pz, gh * py, gw * px)

    def random_mask(self, bsz: int, device: torch.device) -> torch.Tensor:
        num_mask = max(1, int(round(self.mask_ratio * self.num_patches)))
        noise = torch.rand(bsz, self.num_patches, device=device)
        ids = noise.argsort(dim=1)
        mask = torch.zeros(bsz, self.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :num_mask], True)
        return mask

    def encode_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, grid = self.patch_embed(x)
        if grid != self.grid_size:
            raise ValueError(f"input grid {grid} does not match configured grid {self.grid_size}")
        return tokens + self.pos_embed, tokens

    def forward(
        self,
        x: torch.Tensor,
        decision_module: DecisionModule | None = None,
        target_mask_ratio: float | None = None,
        deterministic_policy: bool = False,
    ) -> MAEOutput:
        target = self.patchify(x)
        tokens, raw_tokens = self.encode_tokens(x)
        decision = None
        if decision_module is None:
            mask = self.random_mask(x.shape[0], x.device)
        else:
            decision = decision_module(
                raw_tokens.detach(),
                target_mask_ratio=target_mask_ratio,
                deterministic=deterministic_policy,
            )
            mask = decision["mask"]
        encoded = torch.where(mask.unsqueeze(-1), self.mask_token + self.pos_embed, tokens)
        for block in self.encoder_blocks:
            encoded = block(encoded)
        encoded = self.norm(encoded)
        pred = self.decoder(encoded)
        per_patch = (pred - target).pow(2).mean(dim=-1)
        patch_weight = mask.new_ones(per_patch.shape, dtype=per_patch.dtype)
        if self.membrane_weight > 0.0:
            edge = membrane_edge_map_3d(
                x,
                axis_weights=self.membrane_axis_weights,
                clip=self.membrane_clip,
            )
            patch_weight = 1.0 + self.membrane_weight * self.patchify(edge).mean(dim=-1)
        weighted_mask = mask.float() * patch_weight
        pixel_loss_per_sample = (per_patch * weighted_mask).sum(dim=1) / weighted_mask.sum(dim=1).clamp_min(1.0)
        pixel_loss = pixel_loss_per_sample.mean()
        pred_volume = self.unpatchify(pred)
        structure_loss = gradient_loss_3d(pred_volume, x, axis_weights=self.structure_axis_weights)
        loss = pixel_loss + self.structure_weight * structure_loss
        return MAEOutput(
            loss=loss,
            pixel_loss=pixel_loss,
            structure_loss=structure_loss,
            membrane_weight_mean=patch_weight.detach().mean(),
            pred=pred_volume,
            target=x,
            mask=mask,
            decision=decision,
        )


class MAEBackboneAffinityNet(nn.Module):
    """Patch MAE encoder with a light transposed-conv affinity head."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        volume_size: tuple[int, int, int] = (16, 64, 64),
        patch_size: tuple[int, int, int] = (4, 16, 16),
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.volume_size = _triple(volume_size)
        self.patch_size = _triple(patch_size)
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, self.patch_size)
        self.grid_size = tuple(v // p for v, p in zip(self.volume_size, self.patch_size))
        self.num_patches = math.prod(self.grid_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.ConvTranspose3d(
            embed_dim,
            out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, grid = self.patch_embed(x)
        if grid != self.grid_size:
            raise ValueError(f"input grid {grid} does not match configured grid {self.grid_size}")
        tokens = tokens + self.pos_embed
        for block in self.encoder_blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        bsz, _, channels = tokens.shape
        gd, gh, gw = self.grid_size
        feat = tokens.transpose(1, 2).reshape(bsz, channels, gd, gh, gw)
        return self.head(feat)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.residual(x))


class ResidualAnisotropicBlock3D(nn.Module):
    """EM-friendly residual block that emphasizes XY membranes before Z context."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class EMAffinityHead3D(nn.Module):
    """Separate z and xy affinity heads with learnable channel calibration."""

    def __init__(
        self,
        channels: int,
        out_channels: int = 3,
        dropout: float = 0.0,
        refine_depth: int = 2,
        channel_bias_init: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if out_channels < 3:
            raise ValueError(f"EMAffinityHead3D expects at least z/y/x output channels, got {out_channels}")
        self.out_channels = int(out_channels)
        self.shared = nn.Sequential(
            *[ResidualAnisotropicBlock3D(channels, dropout=dropout) for _ in range(max(1, int(refine_depth)))]
        )
        self.z_refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )
        self.xy_refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )
        self.z_head = nn.Conv3d(channels, 1, kernel_size=1)
        self.xy_head = nn.Conv3d(channels, 2, kernel_size=1)
        self.extra_head = (
            nn.Conv3d(channels, self.out_channels - 3, kernel_size=1)
            if self.out_channels > 3
            else nn.Identity()
        )
        bias = torch.zeros(self.out_channels, dtype=torch.float32)
        if channel_bias_init is not None:
            values = [float(v) for v in channel_bias_init]
            if len(values) > self.out_channels:
                raise ValueError(
                    f"channel_bias_init has {len(values)} values for {self.out_channels} output channels"
                )
            bias[: len(values)] = torch.tensor(values, dtype=torch.float32)
        self.channel_bias = nn.Parameter(bias.view(1, self.out_channels, 1, 1, 1))
        self.channel_scale = nn.Parameter(torch.ones(1, self.out_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        parts = [self.z_head(self.z_refine(feat)), self.xy_head(self.xy_refine(feat))]
        if self.out_channels > 3:
            parts.append(self.extra_head(feat))
        logits = torch.cat(parts, dim=1)
        return logits * self.channel_scale + self.channel_bias


class TokenProjectUpBlock3D(nn.Module):
    """Project ViT tokens to a 3D feature map and upsample like MONAI UnetrPrUpBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_upsamples: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
        ]
        for _ in range(int(num_upsamples)):
            layers.extend(
                [
                    nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.GELU(),
                    ResidualConvBlock3D(out_channels, out_channels, dropout=dropout),
                ]
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConcatBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.conv = ResidualConvBlock3D(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if tuple(x.shape[-3:]) != tuple(skip.shape[-3:]):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNETRAffinityNet(nn.Module):
    """UNETR-style affinity model initialized from the dbMiM ViT encoder.

    The encoder module/key names intentionally match ``DBMIM3DMAE`` so that
    pretraining checkpoints load into the segmentation backbone. Transformer
    hidden states are fused at patch-grid resolution and decoded with a
    convolutional U-Net style refinement path to z/y/x affinity logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        volume_size: tuple[int, int, int] = (16, 64, 64),
        patch_size: tuple[int, int, int] = (4, 16, 16),
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        feature_size: int = 32,
        dropout: float = 0.0,
        skip_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.volume_size = _triple(volume_size)
        self.patch_size = _triple(patch_size)
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, self.patch_size)
        self.grid_size = tuple(v // p for v, p in zip(self.volume_size, self.patch_size))
        self.num_patches = math.prod(self.grid_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if skip_indices is None:
            raw = {
                max(0, depth // 4 - 1),
                max(0, depth // 2 - 1),
                max(0, (3 * depth) // 4 - 1),
            }
            skip_indices = sorted(raw)
        self.skip_indices = tuple(int(i) for i in skip_indices if 0 <= int(i) < depth)
        self.final_proj = nn.Sequential(
            nn.Conv3d(embed_dim, feature_size * 4, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feature_size * 4, affine=True),
            nn.GELU(),
            ConvBlock3D(feature_size * 4, feature_size * 4, dropout=dropout),
        )
        self.up_late = nn.ConvTranspose3d(
            feature_size * 4,
            feature_size * 4,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.skip_late_up = nn.Sequential(
            nn.ConvTranspose3d(
                embed_dim,
                feature_size * 4,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                bias=False,
            ),
            nn.InstanceNorm3d(feature_size * 4, affine=True),
            nn.GELU(),
        )
        self.decode_late = ConvBlock3D(feature_size * 8, feature_size * 4, dropout=dropout)
        self.up_mid = nn.ConvTranspose3d(
            feature_size * 4,
            feature_size * 2,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
        )
        self.skip_mid_up = nn.Sequential(
            nn.ConvTranspose3d(
                embed_dim,
                feature_size * 2,
                kernel_size=(2, 4, 4),
                stride=(2, 4, 4),
                bias=False,
            ),
            nn.InstanceNorm3d(feature_size * 2, affine=True),
            nn.GELU(),
        )
        self.decode_mid = ConvBlock3D(feature_size * 4, feature_size * 2, dropout=dropout)
        full_stride = (
            max(1, self.patch_size[0] // 2),
            max(1, self.patch_size[1] // 4),
            max(1, self.patch_size[2] // 4),
        )
        self.up_full = nn.ConvTranspose3d(
            feature_size * 2,
            feature_size,
            kernel_size=full_stride,
            stride=full_stride,
        )
        self.input_skip = ConvBlock3D(in_channels, feature_size, dropout=dropout)
        self.decode_full = ConvBlock3D(feature_size * 2, feature_size, dropout=dropout)
        self.head = nn.Conv3d(feature_size, out_channels, kernel_size=1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, _, channels = tokens.shape
        gd, gh, gw = self.grid_size
        return tokens.transpose(1, 2).reshape(bsz, channels, gd, gh, gw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, grid = self.patch_embed(x)
        if grid != self.grid_size:
            raise ValueError(f"input grid {grid} does not match configured grid {self.grid_size}")
        tokens = tokens + self.pos_embed
        hidden: list[torch.Tensor] = []
        for idx, block in enumerate(self.encoder_blocks):
            tokens = block(tokens)
            if idx in self.skip_indices:
                hidden.append(tokens)
        tokens = self.norm(tokens)
        if len(hidden) >= 2:
            mid_tokens, late_tokens = hidden[-2], hidden[-1]
        elif len(hidden) == 1:
            mid_tokens = late_tokens = hidden[-1]
        else:
            mid_tokens = late_tokens = tokens
        decoded = self.final_proj(self._tokens_to_grid(tokens))
        decoded = self.decode_late(
            torch.cat(
                [
                    self.up_late(decoded),
                    self.skip_late_up(self._tokens_to_grid(late_tokens)),
                ],
                dim=1,
            )
        )
        decoded = self.decode_mid(
            torch.cat(
                [
                    self.up_mid(decoded),
                    self.skip_mid_up(self._tokens_to_grid(mid_tokens)),
                ],
                dim=1,
            )
        )
        decoded = self.up_full(decoded)
        if tuple(decoded.shape[-3:]) != tuple(x.shape[-3:]):
            decoded = F.interpolate(decoded, size=x.shape[-3:], mode="trilinear", align_corners=False)
        skip = self.input_skip(x)
        return self.head(self.decode_full(torch.cat([decoded, skip], dim=1)))


def _default_unetr_skip_indices(depth: int) -> tuple[int, int, int]:
    if depth <= 0:
        raise ValueError("UNETR depth must be positive")
    raw = [depth // 4, depth // 2, (3 * depth) // 4]
    return tuple(max(0, min(depth - 1, int(idx))) for idx in raw)  # type: ignore[return-value]


def _normalize_skip_indices(skip_indices: Sequence[int] | None, depth: int) -> tuple[int, int, int]:
    if skip_indices is None:
        return _default_unetr_skip_indices(depth)
    indices = [max(0, min(depth - 1, int(idx))) for idx in skip_indices]
    if not indices:
        indices = list(_default_unetr_skip_indices(depth))
    while len(indices) < 3:
        indices.append(indices[-1])
    return tuple(indices[:3])  # type: ignore[return-value]


class UNETRAnisotropicAffinityNet(nn.Module):
    """Paper-style UNETR affinity model with the original anisotropic z decoder.

    This migrates the old ``model_unetr.py`` topology without depending on
    MONAI or the private ViT path. The ViT encoder key names intentionally match
    ``DBMIM3DMAE`` so dbMiM pretraining can initialize the backbone. The decoder
    mirrors the original skip path:

    ``encoder2/3/4`` upsample hidden states by 3/2/1 times, ``decoder5/4/3``
    bring the final tokens back to the high-resolution patch grid, and the
    ``dtrans`` convolution compresses z by ``patch_y / patch_z`` before the
    final upsample to full resolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        volume_size: tuple[int, int, int] = (32, 160, 160),
        patch_size: tuple[int, int, int] = (4, 16, 16),
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        feature_size: int = 32,
        dropout: float = 0.0,
        skip_indices: Sequence[int] | None = None,
        use_dtrans: bool | None = None,
        dtrans_stride_z: int | None = None,
    ) -> None:
        super().__init__()
        self.volume_size = _triple(volume_size)
        self.patch_size = _triple(patch_size)
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, self.patch_size)
        self.grid_size = tuple(v // p for v, p in zip(self.volume_size, self.patch_size))
        self.num_patches = math.prod(self.grid_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.skip_indices = _normalize_skip_indices(skip_indices, depth)

        self.encoder1 = ResidualConvBlock3D(in_channels, feature_size, dropout=dropout)
        self.encoder2 = TokenProjectUpBlock3D(embed_dim, feature_size * 2, num_upsamples=3, dropout=dropout)
        self.encoder3 = TokenProjectUpBlock3D(embed_dim, feature_size * 4, num_upsamples=2, dropout=dropout)
        self.encoder4 = TokenProjectUpBlock3D(embed_dim, feature_size * 8, num_upsamples=1, dropout=dropout)

        self.decoder5 = UpConcatBlock3D(embed_dim, feature_size * 8, feature_size * 8, dropout=dropout)
        self.decoder4 = UpConcatBlock3D(feature_size * 8, feature_size * 4, feature_size * 4, dropout=dropout)
        self.decoder3 = UpConcatBlock3D(feature_size * 4, feature_size * 2, feature_size * 2, dropout=dropout)
        self.decoder2 = UpConcatBlock3D(feature_size * 2, feature_size, feature_size, dropout=dropout)

        default_stride_z = max(1, int(self.patch_size[1] / self.patch_size[0]))
        stride_z = default_stride_z if dtrans_stride_z is None else max(1, int(dtrans_stride_z))
        self.use_dtrans = (self.patch_size[0] != self.patch_size[1]) if use_dtrans is None else bool(use_dtrans)
        self.dtrans = nn.Conv3d(
            feature_size * 2,
            feature_size * 2,
            kernel_size=(3, 3, 3),
            stride=(stride_z, 1, 1),
            padding=(1, 1, 1),
            bias=False,
        )
        self.head = nn.Conv3d(feature_size, out_channels, kernel_size=1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, _, channels = tokens.shape
        gd, gh, gw = self.grid_size
        return tokens.transpose(1, 2).reshape(bsz, channels, gd, gh, gw)

    def decode_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens, grid = self.patch_embed(x)
        if grid != self.grid_size:
            raise ValueError(f"input grid {grid} does not match configured grid {self.grid_size}")
        tokens = tokens + self.pos_embed
        hidden_by_index: dict[int, torch.Tensor] = {}
        for idx, block in enumerate(self.encoder_blocks):
            tokens = block(tokens)
            if idx in self.skip_indices:
                hidden_by_index[idx] = tokens
        tokens = self.norm(tokens)

        x2, x3, x4 = [hidden_by_index.get(idx, tokens) for idx in self.skip_indices]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self._tokens_to_grid(x2))
        enc3 = self.encoder3(self._tokens_to_grid(x3))
        enc4 = self.encoder4(self._tokens_to_grid(x4))

        dec4 = self._tokens_to_grid(tokens)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        if self.use_dtrans:
            dec1 = self.dtrans(dec1)
        out = self.decoder2(dec1, enc1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decode_features(x)
        return self.head(out)


class UNETREMAffinityNet(UNETRAnisotropicAffinityNet):
    """Anisotropic UNETR variant for EM affinity prediction.

    The transformer encoder and positional embedding names are inherited from
    ``UNETRAnisotropicAffinityNet``/``DBMIM3DMAE`` so dbMiM pretraining loads
    unchanged. The segmentation-specific change is confined to the decoder
    output head: extra XY-heavy residual refinement models membrane continuity,
    while a separate z-affinity path lets the network learn more conservative
    cross-section linking for anisotropic EM stacks.
    """

    def __init__(
        self,
        *args,
        em_refine_depth: int = 2,
        channel_bias_init: Sequence[float] | None = None,
        **kwargs,
    ) -> None:
        out_channels = int(kwargs.get("out_channels", 3))
        feature_size = int(kwargs.get("feature_size", 32))
        dropout = float(kwargs.get("dropout", 0.0))
        super().__init__(*args, **kwargs)
        self.head = EMAffinityHead3D(
            feature_size,
            out_channels=out_channels,
            dropout=dropout,
            refine_depth=em_refine_depth,
            channel_bias_init=channel_bias_init,
        )


class LearnableAffinityPostProcessor(nn.Module):
    """Differentiable local affinity calibration used as train-time postprocess.

    Waterz/connected-components stay non-differentiable at evaluation time. This
    module learns the part that is actually continuous: z/xy affinity scaling,
    biasing, and a local smoothing gate before threshold/agglomeration. The
    calibrated logits can be supervised end-to-end and used for metric-aligned
    regularization without replacing the final waterz sweep.
    """

    def __init__(
        self,
        channels: int = 3,
        init_bias: Sequence[float] | None = None,
        init_scale: Sequence[float] | None = None,
        smooth_kernel: int = 3,
        residual_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError("LearnableAffinityPostProcessor requires channels > 0")
        bias = torch.zeros(self.channels, dtype=torch.float32)
        if init_bias is not None:
            values = [float(v) for v in init_bias]
            if len(values) > self.channels:
                raise ValueError(f"init_bias has {len(values)} values for {self.channels} channels")
            bias[: len(values)] = torch.tensor(values, dtype=torch.float32)
        scale = torch.ones(self.channels, dtype=torch.float32)
        if init_scale is not None:
            values = [float(v) for v in init_scale]
            if len(values) > self.channels:
                raise ValueError(f"init_scale has {len(values)} values for {self.channels} channels")
            scale[: len(values)] = torch.tensor(values, dtype=torch.float32)
        self.log_scale = nn.Parameter(scale.clamp_min(1e-3).log().view(1, self.channels, 1, 1, 1))
        self.bias = nn.Parameter(bias.view(1, self.channels, 1, 1, 1))
        self.residual_weight = nn.Parameter(torch.tensor(float(residual_weight), dtype=torch.float32))
        self.smooth_kernel = max(1, int(smooth_kernel))
        self.local_refine = nn.Conv3d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=False,
        )
        with torch.no_grad():
            self.local_refine.weight.zero_()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if int(logits.shape[1]) != self.channels:
            raise ValueError(f"expected {self.channels} affinity channels, got {int(logits.shape[1])}")
        calibrated = logits * self.log_scale.exp() + self.bias
        if self.smooth_kernel > 1:
            prob = torch.sigmoid(calibrated)
            smooth = F.avg_pool3d(
                prob,
                kernel_size=self.smooth_kernel,
                stride=1,
                padding=self.smooth_kernel // 2,
            )
            residual = self.local_refine(prob - smooth)
            calibrated = calibrated + self.residual_weight.tanh() * residual
        return calibrated


class LearnedRAGMergeScorer(nn.Module):
    """Small MLP that scores whether neighboring watershed fragments should merge."""

    def __init__(
        self,
        in_features: int = 13,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = int(in_features)
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(last, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            last = int(hidden_dim)
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class DecoderAwareDBMIM3DMAE(UNETREMAffinityNet):
    """dbMiM variant that pretrains both encoder and EM decoder/head.

    The original dbMiM checkpoint transfers only the ViT encoder into the
    segmentation network. This variant keeps the masked-image reconstruction
    objective, but also predicts a differentiable membrane-derived affinity
    proxy with the exact UNETR-EM decoder/head names used during finetuning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        volume_size: tuple[int, int, int] = (32, 160, 160),
        patch_size: tuple[int, int, int] = (4, 16, 16),
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        feature_size: int = 32,
        decoder_dim: int = 192,
        mask_ratio: float = 0.75,
        structure_weight: float = 0.1,
        structure_axis_weights: Sequence[float] | None = None,
        membrane_weight: float = 0.0,
        membrane_axis_weights: Sequence[float] | None = None,
        membrane_clip: float = 5.0,
        affinity_weight: float = 0.35,
        affinity_temperature: float = 1.0,
        affinity_axis_weights: Sequence[float] | None = None,
        affinity_membrane_weight: float = 0.0,
        dropout: float = 0.0,
        skip_indices: Sequence[int] | None = None,
        em_refine_depth: int = 2,
        channel_bias_init: Sequence[float] | None = None,
        use_dtrans: bool | None = None,
        dtrans_stride_z: int | None = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            volume_size=volume_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            feature_size=feature_size,
            dropout=dropout,
            skip_indices=skip_indices,
            use_dtrans=use_dtrans,
            dtrans_stride_z=dtrans_stride_z,
            em_refine_depth=em_refine_depth,
            channel_bias_init=channel_bias_init,
        )
        self.in_channels = int(in_channels)
        self.mask_ratio = float(mask_ratio)
        self.structure_weight = float(structure_weight)
        self.structure_axis_weights = _normalize_axis_weights(structure_axis_weights)
        self.membrane_weight = float(membrane_weight)
        self.membrane_axis_weights = _normalize_axis_weights(membrane_axis_weights)
        self.membrane_clip = float(membrane_clip)
        self.affinity_weight = float(affinity_weight)
        self.affinity_temperature = float(affinity_temperature)
        self.affinity_axis_weights = _normalize_axis_weights(affinity_axis_weights)
        self.affinity_membrane_weight = float(affinity_membrane_weight)
        patch_dim = self.in_channels * math.prod(self.patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(embed_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, patch_dim),
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for module in self.reconstruction_decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        pz, py, px = self.patch_size
        bsz, ch, depth, height, width = x.shape
        x = x.reshape(bsz, ch, depth // pz, pz, height // py, py, width // px, px)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(bsz, -1, ch * pz * py * px)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        pz, py, px = self.patch_size
        gd, gh, gw = self.grid_size
        bsz = patches.shape[0]
        x = patches.reshape(bsz, gd, gh, gw, self.in_channels, pz, py, px)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return x.reshape(bsz, self.in_channels, gd * pz, gh * py, gw * px)

    def random_mask(self, bsz: int, device: torch.device) -> torch.Tensor:
        num_mask = max(1, int(round(self.mask_ratio * self.num_patches)))
        noise = torch.rand(bsz, self.num_patches, device=device)
        ids = noise.argsort(dim=1)
        mask = torch.zeros(bsz, self.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :num_mask], True)
        return mask

    def _pseudo_affinity_target(self, x: torch.Tensor) -> torch.Tensor:
        channels = []
        weights = self.affinity_axis_weights
        for dim, weight in zip((-3, -2, -1), weights):
            grad = x.diff(dim=dim).abs()
            pad_shape = list(grad.shape)
            pad_shape[dim] = 1
            grad = torch.cat([x.new_zeros(pad_shape), grad], dim=dim)
            reduce_dims = tuple(range(1, grad.ndim))
            grad = grad / grad.mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            affinity = torch.exp(-float(weight) * float(self.affinity_temperature) * grad)
            channels.append(affinity)
        return torch.cat(channels, dim=1).clamp(0.0, 1.0)

    def _encode_with_mask(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        tokens, grid = self.patch_embed(x)
        if grid != self.grid_size:
            raise ValueError(f"input grid {grid} does not match configured grid {self.grid_size}")
        tokens = tokens + self.pos_embed
        encoded = torch.where(mask.unsqueeze(-1), self.mask_token + self.pos_embed, tokens)
        hidden_by_index: dict[int, torch.Tensor] = {}
        for idx, block in enumerate(self.encoder_blocks):
            encoded = block(encoded)
            if idx in self.skip_indices:
                hidden_by_index[idx] = encoded
        return self.norm(encoded), hidden_by_index

    def _decode_features_from_tokens(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        hidden_by_index: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        x2, x3, x4 = [hidden_by_index.get(idx, tokens) for idx in self.skip_indices]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self._tokens_to_grid(x2))
        enc3 = self.encoder3(self._tokens_to_grid(x3))
        enc4 = self.encoder4(self._tokens_to_grid(x4))
        dec4 = self._tokens_to_grid(tokens)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        if self.use_dtrans:
            dec1 = self.dtrans(dec1)
        return self.decoder2(dec1, enc1)

    def forward(
        self,
        x: torch.Tensor,
        decision_module: DecisionModule | None = None,
        target_mask_ratio: float | None = None,
        deterministic_policy: bool = False,
    ) -> MAEOutput:
        target = self.patchify(x)
        raw_tokens, grid = self.patch_embed(x)
        if grid != self.grid_size:
            raise ValueError(f"input grid {grid} does not match configured grid {self.grid_size}")
        decision = None
        if decision_module is None:
            mask = self.random_mask(x.shape[0], x.device)
        else:
            decision = decision_module(
                raw_tokens.detach(),
                target_mask_ratio=target_mask_ratio,
                deterministic=deterministic_policy,
            )
            mask = decision["mask"]
        encoded, hidden_by_index = self._encode_with_mask(x, mask)
        pred_patches = self.reconstruction_decoder(encoded)
        per_patch = (pred_patches - target).pow(2).mean(dim=-1)
        patch_weight = mask.new_ones(per_patch.shape, dtype=per_patch.dtype)
        if self.membrane_weight > 0.0:
            edge = membrane_edge_map_3d(
                x,
                axis_weights=self.membrane_axis_weights,
                clip=self.membrane_clip,
            )
            patch_weight = 1.0 + self.membrane_weight * self.patchify(edge).mean(dim=-1)
        weighted_mask = mask.float() * patch_weight
        pixel_loss_per_sample = (per_patch * weighted_mask).sum(dim=1) / weighted_mask.sum(dim=1).clamp_min(1.0)
        pixel_loss = pixel_loss_per_sample.mean()
        pred_volume = self.unpatchify(pred_patches)
        structure_loss = gradient_loss_3d(pred_volume, x, axis_weights=self.structure_axis_weights)
        features = self._decode_features_from_tokens(x, encoded, hidden_by_index)
        affinity_logits = self.head(features)
        affinity_target = self._pseudo_affinity_target(x)
        affinity_error = (torch.sigmoid(affinity_logits) - affinity_target).pow(2)
        if self.affinity_membrane_weight > 0.0:
            affinity_edge = membrane_edge_map_3d(
                x,
                axis_weights=self.membrane_axis_weights,
                clip=self.membrane_clip,
            )
            affinity_edge_weight = 1.0 + self.affinity_membrane_weight * affinity_edge
            denom = affinity_edge_weight.sum().clamp_min(1.0) * affinity_error.shape[1]
            affinity_loss = (affinity_error * affinity_edge_weight).sum() / denom
        else:
            affinity_loss = affinity_error.mean()
        loss = pixel_loss + self.structure_weight * structure_loss + self.affinity_weight * affinity_loss
        return MAEOutput(
            loss=loss,
            pixel_loss=pixel_loss,
            structure_loss=structure_loss,
            membrane_weight_mean=patch_weight.detach().mean(),
            pred=pred_volume,
            target=x,
            mask=mask,
            decision=decision,
            affinity_loss=affinity_loss,
            affinity_pred=torch.sigmoid(affinity_logits),
            affinity_target=affinity_target,
        )


def gradient_loss_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    axis_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    weights = _normalize_axis_weights(axis_weights)
    weight_sum = sum(max(float(v), 0.0) for v in weights)
    for dim, weight in zip((-3, -2, -1), weights):
        if weight <= 0.0:
            continue
        pred_grad = pred.diff(dim=dim)
        target_grad = target.diff(dim=dim)
        loss = loss + float(weight) * F.l1_loss(pred_grad, target_grad)
    return loss / max(weight_sum, 1e-6)


def load_pretrained_backbone(model: nn.Module, checkpoint: dict[str, Any]) -> list[str]:
    state = checkpoint.get("model", checkpoint)
    own = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key in own and own[key].shape == value.shape:
            loaded[key] = value
        elif key == "pos_embed" and key in own:
            interpolated = _interpolate_pos_embed_for_model(value, own[key], model, checkpoint)
            if interpolated is not None:
                loaded[key] = interpolated
    model.load_state_dict(loaded, strict=False)
    return sorted(loaded.keys())


def _grid_from_model_cfg(cfg: dict[str, Any] | None) -> tuple[int, int, int] | None:
    if not isinstance(cfg, dict):
        return None
    volume = cfg.get("volume_size")
    patch = cfg.get("patch_size")
    if volume is None or patch is None:
        return None
    if len(volume) != 3 or len(patch) != 3:
        return None
    return tuple(int(v) // int(p) for v, p in zip(volume, patch))


def _factor_grid(num_patches: int) -> tuple[int, int, int] | None:
    # Prefer anisotropic CREMI-like grids where z is the smallest dimension.
    best: tuple[int, int, int] | None = None
    best_score = float("inf")
    for d in range(1, int(round(num_patches ** (1.0 / 3.0))) + 8):
        if num_patches % d != 0:
            continue
        rest = num_patches // d
        for h in range(1, int(math.sqrt(rest)) + 1):
            if rest % h != 0:
                continue
            w = rest // h
            dims = sorted((d, h, w))
            score = (dims[2] - dims[1]) + 2 * (dims[1] - dims[0])
            if score < best_score:
                best_score = float(score)
                best = (dims[0], dims[1], dims[2])
    return best


def _interpolate_pos_embed_for_model(
    value: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module,
    checkpoint: dict[str, Any],
) -> torch.Tensor | None:
    if value.ndim != 3 or target.ndim != 3:
        return None
    if value.shape[0] != 1 or target.shape[0] != 1:
        return None
    if value.shape[-1] != target.shape[-1]:
        return None
    if value.shape[1] == target.shape[1]:
        return value
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    old_grid = _grid_from_model_cfg(cfg.get("model") if isinstance(cfg, dict) else None)
    if old_grid is None or math.prod(old_grid) != int(value.shape[1]):
        old_grid = _factor_grid(int(value.shape[1]))
    new_grid = getattr(model, "grid_size", None)
    if new_grid is None or math.prod(tuple(int(v) for v in new_grid)) != int(target.shape[1]):
        new_grid = _factor_grid(int(target.shape[1]))
    if old_grid is None or new_grid is None:
        return None
    channels = int(value.shape[-1])
    pos = value.reshape(1, old_grid[0], old_grid[1], old_grid[2], channels).permute(0, 4, 1, 2, 3)
    pos = F.interpolate(pos.float(), size=tuple(int(v) for v in new_grid), mode="trilinear", align_corners=False)
    pos = pos.permute(0, 2, 3, 4, 1).reshape(1, int(target.shape[1]), channels)
    return pos.to(dtype=target.dtype)
