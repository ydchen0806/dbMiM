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
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1).mean(dim=1)
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
    pred: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    decision: dict[str, torch.Tensor] | None


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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.volume_size = _triple(volume_size)
        self.patch_size = _triple(patch_size)
        self.mask_ratio = mask_ratio
        self.structure_weight = structure_weight
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
        pixel_loss_per_sample = (per_patch * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
        pixel_loss = pixel_loss_per_sample.mean()
        pred_volume = self.unpatchify(pred)
        structure_loss = gradient_loss_3d(pred_volume, x)
        loss = pixel_loss + self.structure_weight * structure_loss
        return MAEOutput(
            loss=loss,
            pixel_loss=pixel_loss,
            structure_loss=structure_loss,
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


def gradient_loss_3d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    for dim in (-3, -2, -1):
        pred_grad = pred.diff(dim=dim)
        target_grad = target.diff(dim=dim)
        loss = loss + F.l1_loss(pred_grad, target_grad)
    return loss / 3.0


def load_pretrained_backbone(model: nn.Module, checkpoint: dict[str, Any]) -> list[str]:
    state = checkpoint.get("model", checkpoint)
    own = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key in own and own[key].shape == value.shape:
            loaded[key] = value
    model.load_state_dict(loaded, strict=False)
    return sorted(loaded.keys())
