# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from model_superhuman2 import UNet_PNI, UNet_PNI_Noskip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from unet3d_mala import UNet3D_MALA
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset_hdf import hdfDataset
import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from dataloader.data_provider_pretraining import Train as EMDataset
import sys
sys.path.append('/data/ydchen/VLP/EM_Mamba/mambamae_EM')
import util_mae.misc as misc
from util_mae.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from segmamba import SegMamba
from model_unetr import UNETR
from omegaconf import OmegaConf
from engine_pretrain import train_one_epoch, train_one_epoch_with_rl
from torch.distributions import Categorical

class DecisionModule(nn.Module):
    """Decision module that uses multi-agent reinforcement learning 
    to determine masking strategy for patches in MAE"""
    
    def __init__(self, feature_dim=768, hidden_dim=256, num_patches=196, min_mask_ratio=0.4, max_mask_ratio=0.9):
        super().__init__()
        self.num_patches = num_patches
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        
        # Feature extractor (shared across agents)
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 actions: mask or keep
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Global context network to enable coordination between agents
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Parameters for PPO-style training
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01  # Encourage exploration
        
        # Discount factor for reward calculation
        self.gamma = 0.99
        
        # Adaptive masking ratio control
        self.target_mask_ratio = 0.75  # Default target, can be overridden
        self.mask_ratio_tolerance = 0.1  # Acceptable range around target
        
        # Memory for policy learning
        self.saved_actions = []
        self.saved_masks = []
        self.rewards = []
        self.entropies = []
        self.old_probs = []
        self.prev_loss = None
        
    def forward(self, patch_features, batch_mask_ratio=None):
        """
        Args:
            patch_features: Tensor of shape [batch_size, num_patches, feature_dim]
            batch_mask_ratio: Optional override of target mask ratio for this batch
        
        Returns:
            mask_decisions: Tensor of shape [batch_size, num_patches] with binary values (0: keep, 1: mask)
            log_probs: Log probabilities of chosen actions for policy learning
            entropy: Entropy of the policy for exploration
        """
        batch_size, num_patches, feature_dim = patch_features.shape
        
        # Extract features for all patches/agents
        features = self.feature_extractor(patch_features)
        
        # Compute global context for coordination between agents
        # Aggregate information across all patches
        global_feat = features.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]
        global_context = self.global_context(global_feat)
        
        # Add global context to each patch's feature
        features = features + global_context.expand(-1, num_patches, -1)
        
        # Actor: compute action probabilities
        action_logits = self.actor(features)
        probs = F.softmax(action_logits, dim=-1)
        
        # Critic: estimate state value
        values = self.critic(features).squeeze(-1)
        
        # Sample actions from the probability distribution
        m = Categorical(probs)
        actions = m.sample()
        
        # Calculate log probabilities and entropy
        log_probs = m.log_prob(actions)
        entropy = m.entropy().mean()
        
        # Store old probabilities for PPO
        old_probs = probs.detach()
        
        # Convert actions to mask decisions (0: keep, 1: mask)
        mask_decisions = actions
        
        # Check the masking ratio
        curr_mask_ratio = mask_decisions.float().mean()
        
        # Apply masking ratio constraints if needed
        if batch_mask_ratio is not None:
            target_ratio = batch_mask_ratio
        else:
            target_ratio = self.target_mask_ratio
            
        # If too many or too few patches are masked, enforce constraints
        if curr_mask_ratio < self.min_mask_ratio or curr_mask_ratio > self.max_mask_ratio:
            # Sort patches by their masking probability
            mask_probs = probs[:, :, 1]  # Probability of masking for each patch
            
            # Flatten for easier processing
            flat_probs = mask_probs.view(-1)
            flat_decisions = mask_decisions.view(-1)
            
            if curr_mask_ratio < self.min_mask_ratio:
                # Need to mask more patches
                # Find patches currently not masked with highest masking probability
                unmasked = (flat_decisions == 0)
                if unmasked.any():
                    unmasked_probs = flat_probs[unmasked]
                    unmasked_indices = torch.nonzero(unmasked).squeeze(-1)
                    
                    # Sort by probability (highest first)
                    _, indices = torch.sort(unmasked_probs, descending=True)
                    
                    # Calculate how many more patches to mask
                    target_count = int(self.min_mask_ratio * num_patches * batch_size)
                    current_count = flat_decisions.sum().item()
                    to_mask = min(target_count - current_count, len(indices))
                    
                    # Mask additional patches
                    if to_mask > 0:
                        to_flip = unmasked_indices[indices[:to_mask]]
                        flat_decisions[to_flip] = 1
                        
            elif curr_mask_ratio > self.max_mask_ratio:
                # Need to unmask some patches
                # Find patches currently masked with lowest masking probability
                masked = (flat_decisions == 1)
                if masked.any():
                    masked_probs = flat_probs[masked]
                    masked_indices = torch.nonzero(masked).squeeze(-1)
                    
                    # Sort by probability (lowest first)
                    _, indices = torch.sort(masked_probs)
                    
                    # Calculate how many patches to unmask
                    target_count = int(self.max_mask_ratio * num_patches * batch_size)
                    current_count = flat_decisions.sum().item()
                    to_unmask = min(current_count - target_count, len(indices))
                    
                    # Unmask some patches
                    if to_unmask > 0:
                        to_flip = masked_indices[indices[:to_unmask]]
                        flat_decisions[to_flip] = 0
            
            # Reshape back
            mask_decisions = flat_decisions.view(batch_size, num_patches)
        
        # Save for policy update
        self.saved_actions.append((log_probs, values))
        self.saved_masks.append((mask_decisions, log_probs, entropy, values))
        self.entropies.append(entropy)
        self.old_probs.append(old_probs)
        
        return mask_decisions, log_probs, entropy, values
    
    def get_reward(self, current_loss, prev_loss=None, mask_decisions=None):
        """Calculate reward based on reconstruction loss and masking ratio"""
        batch_size = current_loss.shape[0] if hasattr(current_loss, 'shape') else 1
        
        # Base reward from reconstruction difficulty
        if prev_loss is None:
            # For first iteration, use a neutral reward
            reconstruction_reward = torch.zeros(batch_size, device=current_loss.device)
        else:
            # Reward is positive if current loss is higher than previous (more challenging task)
            # But not too much higher (to prevent extreme masking)
            loss_diff = current_loss - prev_loss
            reconstruction_reward = torch.clamp(loss_diff, -2.0, 2.0)
        
        # Add mask ratio regularization reward
        mask_ratio_reward = torch.zeros_like(reconstruction_reward)
        if mask_decisions is not None:
            curr_mask_ratio = mask_decisions.float().mean()
            
            # Reward is higher when mask ratio is close to target
            ratio_diff = abs(curr_mask_ratio - self.target_mask_ratio)
            
            # No penalty if within tolerance range
            if ratio_diff <= self.mask_ratio_tolerance:
                mask_ratio_reward += 0.5
            else:
                # Gradually increasing penalty for deviation
                mask_ratio_reward -= ratio_diff * 2.0
        
        # Combine rewards
        total_reward = reconstruction_reward + mask_ratio_reward
        
        self.rewards.append(total_reward.detach())
        self.prev_loss = current_loss.detach()
        
        return total_reward
    
    def update_policy(self, optimizer, masks_ratio=None):
        """Update policy using PPO-style optimization"""
        if len(self.saved_actions) == 0:
            return 0, 0
            
        # Set target mask ratio if provided
        if masks_ratio is not None:
            self.target_mask_ratio = masks_ratio
            
        # Calculate returns and advantages
        R = 0
        policy_losses = []
        value_losses = []
        returns = []
        
        # Calculate returns
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.cat(returns).detach()
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Unpack saved data
        old_log_probs = torch.cat([lp for lp, _ in self.saved_actions])
        old_values = torch.cat([v for _, v in self.saved_actions])
        old_action_probs = torch.cat(self.old_probs)
        
        # Calculate advantages
        advantages = returns - old_values.detach()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        # PPO update (multiple epochs for better optimization)
        for _ in range(4):  # PPO typically uses multiple update epochs
            # Calculate PPO policy loss
            ratio = torch.exp(old_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(old_values, returns)
            
            # Entropy loss (to encourage exploration)
            entropy_loss = -torch.cat(self.entropies).mean()
            
            # Combine losses with coefficients
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            optimizer.step()
        
        # Calculate average masking ratio for monitoring
        mask_ratio = 0
        if len(self.saved_masks) > 0:
            mask_ratios = [mask.float().mean().item() for mask, _, _, _ in self.saved_masks]
            mask_ratio = sum(mask_ratios) / len(mask_ratios)
        
        # Clear memory
        self.saved_actions = []
        self.saved_masks = []
        self.rewards = []
        self.entropies = []
        self.old_probs = []
        
        return loss.item(), mask_ratio
        
    def set_mask_ratio_bounds(self, min_ratio, max_ratio):
        """Set bounds for the masking ratio"""
        self.min_mask_ratio = min_ratio
        self.max_mask_ratio = max_ratio

def calculate_hog_features(images, cell_size=8, block_size=2, nbins=9):
    """Calculate HOG features for the input images
    Simplified implementation for demonstration purposes"""
    batch_size, channels, depth, height, width = images.shape
    
    # Reshape for 2D processing if necessary (for 3D data)
    if len(images.shape) == 5:  # 3D data
        # Process each depth slice separately or average across depth
        # This is a simplified approach
        images = images.view(batch_size * depth, channels, height, width)
    
    # Calculate gradients
    gx = images[:, :, :, 1:] - images[:, :, :, :-1]
    gy = images[:, :, 1:, :] - images[:, :, :-1, :]
    
    # Pad gradients to match original size
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    
    # Calculate magnitude and orientation
    magnitude = torch.sqrt(gx**2 + gy**2)
    orientation = torch.atan2(gy, gx)
    
    # Convert orientation to degrees and shift to [0, 180)
    orientation = orientation * (180.0 / np.pi) % 180.0
    
    # Initialize HOG features
    hog_features = []
    
    # Create cell grid
    for i in range(0, height, cell_size):
        for j in range(0, width, cell_size):
            if i + cell_size > height or j + cell_size > width:
                continue
                
            # Get magnitude and orientation for this cell
            cell_magnitude = magnitude[:, :, i:i+cell_size, j:j+cell_size]
            cell_orientation = orientation[:, :, i:i+cell_size, j:j+cell_size]
            
            # Create histogram
            hist = torch.zeros(batch_size, channels, nbins, device=images.device)
            
            for b in range(nbins):
                bin_lower = b * (180.0 / nbins)
                bin_upper = (b + 1) * (180.0 / nbins)
                
                # Calculate weights for each bin
                weight = torch.zeros_like(cell_orientation)
                mask = (cell_orientation >= bin_lower) & (cell_orientation < bin_upper)
                weight[mask] = 1.0
                
                # Weighted sum of magnitudes
                hist[:, :, b] = (cell_magnitude * weight).sum(dim=(2, 3))
            
            hog_features.append(hist)
    
    # Concatenate all histograms
    if hog_features:
        hog_features = torch.cat(hog_features, dim=2)
        
    # Reshape back to original batch size if necessary
    if len(images.shape) == 5:
        hog_features = hog_features.view(batch_size, depth, -1)
    
    return hog_features


def get_args_parser():
    parser = argparse.ArgumentParser('pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='segmamba', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--fill_mode', default=0, type=int,
                        help='fill mode for the holes in the image')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.4, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    parser.add_argument('--target_mask_ratio', default=0.85, type=float,
                        help='Target masking ratio for the RL policy to converge to.')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # HOG-related parameters
    parser.add_argument('--if_hog', default=True, type=bool, 
                        help='Whether to use HOG features for multi-task reconstruction')
    parser.add_argument('--mse_weight', type=float, default=0.1,
                        help='Weight for MSE loss in multi-task learning')
    parser.add_argument('--hog_weight', type=float, default=1.0,
                        help='Weight for HOG loss in multi-task learning')
    parser.add_argument('--hog_orientations', type=int, default=9,
                        help='Number of orientation bins for HOG features')
    parser.add_argument('--hog_pixels_per_cell', type=int, default=8,
                        help='Pixels per cell for HOG feature extraction')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    parser.add_argument('--rl_lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate for the RL policy network')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--use_amp', type=bool, default=False,
                        help='use mixed precision training')
    
    # RL training parameters
    parser.add_argument('--joint_training_epochs', type=int, default=100,
                        help='number of epochs to jointly train MAE and decision module')
    parser.add_argument('--rl_update_freq', type=int, default=10,
                        help='frequency of RL policy updates (in iterations)')
    
    # Multi-task learning parameters
    parser.add_argument('--mse_weight', type=float, default=0.1,
                        help='weight for MSE loss in multi-task learning')
    parser.add_argument('--hog_weight', type=float, default=1.0,
                        help='weight for HOG loss in multi-task learning')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--hdf_path', default='/img_video/img/COCOunlabeled2017.hdf5', type=str,
                        help='dataset path')
    parser.add_argument('--EM_cfg_path', default='/data/ydchen/VLP/EM_Mamba/mambamae_EM/config/pretraining_all.yaml', type=str,
                        help='EM cfg dataset path')
    parser.add_argument('--output_dir', default='/h3cstore_ns/EM_pretrain/mamba_pretrain_MAE',
                        help='path where to save, empty for no saving')
    parser.add_argument('--visual_dir', default='/h3cstore_ns/EM_pretrain/mamba_pretrain_MAE/visual',
                        help='path where to save visual images')
    parser.add_argument('--log_dir', default='/h3cstore_ns/EM_pretrain/mamba_pretrain_MAE/tensorboard_log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--pretrain_path', default='', type=str,
                        help='path to pretrained model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--save_freq', default=10, type=int,
                       help='Frequency of saving checkpoints')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--if_hog', default=True, type=bool, help='if use hog feature')
    
    # Add decision module flag
    parser.add_argument('--use_decision_module', action='store_true',
                        help='Use the MARL decision module for adaptive masking')
    parser.add_argument('--decision_module_checkpoint', default='',
                        help='Path to a pretrained decision module checkpoint')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    hog_params = None
    if args.if_hog:
        hog_params = {
            'orientations': args.hog_orientations,
            'pixels_per_cell': (args.hog_pixels_per_cell, args.hog_pixels_per_cell),
            'cells_per_block': (2, 2),
            'visualize': True,
            'multichannel': False  # For grayscale EM data
        }
    
    # Load configuration for EM dataset
    cfg = OmegaConf.load(args.EM_cfg_path)
    dataset_train = EMDataset(cfg, args=args, if_hog=args.if_hog, hog_params=hog_params)
    print(f'total len of dataset: {len(dataset_train)}')
    # Set up sampler for distributed training
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # Set up logging
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Create data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # Define the model based on the specified architecture
    if args.model == 'segmamba':
        model = SegMamba(in_chans=1, out_chans=1)
    elif args.model == 'superhuman':
        model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                            out_planes=1,
                            filters=cfg.MODEL.filters,
                            upsample_mode=cfg.MODEL.upsample_mode,
                            decode_ratio=cfg.MODEL.decode_ratio,
                            merge_mode=cfg.MODEL.merge_mode,
                            pad_mode=cfg.MODEL.pad_mode,
                            bn_mode=cfg.MODEL.bn_mode,
                            relu_mode=cfg.MODEL.relu_mode,
                            init_mode=cfg.MODEL.init_mode)
    elif args.model == 'mala':
        model = UNet3D_MALA(output_nc=1, if_sigmoid=cfg.MODEL.if_sigmoid,
                    init_mode=cfg.MODEL.init_mode_mala)
    elif args.model == 'unetr':
        model = UNETR(
                in_channels=cfg.MODEL.input_nc,
                out_channels=1,
                img_size=cfg.MODEL.unetr_size,
                patch_size=cfg.MODEL.patch_size,
                feature_size=16,
                hidden_size=768,
                mlp_dim=2048,
                num_heads=8,
                pos_embed='perceptron',
                norm_name='instance',
                conv_block=True,
                res_block=True,
                kernel_size=cfg.MODEL.kernel_size,
                skip_connection=False,
                show_feature=False,
                dropout_rate=0.1)
    else:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        
    # Load pretrained weights if provided
    if args.pretrain_path:
        print("Loading pretrained model from %s" % args.pretrain_path)
        weights = torch.load(args.pretrain_path, map_location='cpu')
        model.load_state_dict(weights, strict=False)
    
    model.to(device)
    
    # Initialize the decision module for adaptive masking if enabled
    if args.use_decision_module:
        # Estimate number of patches based on model architecture
        if hasattr(model, 'patch_embed'):
            feature_dim = model.patch_embed.proj.out_channels
            img_size = model.patch_embed.img_size
            patch_size = model.patch_embed.patch_size
            if isinstance(img_size, tuple):
                num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            else:
                num_patches = (img_size // patch_size) ** 2
        else:
            # Default values if we can't determine from model
            feature_dim = 768
            num_patches = 196  # 14x14 patches
        
        decision_module = DecisionModule(
            feature_dim=feature_dim,
            hidden_dim=256,
            num_patches=num_patches
        )
        
        # Load pretrained decision module if provided
        if args.decision_module_checkpoint:
            print(f"Loading pretrained decision module from {args.decision_module_checkpoint}")
            decision_module.load_state_dict(
                torch.load(args.decision_module_checkpoint, map_location='cpu')
            )
            
        decision_module.to(device)
    else:
        decision_module = None
        
    # Set up distributed training
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
        
        if args.use_decision_module:
            decision_module = torch.nn.parallel.DistributedDataParallel(
                decision_module, device_ids=[args.gpu], find_unused_parameters=True
            )

    # Calculate effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Setup optimizers with appropriate params for multi-task learning
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    
    if args.use_decision_module:
        rl_optimizer = torch.optim.Adam(
            decision_module.parameters(), lr=args.rl_lr, betas=(0.9, 0.999)
        )
    else:
        rl_optimizer = None

    print(optimizer)
    loss_scaler = NativeScaler()

    # Load checkpoint if resuming training
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Create output and visualization directories
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.visual_dir:
        os.makedirs(args.visual_dir, exist_ok=True)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # Phase 1: Joint training of MAE and decision module
        if args.use_decision_module and epoch < args.joint_training_epochs:
            print(f"Epoch {epoch}: Joint training of MAE and decision module")
            train_stats = train_one_epoch_with_rl(
                model, decision_module, data_loader_train,
                optimizer, rl_optimizer, device, epoch, loss_scaler,
                log_writer=log_writer, args=args
            )
        # Phase 2: Continue training MAE with fixed decision module
        elif args.use_decision_module:
            print(f"Epoch {epoch}: Training MAE with fixed decision module")
            # Freeze decision module parameters
            for param in decision_module.parameters():
                param.requires_grad = False
                
            train_stats = train_one_epoch_with_rl(
                model, decision_module, data_loader_train,
                optimizer, None, device, epoch, loss_scaler,
                log_writer=log_writer, args=args
            )
        # Standard training without decision module
        else:
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer, args=args
            )
        
        # Save checkpoint
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs) and misc.is_main_process():
            if args.use_decision_module:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, 
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                    decision_module=decision_module.module if args.distributed else decision_module,
                    rl_optimizer=rl_optimizer
                )
            else:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, 
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
                )

        # Log statistics
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)