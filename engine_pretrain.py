# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import util_mae.misc as misc
import util_mae.lr_sched as lr_sched
import os
from torch.distributions import Categorical
from main_pretrain import DecisionModule


def calculate_hog_features(images, cell_size=8, block_size=2, nbins=9):
    """Calculate HOG features for the input images"""
    # This is a simplified HOG implementation
    # In a real implementation, you might want to use a dedicated library
    
    batch_size, channels, height, width = images.shape
    
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
        
    return hog_features


def train_one_epoch_with_rl(model: torch.nn.Module,
                          decision_module: torch.nn.Module,
                          data_loader: Iterable, 
                          optimizer: torch.optim.Optimizer,
                          rl_optimizer: torch.optim.Optimizer,
                          device: torch.device, 
                          epoch: int, 
                          loss_scaler,
                          log_writer=None,
                          args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mask_ratio', misc.SmoothedValue(window_size=20, fmt='{value:.3f}'))
    metric_logger.add_meter('rl_loss', misc.SmoothedValue(window_size=20, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    
    optimizer.zero_grad()
    rl_optimizer.zero_grad()
    
    if log_writer is not None:
        print('log_dir: {}'.format(args.log_dir))
    
    # For tracking decision module's learning progress
    rewards_history = []
    
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Adjust learning rate
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # Process samples based on whether HOG features are included
        if args.if_hog:
            augsamples, gtsamples, hogsamples = samples
            augsamples = augsamples.to(device, non_blocking=True)
            gtsamples = gtsamples.to(device, non_blocking=True)
            hogsamples = hogsamples.to(device, non_blocking=True)
        else:
            augsamples, gtsamples = samples
            augsamples = augsamples.to(device, non_blocking=True)
            gtsamples = gtsamples.to(device, non_blocking=True)
            hogsamples = None
        
        # Extract patch features from the encoder for decision making
        with torch.no_grad():
            patch_features = model.module.get_patch_features(augsamples)
        
        # Get masking decisions from the RL module
        mask_decisions, log_probs, entropy, _ = decision_module(patch_features)
        
        # Apply the mask decisions to the model
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            if args.if_hog:
                # Multi-task reconstruction with both pixel and HOG features
                mse_loss = model(augsamples, gtsamples, mask_decisions)
                hog_loss = model(augsamples, hogsamples, mask_decisions, target_type='hog')
                loss = args.mse_weight * mse_loss + args.hog_weight * hog_loss
            else:
                # Single task - pixel reconstruction only
                loss = model(augsamples, gtsamples, mask_decisions)
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            torch.cuda.empty_cache()
            continue
        
        # Calculate reward for the RL module
        reward = decision_module.get_reward(loss.detach())
        rewards_history.append(reward.item())
        
        # Update target network (MAE)
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # Add gradient clipping
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        # Update decision module periodically
        if (data_iter_step + 1) % args.rl_update_freq == 0:
            rl_loss, mask_ratio = decision_module.update_policy(rl_optimizer, rewards_history, 
                                                              masks_ratio=args.target_mask_ratio)
            rewards_history = []
            metric_logger.update(rl_loss=rl_loss)
            metric_logger.update(mask_ratio=mask_ratio)
        
        torch.cuda.synchronize()
        
        # Log metrics
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
            # Log RL-specific metrics
            if hasattr(metric_logger.meters, 'rl_loss'):
                log_writer.add_scalar('rl_loss', metric_logger.meters['rl_loss'].global_avg, epoch_1000x)
            if hasattr(metric_logger.meters, 'mask_ratio'):
                log_writer.add_scalar('mask_ratio', metric_logger.meters['mask_ratio'].global_avg, epoch_1000x)
            log_writer.add_scalar('policy_entropy', entropy.mean().item(), epoch_1000x)
        
        # Visualize reconstructions
        if data_iter_step == 0 and args.model != 'mala' and args.model != 'superhuman' and args.model != 'unetr' and args.model != 'segmamba':
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            psnr = model.module.recons_visualize(augsamples, gtsamples, epoch_1000x, save_dir=args.visual_dir, 
                                               mask_decisions=mask_decisions)
            if log_writer is not None:
                log_writer.add_scalar('train_psnr', psnr, epoch_1000x)
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(args.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # Process samples based on whether HOG features are included
        if args.if_hog:
            augsamples, gtsamples, hogsamples = samples
            augsamples = augsamples.to(device, non_blocking=True)
            gtsamples = gtsamples.to(device, non_blocking=True)
            hogsamples = hogsamples.to(device, non_blocking=True)
        else:
            augsamples, gtsamples = samples
            augsamples = augsamples.to(device, non_blocking=True)
            gtsamples = gtsamples.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            if args.if_hog:
                # Multi-task reconstruction with both pixel and HOG features
                mse_loss = model(augsamples, gtsamples)
                hog_loss = model(augsamples, hogsamples, target_type='hog')
                loss = args.mse_weight * mse_loss + args.hog_weight * hog_loss
            else:
                loss = model(augsamples, gtsamples)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            torch.cuda.empty_cache()
            continue

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # add gradient clipping
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
        if data_iter_step == 0 and args.model != 'mala' and args.model != 'superhuman' and args.model != 'unetr' and args.model != 'segmamba':
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            psnr = model.module.recons_visualize(augsamples, gtsamples, epoch_1000x, save_dir = args.visual_dir)
            if log_writer is not None:
                log_writer.add_scalar('train_psnr', psnr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_with_decision_based_mim(model, decision_module, train_loader, optimizer, rl_optimizer, device, args):
    """Complete training procedure for the decision-based MIM approach"""
    
    # Setup tensorboard logging
    log_writer = None
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = misc.TensorboardLogger(log_dir=args.log_dir)
    
    # Create loss scaler for mixed precision training
    loss_scaler = misc.NativeScalerWithGradNormCount() if args.use_amp else None
    
    # Training loop with three phases as described in the paper:
    # 1. Joint training of MAE and decision module
    # 2. Freeze decision module and continue MAE training
    # 3. Finetune for downstream task
    
    print("Starting joint training of MAE and decision module")
    for epoch in range(args.start_epoch, args.joint_training_epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch_with_rl(
            model, decision_module, train_loader, optimizer, rl_optimizer,
            device, epoch, loss_scaler, log_writer=log_writer, args=args
        )
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.joint_training_epochs:
            misc.save_model(
                args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, decision_module=decision_module, 
                rl_optimizer=rl_optimizer
            )
    
    print("Freezing decision module and continuing MAE training")
    # Freeze decision module
    for param in decision_module.parameters():
        param.requires_grad = False
    
    # Continue training MAE with fixed masking policy
    for epoch in range(args.joint_training_epochs, args.total_epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch_with_fixed_policy(
            model, decision_module, train_loader, optimizer,
            device, epoch, loss_scaler, log_writer=log_writer, args=args
        )
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.total_epochs:
            misc.save_model(
                args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )
    
    # Final saved model is ready for downstream finetuning


def train_one_epoch_with_fixed_policy(model, decision_module, data_loader, optimizer, 
                                    device, epoch, loss_scaler, log_writer=None, args=None):
    """Training with fixed policy after the decision module is frozen"""
    model.train(True)
    decision_module.eval()  # Decision module is in evaluation mode
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mask_ratio', misc.SmoothedValue(window_size=20, fmt='{value:.3f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    
    optimizer.zero_grad()
    
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Adjust learning rate
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # Process samples
        if not args.if_hog:
            augsamples, gtsamples = samples
            augsamples = augsamples.to(device, non_blocking=True)
            gtsamples = gtsamples.to(device, non_blocking=True)
            hogsamples = None
        else:
            augsamples, gtsamples, hogsamples = samples
            augsamples = augsamples.to(device, non_blocking=True)
            gtsamples = gtsamples.to(device, non_blocking=True)
            hogsamples = hogsamples.to(device, non_blocking=True)
        
        # Get patch features for decision making
        with torch.no_grad():
            patch_features = model.module.get_patch_features(augsamples)
            mask_decisions, _, _, _ = decision_module(patch_features)
            mask_ratio = mask_decisions.float().mean().item()
            metric_logger.update(mask_ratio=mask_ratio)
        
        # Forward pass with fixed masking policy
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            if not args.if_hog:
                loss = model(augsamples, gtsamples, mask_decisions)
            else:
                # Multi-task reconstruction with both pixel and HOG features
                mse_loss = model(augsamples, gtsamples, mask_decisions)
                hog_loss = model(augsamples, hogsamples, mask_decisions, target_type='hog')
                loss = args.mse_weight * mse_loss + args.hog_weight * hog_loss
        
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            torch.cuda.empty_cache()
            continue
        
        # Update model
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                   update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # Add gradient clipping
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        # Log metrics
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('mask_ratio', mask_ratio, epoch_1000x)
        
        # Visualize reconstructions
        if data_iter_step == 0 and args.model != 'mala' and args.model != 'superhuman' and args.model != 'unetr' and args.model != 'segmamba':
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            psnr = model.module.recons_visualize(augsamples, gtsamples, epoch_1000x, save_dir=args.visual_dir, 
                                               mask_decisions=mask_decisions)
            if log_writer is not None:
                log_writer.add_scalar('train_psnr', psnr, epoch_1000x)
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}