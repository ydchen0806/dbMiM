import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import tempfile
import os
from torch import optim
import torch.nn as nn
import numpy as np
import torchvision
import torch
import time
import sys
sys.path.append('/data/ydchen/VLP/imgSSL')
from dataloader.data_provider_pretraining import *
import yaml
import time
from attrdict import AttrDict
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse
# tensorboard
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from vit_3d import ViT
from mae import MAE
# logging
import logging
from visual2d import visual_2d
import torch.cuda.amp as amp
# import wandb

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, rank=rank, world_size=num_gpus, **kwargs)

def load_dataset(cfg):
    print('Caching datasets ... ', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider



def ddp_main():
    # accelerate = Accelerator()
    device = torch.device('cuda')
    scaler = amp.GradScaler()
    init_dist()
    torch.cuda.empty_cache()
    with open("/data/ydchen/VLP/bigmodel/IJCAI23/MAE/config/pretraining_all.yaml", "r") as f:
        cfg = AttrDict(yaml.safe_load(f))
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    # loading data path

    # define image-text dataset
    train_dataset = Train(cfg)
    # building model part
    # --------------------
    if cfg.trainer.backbone == 'Vit3d':
        model = ViT(
                        image_size = 320,          # image size
                        frames = 32,               # number of frames
                        image_patch_size = 16,     # image patch size
                        frame_patch_size = 4,      # frame patch size
                        channels=1,
                        num_classes = 1000,
                        dim = 768 * 4,
                        depth = 12 * 2,
                        heads = 12 * 2,
                        mlp_dim = 5120,
                        dropout = 0.1,
                        emb_dropout = 0.1
                    )
    else:
        print('wrong backbone')
        raise NotImplementedError
    learner = MAE(
    encoder = model,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6,       # anywhere from 1 to 8
    hog = False,
    make_decision = False
    )
    current_rank = dist.get_rank()
    if current_rank == 0:
        params = sum(p.numel() for p in learner.parameters() if p.requires_grad)
        print(f'Params: {params / 1000000}M')
    learner = torch.nn.SyncBatchNorm.convert_sync_batchnorm(learner)
    learner = learner.to(device)
    learner = DDP(learner, device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
    # learner = accelerate.prepare(learner)
    opt = torch.optim.AdamW(
        learner.parameters(),
        **cfg['optimizer']['params'],
        betas=(0.9, 0.999)
    )
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    model_dir = os.path.join(os.path.dirname(cfg.trainer.MAE_dir), now + '_' + os.path.basename(cfg.trainer.MAE_dir))
    
    os.makedirs(model_dir, exist_ok=True)  
    
    log_dir = os.path.join(model_dir, 'Logs')
    os.makedirs(log_dir, exist_ok=True) 
    if current_rank == 0:
        log_save = os.path.join(log_dir, cfg.trainer.log_name_MAE)
        with open(log_save, 'w') as file:
            file.write(str(cfg))
            file.write('\n')
            file.write(f'Params: {params / 1000000}M')
            file.write('\n')


    # os.makedirs(cfg.trainer.log_dir, exist_ok=True)  
    visual_dir = os.path.join(model_dir,'Visuals')
    os.makedirs(visual_dir, exist_ok=True)  
   
    if current_rank == 0:
        logging.basicConfig(filename=log_save, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info("Start training")
    train_provider = load_dataset(cfg)
    # --------------------
    for iters in tqdm(range(cfg.trainer.max_iterations)):
        # images = sample_unlabelled_images().cuda()
        img1 = train_provider.next()
        # print(f'gt shape is {gt.shape}, img1 shape is {img1.shape}, img2 shape is {img2.shape}')
        with amp.autocast():
            loss = learner(img1)
            # print(f'img1 shape is {img1.shape}')
            # print(f'img max is {img1.max()}, img min is {img1.min()}')
            # print(loss)
        if current_rank == 0:
            logger.info(f"loss: {loss}, iters: {iters}")
            logger.info(f'img1 max is {img1.max()}, img1 min is {img1.min()}')
        # if iters % 20 == 0 and current_rank == 0:
        #     logger.info(f"loss: {loss}, iters: {iters}")
        opt.zero_grad()
        # scaler is used to scale the gradients before applying
        # optimizer.step(). It is used in conjunction with autocast.
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if (iters % cfg.trainer.save_iters == 0 or iters == 0) and current_rank == 0:
            if cfg.trainer.backbone == 'resnet50':
                torch.save(model.state_dict(), os.path.join(model_dir, f'resnet50_{iters}.pth'))
            elif cfg.trainer.backbone == 'SWINUNETR':
                torch.save(model.state_dict(), os.path.join(model_dir, f'swin_{iters}.pth'))
                print('save model, iters is ',iters)
            elif cfg.trainer.backbone == 'Vit3d':
                torch.save(model.state_dict(), os.path.join(model_dir, f'vit3d_{iters}.pth'))
                # torch.save(learner.state_dict(), os.path.join(model_dir, f'mae_learner_{iters}.pth'))
                print('save model, iters is ',iters)
            else:
                print('wrong backbone')
                raise NotImplementedError
        if (iters % cfg.trainer.visual_iters == 0 or iters == 0) and current_rank == 0:
            # save_name = os.path.join(visual_dir, f'{iters}.png')
            visual_2d(learner, img1, visual_dir, iters)
        torch.cuda.empty_cache()
    # save logger
    logger.info("Training finished")


if __name__ == '__main__':
    ddp_main()

    

    
    