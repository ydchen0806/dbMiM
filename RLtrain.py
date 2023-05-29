# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import random
import _init_paths
import models
from config import config
from config import update_config
from core.function import train
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import torchvision

from autoaugment import ImageNetPolicy
from PAA import *
from A2Cmodel import ActorCritic

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        default='./experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='/output')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--is_leinao',
                        action='store_true',
                        default=False)

    parser.add_argument('--is_pretrained',
                        action='store_true',
                        default=False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ code bellowed only works when is_leinao is True 
    #& (bad code)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parser.add_argument('--AUG',
                        default='base',
                        type=str,
                        choices=['base', 'AA', 'PAA'])
    parser.add_argument('--N_GRID',
                        default=4,
                        type=int)
    parser.add_argument('--DATASET',
                        default='cub',
                        type=str)
    parser.add_argument('--IMAGE_SIZE',
                    default=224,
                    type=int)   
    parser.add_argument('--EPOCHS',
                    default=300,
                    type=int)                                 
    parser.add_argument('--BATCH_SIZE',
                    default=64,
                    type=int)      
    parser.add_argument('--RLPROB',
                    action='store_true',
                    default=False)

    parser.add_argument('--da-method', default='mixup', type=str, choices=['baseline','mixup', 'cutmix', 'gridmask','translate','flip','brightness'],
                    help='data augmentation method')    

    parser.add_argument('--prob-method', default='offline', type=str, choices=['offline','HRL','HRL-low','HRL-high','all', 'fix_prob', 'linear_increase','random','random_fix'],
                    help='prob')                                
    # FRY
    # parser.add_argument('--aug', default='base', type=str, choices=['base', 'AA', 'PAA'],
    #                     help='augmentation type')

    args = parser.parse_args()
    update_config(config, args)

    return args

def build_model(args, config):
    is_pretrained = args.is_pretrained
    if not args.is_leinao:
        if config.MODEL.NAME == 'resnet18':
            model = torchvision.models.resnet18(pretrained=is_pretrained)
        elif config.MODEL.NAME == 'resnet34':
            model = torchvision.models.resnet34(pretrained=is_pretrained)
        elif config.MODEL.NAME == 'resnet50':
            model = torchvision.models.resnet50(pretrained=is_pretrained)
        elif config.MODEL.NAME == 'resnet101':
            model = torchvision.models.resnet101(pretrained=is_pretrained)
        elif config.MODEL.NAME == 'resnet152':
            model = torchvision.models.resnet152(pretrained=is_pretrained)
        elif config.MODEL.NAME == 'vgg16':
            model = torchvision.models.vgg16(pretrained=is_pretrained)
    else:
        if config.MODEL.NAME == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False)
            if is_pretrained:
                model.load_state_dict(torch.load('/model/bitahub/ResNet/resnet18.pth'))
        elif config.MODEL.NAME == 'resnet34':
            model = torchvision.models.resnet34(pretrained=False)
            if is_pretrained:
                model.load_state_dict(torch.load('/model/bitahub/ResNet/resnet34.pth'))
        elif config.MODEL.NAME == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False)
            if is_pretrained:
                model.load_state_dict(torch.load('/model/bitahub/ResNet/resnet50.pth'))
        elif config.MODEL.NAME == 'resnet101':
            model = torchvision.models.resnet101(pretrained=False)
            if is_pretrained:
                model.load_state_dict(torch.load('/model/bitahub/ResNet/resnet101.pth'))
        elif config.MODEL.NAME == 'resnet152':
            model = torchvision.models.resnet152(pretrained=False)
            if is_pretrained:
                model.load_state_dict(torch.load('/model/bitahub/ResNet/resnet152.pth'))
        elif config.MODEL.NAME == 'vgg16':
            model = torchvision.models.vgg16(pretrained=False)
            if is_pretrained:
                model.load_state_dict(torch.load('/model/YunanZhu/torchvision_vgg_model/vgg_model/vgg16-397923af.pth'))
    # if load paa model 
    # model_state_file = "./checkpoints/resnet50pa.tar"
    # if os.path.isfile(model_state_file):
    #     checkpoint = torch.load(model_state_file)
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)
    #     print("=> loaded checkpoint (epoch)")
    #     print("load over!")

    if (config.MODEL.NAME!='vgg16'):
        num_classes = config.MODEL.NUM_CLASSES
        print("***********")
        print(num_classes)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)

    else:
        num_classes = config.MODEL.NUM_CLASSES
        num_classes = config.MODEL.NUM_CLASSES
        print("***********")
        print(num_classes)
        model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=1280, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1280, out_features=num_classes, bias=True),
        )
    return model

def build_dataloaders(args, config, gpus):

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print(config.DATASET)
    root = config.DATASET.ROOT

    print(root)
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    
    if args.AUG == 'base':
        transfrom_train = transforms.Compose([
                transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
                
                # transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
                # transforms.RandomCrop(config.MODEL.IMAGE_SIZE[0]),

                # transforms.Resize(size=(config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])),

                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    elif args.AUG == 'AA':
        transfrom_train = transforms.Compose([
                transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
                
                # transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
                # transforms.RandomCrop(config.MODEL.IMAGE_SIZE[0]),
                
                # transforms.Resize(size=(config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])),

                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize,
            ])
    elif args.AUG == 'PAA':
        if config.N_GRID == 1:
            transfrom_train = transforms.Compose([
                transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
                
                # transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
                # transforms.RandomCrop(config.MODEL.IMAGE_SIZE[0]),
                
                # transforms.Resize(size=(config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])),

                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
            ])
        else:
            transfrom_train = transforms.Compose([
                # transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
                
                transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
                transforms.RandomCrop(config.MODEL.IMAGE_SIZE[0]),
                
                # transforms.Resize(size=(config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])),

                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
            ])
    else:
        raise('invalid augmentation type !')

    transform_val = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),

            # transforms.Resize(size=(config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])),

            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.ImageFolder(traindir, transfrom_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_dataset = datasets.ImageFolder(valdir, transform_val)
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=False
        )
    return train_loader, valid_loader

def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')
    
    tb_log_dir = "/output/log/"
    final_output_dir = "/output/"
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = build_model(args, config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, last_epoch-1)

    train_loader, valid_loader = build_dataloaders(args, config, gpus)

    if args.AUG == 'PAA':
        model_a2c = ActorCritic(
            N_grid=config.N_GRID, 
            num_ops=len(ops), 
            img_size=config.MODEL.IMAGE_SIZE[0], 
            RLprob=config.RLPROB).cuda()
        optimizer_a2c = torch.optim.SGD(model_a2c.parameters(), lr=config.LR_A2C, 
                                    momentum=0.9, nesterov=True,
                                    weight_decay=1e-4)
        scheduler_a2c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a2c, last_epoch-1)

    ###mixup cutmix  config.TRAIN.END_EPOCH
    

    low_level = []

    #epoch*0.06 = 150*0.06 = 9
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.06)):  #9
    #     low_level.append(0.0)
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.06)):   #9
    #     low_level.append(0.0)
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.06)):   #9
    #     low_level.append(0.0)
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.06)):   #9
    #     low_level.append(0.0)
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.18)):   #9*3=27
    #     low_level.append(0.0) 
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.12)):  #9*2=18
    #     low_level.append(0.0)
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.06)):  #9
    #     low_level.append(0.0)
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.06)):   #9
    #     low_level.append(0.0)
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.12)):  #9*2=18
    #     low_level.append(0.0)
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.12)):   # 9*2=18
    #     low_level.append(0.0)
    # for _ in range(0,int(config.TRAIN.END_EPOCH*0.12)):   #9*2=18  15
    #     low_level.append(0.0)


    #ori
    for _ in range(0,int(config.TRAIN.END_EPOCH)):  #9
        low_level.append(0.0)


    print(len(low_level))
    print(low_level)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        if args.AUG == 'PAA':
            scheduler_a2c.step()

        # train for one epoch
        if args.AUG == 'PAA':
            train(args,config, train_loader, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict,low_level,
                model_a2c, optimizer_a2c, scheduler_a2c)
        else:
            train(args,config, train_loader, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict,low_level)
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,
                                  final_output_dir, tb_log_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        logger.info('best prec is : {:.4f}'.format(best_perf))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    logger.info('best prec is : {:.4f}'.format(best_perf))

    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
    print('best prec is : {:.4f}'.format(best_perf))


if __name__ == '__main__':
    main()
