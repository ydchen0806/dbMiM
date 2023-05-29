import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from vit_3d import ViT
from mae import MAE
import os
from glob import glob
import sys
import yaml
import time
import cv2
import h5py
import random
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import OrderedDict
import multiprocessing as mp
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
import multiprocessing as mp
from einops import rearrange
from monai import transforms
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
#导入ddp

import torch.distributed
import torch.multiprocessing as mp
import argparse
from attrdict import AttrDict

import waterz
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")

model = ViT(
                        image_size = 160,          # image size
                        frames = 32,               # number of frames
                        image_patch_size = 16,     # image patch size
                        frame_patch_size = 4,      # frame patch size
                        channels=1,
                        num_classes = 1000,
                        dim = 3072,
                        depth = 36,
                        heads = 12,
                        mlp_dim = 4096,
                        dropout = 0.1,
                        emb_dropout = 0.1
                    ).cuda()

mae = MAE(
    encoder = model,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 8,       # anywhere from 1 to 8
    hog = False
).cuda()
print('参数量M: ', sum(p.numel() for p in mae.parameters()) / 1000000.0)
video = torch.randn(2, 1, 32, 160, 160).cuda() # (batch, channels, frames, height, width)

preds = model(video) # (4, 1000)
print(preds.shape)


loss = mae(video)
# loss.backward()
print(loss)