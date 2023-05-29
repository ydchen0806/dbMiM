# package import
import os
from typing import Type
import torch
import torch.nn.functional as F
import torchvision
import pandas as pd
from torch.utils.data.dataloader import DataLoader
# import wandb
import sys
sys.path.append('/braindat/lab/chenyd/code/Neurips23_caption/predict_then_optimize/pretrain/utils')
import utils_builder
# from sklearn.metrics import roc_auc_score
import math
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import time
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import re 
from PIL import Image
import random
from transformers import AutoProcessor, BlipForConditionalGeneration

# image-text embedding diagnosis style trainer Class (with language model)
def text_filter(text):
		pattern = re.compile(r'^figure\s\d+\s:\s|^fig\.\s\d+\.\s|^figure\s\d+\.\s|^fig\.\s\d+\s|figure\s\d+\s')
		filtered_text = re.sub(pattern, '', text)
		return filtered_text

def generate_caption(tempimg, base_name, processor, text_generator):
    # print(tempimg)
    # print(tempimg.shape)
    tempimg = tempimg.detach().cpu().numpy()
    if (len(tempimg.shape) == 3):
        x, _, _= tempimg.shape
        z_select = random.randint(0, x-1)
        imgs1_png = Image.fromarray(tempimg[z_select, :, :] * 255)
        imgs1_inputs = processor(imgs1_png, return_tensors="pt")
        pixel_values = imgs1_inputs['pixel_values'].cuda()
        generated_ids = text_generator.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_caption = text_filter(generated_caption)
        base_name = base_name[0].split('.')[0]
        generated_caption = base_name + ', ' + generated_caption
        
        return generated_caption

    elif (len(tempimg.shape) == 4):
        b, x, _, _= tempimg.shape
        text_list = []
        for i in range(b):
            z_select = random.randint(0, x-1)
            imgs1_png = Image.fromarray(tempimg[i, z_select, :, :] * 255)
            imgs1_inputs = processor(imgs1_png, return_tensors="pt")
            pixel_values = imgs1_inputs['pixel_values'].cuda()
            generated_ids = text_generator.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            generated_caption = text_filter(generated_caption)
            base_name = base_name[i].split('.')[0]
            generated_caption = base_name + ', ' + generated_caption
            text_list.append(generated_caption)
        return text_list
        # print(text_filter(generated_caption))
        # print(basename)
        # generated_caption = basename + ',' + text_filter(generated_caption)
  

class Provider(object):
    def __init__(self, dataset, batch_size, num_workers):
        self.data = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers 
        # self.is_cuda = is_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1
    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=0,
                                            shuffle=False, drop_last=False, pin_memory=True, sampler = DistributedSampler(self.data)))
		
	
    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            # if self.is_cuda:
            #     batch = batch.cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            # if self.is_cuda:
            #     batch = batch.cuda()
            return batch

class trainer_wBert:
    def __init__(self, model,
                 optimizer, device, model_name, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.loss_type = args['loss']
        self.train_batch_size = args['batch_size']
        self.test_batch_size = args['test_batch_size']
        self.max_epochs = args['max_epochs']
        self.lr_max = args['lr']
        self.max_iterations = args['max_iterations']
        self.num_workers = args['num_workers']
        self.checkpoint_interval = args['checkpoint_interval']
        self.smooth = args['smooth']
        self.prior_ratio = args['ratio']
        self.processor = AutoProcessor.from_pretrained('/braindat/lab/chenyd/MODEL/Neurips23/model/processor_0')
        self.text_generator = BlipForConditionalGeneration.from_pretrained('/braindat/lab/chenyd/MODEL/Neurips23/model/model_12').cuda()
    def covar_loss(self, img_embed, text_embed):
        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        logits = torch.mm(img_embed.T, text_embed).to(self.device)

        logits.div_(self.train_batch_size)
        on_diag = torch.diagonal(logits).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(logits).pow_(2).sum()
        loss = on_diag + 0.0051*off_diag
        return loss/2

    def reg_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        loss += 2 - 2 * (y * x).sum(dim=-1)
        return loss.mean()

    def entropy_loss(self, x, y):
        x = F.log_softmax(x, dim=-1)
        y = F.softmax(y, dim=-1)
        metric = nn.KLDivLoss(reduction="batchmean")
        loss = metric(x, y)
        return loss.mean()

    def clip_loss(self, x, y, prior=None):
        smooth = self.smooth

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        sim = torch.einsum('i d, j d -> i j', x, y) * 1 / 0.07

        labels = torch.arange(x.shape[0]).type_as(sim).long().to(self.device)

        loss_t = F.cross_entropy(sim, labels)
        loss_i = F.cross_entropy(sim.T, labels)
        # print(sim.shape,labels.shape)
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            sim, labels, top_k=(1,))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            sim.T, labels, top_k=(1,))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return (loss_t + loss_i), acc1, acc5
        # return loss

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    # traing process
    def train_w_TextEmb(self, train_dataset):
        mp.set_start_method('spawn', force=True)
        # train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size,
        #                           num_workers=self.num_workers,
        #                           drop_last=True, shuffle=False,
        #                           sampler=DistributedSampler(train_dataset))
        print('Start training...')
        train_provider = Provider(train_dataset, self.train_batch_size, self.num_workers)

        model_checkpoints_folder = os.path.join('/braindat/lab/chenyd/MODEL/Neurips_pretrain/','test')
        if not os.path.exists(model_checkpoints_folder):
            print('create directory "{}" for save checkpoint!'.format(
                model_checkpoints_folder))
            print('---------------------------')
            os.makedirs(model_checkpoints_folder)
        else:
            print('directory "{}" existing for save checkpoint!'.format(
                model_checkpoints_folder))

        # automatically resume from checkpoint if it exists
        print('#########################################')
        print('Be patient..., checking checkpoint now...')
        if os.path.exists(model_checkpoints_folder + self.model_name+'_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name+'_checkpoint.pth',
                              map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('continue training successful!')
        else:
            start_epoch = 0
            print('Start training from 0 epoch')

        print('#########################################')
        print('training start!')

        # scheduler
        # 16 is the num of GPUs, set it to 1 if you use single GPU
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=int(self.max_iterations //
                    4//self.train_batch_size * 0.4),
            T_mult=1,
            eta_min=1e-8,
        )
        niter = 0

        skip_scheduler = False
        scaler = GradScaler()
        

        while niter < self.max_iterations:

            epoch_loss = 0
            epoch_loss_BT, epoch_loss_read, epoch_loss_diago = 0, 0, 0
            epoch_loss_clip, epoch_loss_clip_read, epoch_loss_clip_diag = 0, 0, 0

            data, basename = train_provider.next()
            # print(basename[0])
            # get raw text
            # you can use impression or finding or concat them
            # cat imp and find will make some sequence > 512
            imp = generate_caption(data.squeeze(), basename, self.processor, self.text_generator)
            # find = data['raw_text']['FIND']

            # get image
            img = data.to(torch.float32).to(self.device).contiguous()

            self.optimizer.zero_grad()

            # amp style (might decrease precision)
            with autocast():
                # find_tokenize_output = self.model.module._tokenize(find)
                imp_tokenize_output = self.model.module._tokenize(imp)

                # input_ids['find'] = find_tokenize_output.input_ids.to(self.device).contiguous()
                # attention_mask['find'] = find_tokenize_output.attention_mask.to(self.device).contiguous()
                input_ids = imp_tokenize_output.input_ids.to(
                    self.device).contiguous()
                attention_mask = imp_tokenize_output.attention_mask.to(
                    self.device).contiguous()

                output_dict = self.model(img, input_ids, attention_mask)
                _, proj_img_emb, proj_text_emb = output_dict['img_emb'], output_dict[
                    'proj_img_emb'], output_dict['proj_text_emb']

                if self.loss_type == 'only_clip':
                    loss_clip_diag, acc1, acc5 = self.clip_loss(
                        x=proj_img_emb, y=proj_text_emb)

                    loss = loss_clip_diag
                    # accumalate loss for logging
                    epoch_loss += loss.item()
                    epoch_loss_clip_diag += loss_clip_diag.item()
                    if self.device == 0:
                        print(
                            f'epoch {niter // 1000} iter {niter} loss is {loss.item()}, acc1 is {acc1.item()}, acc5 is {acc5.item()}')

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if not skip_scheduler:
                    scheduler.step()
            niter += 1

            if self.device == 0:
                # 16 is the num of GPUs, set it to 1 if you use single GPU
                # please parameterize it and update the code
                # epoch_iter = (len(train_dataset)//self.train_batch_size//4)
                # print(f'{niter // 1000} epoch loss is {epoch_loss/epoch_iter}!')
                # wandb.log({"train loss": epoch_loss,
                #             "train covariance loss": epoch_loss_BT,
                #             "train reading loss": epoch_loss_read,
                #             "train diagnosis loss": epoch_loss_diago,
                #             "train clip loss": epoch_loss_clip,
                #             "train clip loss read": epoch_loss_clip_read,
                #             "train clip loss diag": epoch_loss_clip_diag
                #             })

                if niter % 500 == 0:
                    torch.save(self.model.module.encoder.state_dict(),
                               model_checkpoints_folder + self.model_name+f'_{niter}_iterations_encoder.pth')

        # save final vision encoder
        torch.save(self.model.module.encoder.state_dict(),
                   model_checkpoints_folder + self.model_name+'_encoder.pth')
        # save final total model
        torch.save(self.model.module.state_dict(),
                   model_checkpoints_folder + self.model_name+'total_.pth')

    def save_checkpoints(self, epoch, PATH):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            PATH)
