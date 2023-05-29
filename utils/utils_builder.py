from cgi import test
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.models import resnet as torch_resnet
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from utils_resnet3d import *
import os

# raw resnet with cxrbert-genereal
import torch
import torch.nn as nn


class ResNet_CXRBert(torch.nn.Module):
    def __init__(self):
        super(ResNet_CXRBert, self).__init__()
        resnet = resnet50()

        self.encoder = resnet
        self.encoder.fc = nn.Identity()

        self.proj_v = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False))

        self.proj_t = nn.Sequential(
            nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512, affine=False))

        # url = 'microsoft/BiomedVLP-CXR-BERT-specialized'
        url = 'dmis-lab/biobert-v1.1'
        self.lm_model = AutoModel.from_pretrained(
            url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True, revision='main')

    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=128,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output

    def forward(self, img, input_ids, attention_mask):
      
        img_emb = self.encoder(img)
        # reshape to (b, 2048)
        img_emb = img_emb.view(img_emb.shape[0], img_emb.shape[1])
        # print(img_emb.shape)
        # pooler_output: [b, 1, 768]
        text_emb = self.lm_model(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state

        # project to 512 dim
        proj_img_emb = self.proj_v(img_emb)
        proj_text_emb = self.proj_t(text_emb[:, 0].contiguous())

        return {'img_emb': img_emb,
                'proj_img_emb': proj_img_emb,
                'proj_text_emb': proj_text_emb}


# simple projection head
class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = ResNet_CXRBert()
    print(model)
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    torch.cuda.empty_cache()