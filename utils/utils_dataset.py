import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from torchvision.transforms import transforms
from PIL import Image


class IaT_embed_dataset(Dataset):
    def __init__(self, image_data, database='en', transform=None, **args):
        self.img_data = image_data

        self.text_csv = args['text']
        self.mode = args['train_test']
        self.database = database
        self.transform = transform

    def __len__(self):
        return (self.img_data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        image = self.img_data[idx]
        image = Image.fromarray(image).convert("RGB")

        # get raw text
        if self.database == 'en':
            findings = self.text_csv['findings'].iloc[idx]
            impression = self.text_csv['impression'].iloc[idx]
            if findings == 'dumb' or type(findings) == float:
                pass
            else:
                impression += findings
            text = impression
        elif self.database == 'sp':
            text = self.text_csv['Report'].iloc[idx]

        sample = {'image1': image, 'raw_text': text}

        if self.transform:
            # for 2 branch contrastive vision model (not useful for CLIP)
            if self.mode == 'train':
                sample['image1'] = self.transform[0](sample['image1'])
                # sample['image2'] = self.transform[1](sample['image'])
            elif self.mode == 'test':
                sample['val_image'] = self.transform(sample['image'])

        return sample


class I_T_emb_dataset:

    def __init__(self, image_path, csv_path, database='en', **args):
        self.image_path = image_path
        self.csv_path = csv_path
        self.database = database

    def get_dataset(self, train_test, T=None):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        if train_test == 'train':
            Transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(224),
                normalize
            ])
            print('Apply Train-stage Transform!')

            Transforms_super = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomAutocontrast(p=0.5),
                normalize
            ])

            Trans = [Transforms, Transforms_super]
        else:
            Transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                normalize
            ])
            print('Apply Test-stage Transform!')

        en_img = np.load(
            self.image_path['en_img_path'], allow_pickle=True, mmap_mode='r')
        en_csv = pd.read_csv(
            self.csv_path['en_text_csv_path'], low_memory=False)

        sp_img = np.load(
            self.image_path['sp_img_path'], allow_pickle=True, mmap_mode='r')
        sp_csv = pd.read_csv(
            self.csv_path['sp_text_csv_path'], low_memory=False)

        en_args = {'train_test': train_test,
                   'text': en_csv}

        sp_args = {'train_test': train_test,
                   'text': sp_csv}

        en_dataset = IaT_embed_dataset(image_data=en_img,
                                       database='en',
                                       transform=Trans,
                                       **en_args)

        sp_dataset = IaT_embed_dataset(image_data=sp_img,
                                       database='sp',
                                       transform=Trans,
                                       **sp_args)

        dataset = ConcatDataset([en_dataset, sp_dataset])
        return dataset
