import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool


def read_niigz(data_path):
    img = nib.load(data_path)
    img = img.get_fdata()
    return img


father_path = ['/braindat/lab/chenyd/DATASET/MSD/Task01_BrainTumour/imagesTs', '/braindat/lab/chenyd/DATASET/MSD/Task01_BrainTumour/imagesTr', \
    '/braindat/lab/chenyd/DATASET/MSD/Task04_Hippocampus/imagesTs', '/braindat/lab/chenyd/DATASET/MSD/Task04_Hippocampus/imagesTr', \
        '/braindat/lab/chenyd/DATASET/MSD/Task02_Heart/imagesTs', '/braindat/lab/chenyd/DATASET/MSD/Task02_Heart/imagesTr', \
    '/braindat/lab/chenyd/DATASET/MSD/Task07_Pancreas/imagesTs', '/braindat/lab/chenyd/DATASET/MSD/Task07_Pancreas/imagesTr']

def process(path):
    temp_data = read_niigz(path)
    if len(temp_data.shape) == 4:
        print(os.path.basename(path), temp_data.shape)
    
    return 0

if __name__ == '__main__':
    for path in father_path:
        data_path = glob(os.path.join(path, '*.nii.gz'))
        with Pool(16) as p:
            p.map(process, data_path)
