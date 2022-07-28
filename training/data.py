import glob
import logging
import os
import cv2
import random
import torch
import numpy as np
import torchvision
import albumentations as A
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import Dataset
import pandas as pd  




# prepare data
def get_data(dataset_path):
    train_data = DataloaderDogCat(dataset_path+'train',kind = 'train')
    val_data = DataloaderDogCat(dataset_path+'val',kind = 'val')
    return train_data,val_data

def save_paths(names_list,kind):
    dict = {'image_name':names_list}
    df = pd.DataFrame(dict)     
    # saving the dataframe 
    df.to_csv('./csv/'+kind+'.csv') 


# define our data augmentation transform
def get_transform(out_size,kind='train'):
    if kind == 'train':
        transform = A.Compose([
        A.HorizontalFlip(),
        A.CLAHE(),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
        A.Normalize(),
        A.ToFloat()
        ])
    elif kind == 'test' or kind == 'val':
        transform =  A.Compose([
        A.Normalize(),
        A.ToFloat()
        ])
    return transform

class DataloaderDogCat(Dataset):
    def __init__(self, indir,kind = 'train'):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        random.shuffle(self.in_files)
        # save our train,val,test image names in CSVs (this way we keep track of our datasets)
        save_paths(self.in_files,kind) 
        self.transform = get_transform(kind)
        self.iter_i = 0
        self.kind = kind

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.resize(img, (128,128), interpolation= cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        filename = path.split('/')[-1]
        # we assume here that the dog images are labeled "1" and cats "0"
        label = filename.split('.')[0]=='dog' 
        label = torch.tensor(label, dtype = torch.float32)
        return img,label

## debug
if __name__ =='__main__': 
    data = DataloaderDogCat(indir='./train/',kind = 'train')

