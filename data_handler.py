# Imports
from os.path import exists
from pathlib import Path

import math

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn # Includes all modules, nn.Linear, nn.Conv2d, BatchNorm etc
import torch.optim as optim # Is used for otimization algorithms such as Adam, SGD ...
from torch.utils.data import DataLoader # Helps with managing datasets in mini batches
from torch.utils.data import Dataset

import torchvision
import torchvision.datasets as datasets # Has standard datasets
import torchvision.transforms as transforms # Transformations to be used on images
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.models import vgg19
from itertools import chain
from PIL import Image


class DataHandler(Dataset):
    
    def __init__(self, data_path, img_size=299):
        data_path = Path(data_path)
        
        if not (data_path.exists() and data_path.is_dir()):
            raise ValueError(f"Data root '{data_path}' is invalid")
            
        self.data_path = data_path
        
        # Set transforms
        self.transform = T.Compose([T.Resize((299, 299)), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Collect samples
        self.samples = self.collect_samples()
            
    def __getitem__(self, index):
        # Access the stored path and label for the correct index
        img_path = self.samples[index]
        
        # Convert to image data with RGB (3 channels)
        img = Image.open(img_path).convert('RGB')
        high_res_img = self.transform_both(high_res_img)
        low_res_img = high_res_img.copy()
          
        # Perform transforms, if any.
        high_res_img = self.transform_high(high_res_img)
        low_res_img = self.transform_low(low_res_img)

        if torch.any(torch.isnan(high_res_img)) or torch.any(torch.isnan(low_res_img)):
            raise Exception("Image tensor contains nan values")

        return high_res_img, low_res_img
    
    def __len__(self):
        return len(self.samples)
        
    
    def collect_samples(self):
        if not self.data_path.exists():
            raise ValueError(f"Data root '{self.data_path}' must contain sub dir '{self.data_path.name}'")
        
        # Finds all pathnames matching a specified pattern
        file_extensions = ['jpg', 'jpeg', 'png']
        img_paths = []
        for file_extension in file_extensions:
            img_paths += list(self.data_path.rglob("*." + file_extension))
            
        return img_paths
     
    
    def get_sample_by_name(self, name, img_size=24):
        try:
            img_path = next(path for path in self.samples if path.stem == name)
        except StopIteration:
            print("No image with specified name found. Returning a random image")
            img_path = self.samples[1]
        img = Image.open(img_path).convert('RGB')
        
        return F.to_tensor(F.resize(img ,size = [img_size, img_size], interpolation = Image.BICUBIC))