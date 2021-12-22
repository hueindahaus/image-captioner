# Imports
from os.path import exists
from pathlib import Path
import csv

import math

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
from itertools import chain
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from vocab import Vocabulary


class DataHandler(Dataset):
    def __init__(self, img_folder_path, caption_file_path, img_size=299, gensim_model=None, skip_init=False):
        if not skip_init:
            img_folder_path = Path(img_folder_path)
            caption_file_path = Path(caption_file_path)
            
            if not (img_folder_path.exists() and img_folder_path.is_dir()):
                raise ValueError(f"Image folder path '{img_folder_path}' is invalid")
                
            if not (caption_file_path.exists() and caption_file_path.is_file()):
                raise ValueError(f"Caption file path '{caption_file_path}' is invalid")

            self.img_folder_path = img_folder_path
            self.caption_file_path = caption_file_path
            
            # Set transforms
            self.transform = T.Compose([T.Resize((299, 299)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            self.vocab = Vocabulary(gensim_model=gensim_model)

            # Collect samples
            self.samples = self.collect_samples()
            
    def __getitem__(self, index):
        # Access the stored path and label for the correct index
        sample = self.samples[index]
        
        # Convert to image data with RGB (3 channels)
        img = Image.open(sample[0]).convert('RGB')

        # Perform transforms
        img = self.transform(img)

        if torch.any(torch.isnan(img)):
            raise Exception("Image tensor contains nan values")

        # collect caption
        caption = sample[1]

        return img, caption
    
    def __len__(self):
        return len(self.samples)
        
    def collect_samples(self):
        if not self.img_folder_path.exists():
            raise ValueError(f"Data root '{self.img_folder_path}' must contain sub dir '{self.img_folder_path.name}'")

        if not self.caption_file_path.exists():
            raise ValueError(f"Caption file '{self.caption_file_path}' does not exist")
        

        # dictionary that maps an image file name to a file path
        img_map = {}

        samples = []

        # Finds all pathnames matching a specified pattern
        for img_path in tqdm(list(self.img_folder_path.rglob("*.jpg"))):
            img_map[img_path.name] = img_path

        with open(self.caption_file_path, encoding="utf-8") as csv_file:
            for row in list(csv.reader(csv_file, delimiter='|'))[1:-3]:
                img_file_name = row[0]
                img_file_path = img_map.get(img_file_name)
                if img_file_path:
                    caption = row[2]
                    if caption:
                        samples.append((img_file_path, caption.split(' ')[1:-1]))

        self.vocab.build([sample[1] for sample in samples])

        return samples

    def get_captions(self):
        return [sample[1] for sample in self.samples]

    def split_data(self, percent):
        num_migrations = math.floor(len(self) * percent)
        migration_samples = self.samples[:num_migrations]
        del self.samples[:num_migrations]
        print(f'Splitting data: {len(self) + num_migrations} -> {len(self)} + {num_migrations}')
        migration_data = DataHandler('','', skip_init=True)
        migration_data.samples = migration_samples
        migration_data.vocab = self.vocab
        return migration_data

    def collate_batch(self, batch):
        imgs = [sample[0] for sample in batch]
        captions = [sample[1] for sample in batch]
        encoded_captions, caption_lengths = self.vocab.encode(captions)
        return torch.stack(imgs), torch.tensor(encoded_captions), torch.tensor(caption_lengths)
