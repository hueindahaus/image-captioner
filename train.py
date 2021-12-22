import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn # Includes all modules, nn.Linear, nn.Conv2d, BatchNorm etc
import torch.optim as optim # Is used for otimization algorithms such as Adam, SGD ...
from torch.utils.data import DataLoader # Helps with managing datasets in mini batches
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import torchvision
import torchvision.datasets as datasets # Has standard datasets
import torchvision.transforms as transforms # Transformations to be used on images
import torchvision.transforms as T
import torchvision.transforms.functional as F

from datetime import datetime

from utils import Stat, visualize_attention
import config


device = config.DEVICE

def train(data, encoder, decoder, vocab):

    print('Starting training..')
    torch.cuda.empty_cache()

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    num_epochs = config.NUM_EPOCHS
    batch_size = config.BATCH_SIZE
    num_workers = config.NUM_WORKERS
    encoder_lr = config.ENCODER_LR
    decoder_lr = config.DECODER_LR

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)

    train_data = data
    val_data = data.split_data(0.1)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_data.collate_batch)
    val_loader = DataLoader(val_data, batch_size, shuffle=True, num_workers=num_workers, collate_fn=val_data.collate_batch)

    criterion = nn.CrossEntropyLoss().to(device)

    print_every_nth_batch = config.PRINT_EVERY_NTH_BATCH

    training_losses = Stat()

    now = datetime.now() # current date and time
    writer = SummaryWriter(f'runs/logs/{now.strftime("%Y-%m-%d-h%Hm%M")}')
    step = 0

    for epoch in range(num_epochs):
        
        encoder.train()
        decoder.train()

        training_losses.clear()

        for batch_index, (imgs, captions, caption_lengths) in enumerate(train_loader):

            #print(imgs.shape)
            #print(captions.shape)
            #print(caption_lengths.shape)
            #print('-----')

            imgs = imgs.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)
            
            features = encoder(imgs)
            predictions, captions, output_lengths, alphas, sorted_indices = decoder(features, captions, caption_lengths)

            predictions = pack_padded_sequence(predictions, output_lengths, batch_first = True).data
            targets = pack_padded_sequence(captions[:, 1:], output_lengths, batch_first = True).data
        
            loss = criterion(predictions, targets)
            # doubly stochastic attention regularization
            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
            training_losses(loss.item(), sum(output_lengths))

            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()
            encoder_optimizer.step()

            if batch_index > 0 and batch_index % print_every_nth_batch == 0:
                print(f'[Epoch: {epoch + 1}, Batch: {batch_index}/{len(train_loader)}] Training loss: {training_losses.average()}')

                # Inference example
                with torch.no_grad():

                    # Set eval mode to suppress dropout etc.
                    decoder.eval()
                    encoder.eval()

                    # Fetch random image from batch
                    img = imgs[random.randint(0,len(imgs)-1)]

                    # Forward image through encoder
                    features = encoder.forward(torch.unsqueeze(img, dim=0).to(device))

                    # Generate caption with decoder
                    seq_encoded, alphas = decoder.predict(features)

                    # decode generated sequence from integers to actual words
                    seq_decoded = vocab.decode(seq_encoded.tolist())

                    # Handle writer
                    writer.add_scalar('Training loss', training_losses.average(), global_step=step)
                    writer.add_figure('Inference example', visualize_attention(img, alphas, seq_decoded), global_step=step)

                    # Set train mode to enable dropout etc.
                    decoder.train()
                    encoder.train()

                step += 1

            



    
        

