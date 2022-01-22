import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn # Includes all modules, nn.Linear, nn.Conv2d, BatchNorm etc
import torch.optim as optim # Is used for otimization algorithms such as Adam, SGD ...
from torch.nn.utils.rnn import pack_padded_sequence

from torch.utils.data import DataLoader # Helps with managing datasets in mini batches
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import torchvision
import torchvision.datasets as datasets # Has standard datasets
import torchvision.transforms as transforms # Transformations to be used on images
import torchvision.transforms as T
import torchvision.transforms.functional as F

from nltk.translate.bleu_score import corpus_bleu   # Used to compute bleu score

from datetime import datetime

from utils import Stat, visualize_attention
from simloss import SimLoss


def train(model_handler):

    print('Starting training..')
    torch.cuda.empty_cache()

    # Load models and all other hyper/training parameters
    data, vocab, encoder, decoder, encoder_optimizer, decoder_optimizer, config = model_handler.load()

    num_epochs = config['NUM_EPOCHS']
    batch_size = config['BATCH_SIZE']
    num_workers = config['NUM_WORKERS']
    start_epoch = config['START_EPOCH']
    device = config['DEVICE']

    # lr decay on decoder
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', threshold = 1e-4, patience=20, factor=0.8, verbose=True, cooldown = 10)

    data_loader = DataLoader(data, batch_size, shuffle=True, num_workers=num_workers, collate_fn=data.collate_batch)

    criterion = nn.CrossEntropyLoss().to(device)
    #criterion = SimLoss(vocab)

    print_every_nth_batch = config['PRINT_EVERY_NTH_BATCH']

    training_losses = Stat()

    now = datetime.now() # current date and time
    writer = SummaryWriter(f'runs/logs/{now.strftime("%Y-%m-%d-h%Hm%M")}')
    step = 0

    scheduled_sampling_p = 0    # Should stay between 0 and 1 and increases epochwise (curriculum training)

    for epoch in range(start_epoch, num_epochs):

        # Checkpoint
        model_handler.checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, config_dict = { **config, 'START_EPOCH': epoch  })
        
        encoder.train()
        decoder.train()

        scheduled_sampling_p = (epoch + 1)/num_epochs

        for batch_index, (imgs, captions, caption_lengths) in enumerate(data_loader):

            # Make sure that tensors are set to correct device
            imgs = imgs.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)
            
            # Forward images through encoder and obtain features
            features = encoder(imgs)

            # Forward features alogn with captions etc. through decoder to obtain predictions
            predictions, captions, output_lengths, alphas, sorted_indices = decoder(features, captions, caption_lengths, scheduled_sampling_p=scheduled_sampling_p)

            # Pack sequences (removes padding) and squeezes the tensors so that all predictions lie in one dimension
            predictions = pack_padded_sequence(predictions, output_lengths, batch_first = True).data
            targets = pack_padded_sequence(captions[:, 1:], output_lengths, batch_first = True).data

            # Calculate loss
            loss = criterion(predictions, targets)

            # Add doubly stochastic attention regularization to the loss to encourage pixel alpha for each pixel across all timesteps to sum to 1
            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Add loss to stats
            training_losses(loss.item(), sum(output_lengths))

            # Perform backprop
            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()
            encoder_optimizer.step()

            if batch_index > 0 and batch_index % print_every_nth_batch == 0:
                decoder_scheduler.step(training_losses.average())                

                # eval 
                with torch.no_grad():

                    # Set eval mode to suppress dropout etc.
                    decoder.eval()
                    encoder.eval()
                    data.eval()

                    # Fetch random image from batch
                    img = imgs[random.randint(0,len(imgs)-1)]

                    # Forward image through encoder
                    features = encoder.forward(torch.unsqueeze(img, dim=0).to(device))

                    # Generate caption with decoder
                    preds, alphas = decoder.beam(features)

                    # decode generated sequence from integers to actual words
                    preds_decoded = vocab.decode(preds)

                    # Handle writer
                    writer.add_scalar('Training loss', training_losses.average(), global_step=step)
                    writer.add_figure('Inference example', visualize_attention(img, alphas[0], preds_decoded[0]), global_step=step)

                    #validation loop
                    hypotheses = []
                    references = []
                    for i, (imgs, captions) in enumerate(data_loader):
                        imgs = imgs.to(device)
                        features = encoder(imgs)
                        preds, _ = decoder.beam(features)
                        # Get rid of <start> and <eos> (In the case of max length sentences where <eos> is missing, we simply shave off last token anyway)
                        generated_captions = vocab.decode([pred[1:-1] for pred in preds])
                        references.extend(captions)
                        hypotheses.extend(generated_captions)
                        
                    bleu = corpus_bleu(references, hypotheses)
                    writer.add_scalar('Bleu', bleu, global_step=step)

                    # Set train mode to enable dropout etc.
                    decoder.train()
                    encoder.train()
                    data.train()

                    print(f'[Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_index}/{len(data_loader)}] Training loss: {training_losses.average()} Bleu: {bleu}')

                
                training_losses.clear()

                step += 1

            



    
        

