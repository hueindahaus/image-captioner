from os.path import exists
from os import mkdir
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

import gensim
import gensim.downloader
if 'gensim_model' not in globals():
    gensim_model = gensim.downloader.load("glove-wiki-gigaword-300") # https://github.com/RaRe-Technologies/gensim-data

from model import Encoder, Decoder
from data_handler import DataHandler

import config

device = config.DEVICE

class ModelHandler:

    def __init__(self, checkpoint_name):
        self.checkpoint_path = './checkpoints/' + checkpoint_name

    def checkpoint(self, encoder, decoder, encoder_optimizer, decoder_optimizer, config_dict={}):
        print('Setting checkpoint')
        
        # if config missing, get them from config file
        default_config = config.export_config_dict()
        for key in default_config.keys():
            if key not in config_dict:
                config_dict[key] = default_config[key]
        
        FOLDER = self.checkpoint_path   
        PATH = FOLDER + '/checkpoint.pt'
        
        try:
            # Create a new folder if it does not exist
            if not exists(FOLDER):
                mkdir(FOLDER)
        except OSError as error:
            print(error)

        torch.save({
            'ENCODER_STATE':encoder.state_dict(),
            'DECODER_STATE':decoder.state_dict(),
            'ENCODER_OPTIMIZER_STATE': encoder_optimizer.state_dict(),
            'DECODER_OPTIMIZER_STATE': decoder_optimizer.state_dict(),
            **config_dict
        }, PATH)
        if exists(PATH):
            print('Saved succesfully')
        return FOLDER   # Return path of the folder where the models were saved


    def load(self, image_folder_path = "./data/", captions_file_path = './data/flickr30k_images/results.csv'):
        FOLDER = self.checkpoint_path
        PATH = FOLDER + '/checkpoint.pt'

        # Load data and vocab
        data = DataHandler(image_folder_path, captions_file_path, gensim_model=gensim_model)
        vocab = data.vocab

        if exists(PATH):
            print('[Factory] Loading existing model from ' + PATH)
            checkpoint = torch.load(PATH)

            # Load encoder
            encoder = Encoder(dropout = checkpoint['ENCODER_DROPOUT']).to(device)
            encoder.load_state_dict(checkpoint['ENCODER_STATE'], strict=False)

            # Load decoder
            decoder = Decoder(vocab, embed_size = checkpoint['EMBED_SIZE'], hidden_size = checkpoint['HIDDEN_SIZE'], attention_size=checkpoint['ATTENTION_SIZE'], num_layers = checkpoint['NUM_HIDDEN_LAYERS'], dropout = checkpoint['DECODER_DROPOUT'], attention_dropout=checkpoint['ATTENTION_DROPOUT'], fine_tune_embeddings = checkpoint['FINE_TUNE_EMBEDDINGS']).to(device)
            decoder.load_state_dict(checkpoint['DECODER_STATE'])

            # Load optimizers
            encoder_optimizer = optim.Adam(encoder.parameters())
            decoder_optimizer = optim.Adam(decoder.parameters())
            encoder_optimizer.load_state_dict(checkpoint['ENCODER_OPTIMIZER_STATE'])
            decoder_optimizer.load_state_dict(checkpoint['DECODER_OPTIMIZER_STATE'])

            # populate config dict
            default_config = config.export_config_dict()
            config_dict = {}
            for key in default_config.keys():
                if not (key == 'ENCODER_LR' or key == 'DECODER_LR'):
                    if key in checkpoint:
                        config_dict[key] = checkpoint[key]
                    else:
                        config_dict[key] = default_config[key]
        else:
            print('[Factory] Creating new model')
            encoder = Encoder(dropout = config.ENCODER_DROPOUT).to(device)
            decoder = Decoder(vocab, embed_size = config.EMBED_SIZE, hidden_size = config.HIDDEN_SIZE, attention_size=config.ATTENTION_SIZE, num_layers = config.NUM_HIDDEN_LAYERS, dropout = config.DECODER_DROPOUT, attention_dropout=config.ATTENTION_DROPOUT, fine_tune_embeddings = config.FINE_TUNE_EMBEDDINGS).to(device)
            encoder_optimizer = optim.Adam(encoder.parameters(), lr = config.ENCODER_LR)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr = config.DECODER_LR)
            config_dict = config.export_config_dict()

        print_config_dict(config_dict)
        
        return data, vocab, encoder, decoder, encoder_optimizer, decoder_optimizer, config_dict


def print_config_dict(config_dict):
    print('CONFIG:')
    print('-----------------------------------------------------------')
    for key in config_dict.keys():
        print(f'{key}: {str(config_dict[key])}')
    print('-----------------------------------------------------------')
