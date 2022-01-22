import torch
import torch.nn as nn
from collections import Counter
import numpy as np

PAD = '<pad>'
UNKNOWN = '<unk>'
SOS = '<start>'
EOS = '<end>'

class Vocabulary:
    def __init__(self, gensim_model, max_voc_size=10000, include_unknown=True, lower=False):

        self.gensim_model = gensim_model
        self.include_unknown = include_unknown
        self.dummies = [PAD, UNKNOWN, SOS, EOS] if self.include_unknown else [PAD, SOS, EOS]
                
        # String-to-integer mapping
        self.stoi = None
        # Integer-to-string mapping
        self.itos = None
        # Maximally allowed vocabulary size.
        self.max_voc_size = max_voc_size
        self.lower = lower
        self.vectors = None

        self.build()
        
            
    def build(self):
        print('building vocabulary from gensim model')
        limit = self.max_voc_size if self.max_voc_size else len(self.gensim_model.index_to_key)
        self.vectors = torch.FloatTensor(self.gensim_model.vectors[:limit])
        self.itos = self.dummies + self.gensim_model.index_to_key[:limit]
        self.stoi = {s:i for i, s in enumerate(self.itos)}
        self.lower = True


    def make_embedding_layer(self, fine_tune=False):
        embed_size = self.vectors.shape[1]
        emb_layer = nn.Embedding(len(self.itos), embed_size, sparse=False)
        emb_layer.weight.requires_grad = fine_tune

        with torch.no_grad():
            # Copy the pre-trained embedding weights into our embedding layer.
            emb_layer.weight[len(self.dummies):, :] = self.vectors
        
        return emb_layer
                
    def encode(self, seqs):
        unk = self.stoi.get(UNKNOWN)
        sos = self.stoi.get(SOS)
        eos = self.stoi.get(EOS)
        pad = self.stoi.get(PAD)
        
        if self.lower:
            seqs = [[s.lower() for s in seq] for seq in seqs ]

        encoded_captions = [[sos]+[self.stoi.get(w, unk) for w in seq]+[eos] for seq in seqs]
        caption_lengths = [len(encoded_caption) for encoded_caption in encoded_captions]
        max_length = max(caption_lengths)
        for i in range(len(encoded_captions)):
            padding = [pad for j in range(max_length-caption_lengths[i])]
            encoded_captions[i] = encoded_captions[i] + padding 

        # add dimension for simplicity later
        caption_lengths = [[caption_length] for caption_length in caption_lengths]
        return encoded_captions, caption_lengths

    def decode(self, seqs):
        unk = self.stoi.get(UNKNOWN)
        decoded_captions = [[self.itos[encoding] for encoding in seq] for seq in seqs]
        return decoded_captions
        
    def get_unknown_idx(self):
        return self.stoi[UNKNOWN]
    
    def get_pad_idx(self):
        return self.stoi[PAD]
    
    def __len__(self):
        return len(self.itos)
