import torch
import torch.nn as nn
from collections import Counter

PAD = '<pad>'
UNKNOWN = '<unk>'
SOS = '<start>'
EOS = '<end>'

class Vocabulary:
    """Manages the numerical encoding of the vocabulary."""
    def __init__(self, gensim_model, max_voc_size=25000, include_unknown=True, lower=True):

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
        
            
    def build(self, seqs):
        """Builds the vocabulary."""

        if not self.gensim_model:
            if self.lower:
                seqs = [ [s.lower() for s in seq] for seq in seqs ]
            
            # Sort all words by frequency
            word_freqs = Counter(w for seq in seqs for w in seq)
            word_freqs = sorted(((f, w) for w, f in word_freqs.items()), reverse=True)

            # Build the integer-to-string mapping. The vocabulary starts with the two dummy symbols,
            # and then all words, sorted by frequency. Optionally, limit the vocabulary size.
            
            if self.max_voc_size:
                self.itos = self.dummies + [ w for _, w in word_freqs[:self.max_voc_size-len(dummies)] ]
            else:
                self.itos = self.dummies + [ w for _, w in word_freqs ]

            # Build the string-to-integer map by just inverting the aforementioned map.
            self.stoi = { w: i for i, w in enumerate(self.itos) }
        else:
            limit = self.max_voc_size if self.max_voc_size else len(self.gensim_model.index_to_key)
            self.vectors = torch.FloatTensor(self.gensim_model.vectors[:limit])
            self.itos = self.dummies + self.gensim_model.index_to_key[:limit]
            self.stoi = {s:i for i, s in enumerate(self.itos)}
            for w in self.itos:
                w0 = w[0]
                if w0.isupper():
                    self.lower = False
                    break


    def make_embedding_layer(self, embed_size, fine_tune=True):
        if self.vectors is not None:
            embed_size = self.vectors.shape[1]
            emb_layer = nn.Embedding(len(self.itos), embed_size, sparse=False)

            with torch.no_grad():
                # Copy the pre-trained embedding weights into our embedding layer.
                emb_layer.weight[len(self.dummies):, :] = self.vectors
        else:
            emb_layer = nn.Embedding(len(self.itos), embed_size, sparse=False)
            if not fine_tune:
                # If we don't fine-tune, create a tensor where we don't compute the gradients.
                emb_layer.weight = nn.Parameter(emb_layer.weight, requires_grad=False)
        
        return emb_layer
                
    def encode(self, seqs):
        """Encodes a set of documents."""
        unk = self.stoi.get(UNKNOWN)
        sos = self.stoi.get(SOS)
        eos = self.stoi.get(EOS)
        pad = self.stoi.get(PAD)
        
        if self.lower:
            seqs = [ [s.lower() for s in seq] for seq in seqs ]

        encoded_captions = [[sos]+[self.stoi.get(w, unk) for w in seq] + [eos] for seq in seqs]
        caption_lengths = [len(encoded_caption) for encoded_caption in encoded_captions]
        max_length = max(caption_lengths)
        for i in range(len(encoded_captions)):
            padding = [pad for j in range(max_length-caption_lengths[i])]
            encoded_captions[i] = encoded_captions[i] + padding 

        # add dimension for simplicity later
        caption_lengths = [[i] for caption_length in caption_lengths]
        return encoded_captions, caption_lengths

    def decode(self, seqs):
        unk = self.stoi.get(UNKNOWN)
        decoded_captions = [[self.itos[encoding] for encoding in seq] for seq in seqs]
        return decoded_captions
        
    def get_unknown_idx(self):
        """Returns the integer index of the special dummy word representing unknown words."""
        return self.stoi[UNKNOWN]
    
    def get_pad_idx(self):
        """Returns the integer index of the special padding dummy word."""
        return self.stoi[PAD]
    
    def __len__(self):
        return len(self.itos)
