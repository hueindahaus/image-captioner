import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import torch.nn.functional as F
import config
device = config.DEVICE

class SimLoss:
    def __init__(self, vocab):
        vocab_size = len(vocab)
        stopwords_indices = vocab.encode([stopwords.words('english')])[0][0]
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-08)
        embedding_model = vocab.make_embedding_layer(fine_tune = False).to(device)
        embeddings = embedding_model(torch.tensor([[i for i in range(vocab_size)]]).to(device)).squeeze(dim=0)
        S_l2 = torch.cdist(embeddings, embeddings, p=2).detach().cpu()
        S_l2 /= S_l2.max()
        S_l2 = 1 - S_l2
        
        S_cosine_sim = torch.empty((vocab_size, vocab_size))
        # Loop through each word index and create an expanded embedding tensor which will be evaluated with cosine_sim with the tensor of embeddings for the whole vocab
        # Thus S is created which is a matrice containing a cosine similarity score for each vocab entry pair
        for w in tqdm(range(vocab_size)):
            embedding_expanded = embeddings[w].unsqueeze(dim=0).expand(vocab_size, -1)
            S_cosine_sim[w] = cosine_sim(embedding_expanded, embeddings).clip(0, 1).detach().cpu()

        self.S = (S_l2 * S_cosine_sim)

        # set one hot encodings on the rows of stopwords since we are not interested in similar words (since the network can exploit "same" and "." having high embedding similarity)
        self.S[stopwords_indices] = F.one_hot(torch.tensor(stopwords_indices), num_classes=vocab_size).float()
        # cutoff all scores less than 0.5 and set them to 0
        self.S = torch.where(self.S >= 0.5, self.S, torch.tensor(0).float())   
        torch.cuda.empty_cache()
        
        
    def __call__(self, predictions, targets):
        """
        param predictions: tensor of shape (packed_sequence_length, vocab_size)
        param targets: tensor of shape (packed_sequence_length)
        """
        predictions = nn.functional.softmax(predictions, dim=1)
        S_y = self.S[targets].to(device)  # (packed_sequence_length, vocab_size)
        dots = (predictions * S_y).sum(dim=1) # (packed_sequence_length) represents the dot product in the formula
        dots = -torch.log(dots) # (packed_sequence_length)
        return torch.mean(dots, dim=0)  # scalar





