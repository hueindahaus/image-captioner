import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, embed_size, dropout = 0.5):
        super(Encoder, self).__init__()
        self.inception_v3 = self.build_inception_net(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        print(self.inception_v3)


    def forward(self, images):
        features = self.inception_v3(images)
        features = self.relu(features)
        features = self.dropout(features)
        return features


    # fetches pretrained inception v3 network and freezes its hidden layers, then overrides default top layer with a custom one that has requires_grad = True
    def build_inception_net(self, embed_size):
        inception_v3 = models.inception_v3(pretrained=True, aux_logits=False)

        for param in inception_v3.parameters():
            param.requires_grad = False

        #inception_v3.fc = nn.Linear(inception_v3.fc.in_features, embed_size) # override the default top layer of the inception network
        del inception_v3.avgpool
        del inception_v3.dropout
        del inception_v3.fc
        return inception_v3



class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout = 0.5):
        super(Decoider, self).__init__()
        self.embedding_model = nn.Embedding(vocab_size, embed_size,)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.top_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, captions):
        embeddings = self.embedding_model(captions)
        embeddings = self.dropout(embeddings)
