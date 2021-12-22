# https://www.analyticsvidhya.com/blog/2020/11/attention-mechanism-for-caption-generation/
# https://arxiv.org/pdf/2106.15880.pdf

# decrease exposure bias with e.g. scheduled sampling
# learning rate decay

import torch
import torch.nn as nn
import torchvision.models as models
from vocab import SOS, EOS, PAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, pool_size=14, dropout = 0.5):
        super().__init__()
        self.resnet = self.build_resnet()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.dropout = nn.Dropout(dropout)
        #print(self.resnet)


    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, 10, 10)
        features = self.dropout(features)
        features = self.adaptive_pool(features) # (batch_size, 2048, 14, 14)
        return features


    # fetches pretrained inception v3 network and freezes its hidden layers, then overrides default top layer with a custom one that has requires_grad = True
    def build_resnet(self, fine_tune=True):
        resnet = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-2])

        for param in resnet.parameters():
            param.requires_grad = False

        if(fine_tune):
            for layer in list(resnet)[5:]:
                for param in layer.parameters():
                    param.requires_grad = True

        return resnet


class AttentionModule(nn.Module):
    def __init__(self, feature_size, hidden_size, attention_size):
        super().__init__()
        self.attention_encoder = nn.Linear(feature_size, attention_size)
        self.attention_decoder = nn.Linear(hidden_size, attention_size)
        self.attention_common = nn.Linear(attention_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden_state):
        # compute context vectors for encoders and decoders (previous) output
        encoder_context = self.attention_encoder(features)                      # (batch_size, num_pixels, attention_size)
        decoder_context = self.attention_decoder(hidden_state).unsqueeze(1)       # (batch_size, attention_size) --unsqueeze(1)--> (batch_size, 1, attention_size)

        # compute additive context vector. Note that decoders context vector is broadcasted at dim=1 when adding the vectors
        common_context = self.attention_common(self.relu(encoder_context + decoder_context)).squeeze(2) # (batch_size, num_pixels)
        
        # compute alpha and use it to weigh the features (and sum the values along the num_pixels dimension)
        alpha = self.softmax(common_context)   # (batch_size, num_pixels)
        weighted_features = (features * alpha.unsqueeze(2)).sum(dim=1)   # (batch_size, feature_size)
        
        return weighted_features, alpha


class Decoder(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, attention_size, num_layers=1, feature_size = 2048, num_pixels=196, dropout = 0.5):
        super().__init__()

        if vocab.gensim_model:
            embed_size = vocab.vectors.shape[1]

        self.feature_size = feature_size
        self.vocab_size = len(vocab)
        self.num_pixels = num_pixels
        self.vocab = vocab

        self.embedding_model = vocab.make_embedding_layer(embed_size, fine_tune=False).to(device)
        self.lstm_cell = nn.LSTMCell(feature_size + embed_size, hidden_size, num_layers)
        self.init_h = nn.Linear(feature_size, hidden_size)  # used to compute an initial hidden state for the LSTMCell
        self.init_c = nn.Linear(feature_size, hidden_size)  # used to compute an initial cell state for the LSTMCell
        self.top_layer = nn.Linear(hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionModule(feature_size, hidden_size, attention_size)

    def init_LSTMCell(self, features):
        mean_features = features.mean(dim=1)
        h = self.init_h(mean_features)  # (batch_size, hidden_size)
        c = self.init_c(mean_features)  # (batch_size, hidden_size)
        return h, c
    
    def forward(self, features, captions, caption_lengths, debug=False):
        """
        :param features: encoded images, a tensor of dimension (batch_size, feature_size, feature_dim, feature_dim)
        :param captions: a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: a tensor of dimension (batch_size, 1)
        """

        batch_size = features.size(0)

        # flatten feature images so that all pixels lie in the same dimension and switch dimensions to enable easier computations
        features = features.flatten(start_dim=2).permute((0, 2, 1)) # (batch_size, num_pixels = 14*14, feature_size = 2048)
        if debug: print(f'features shape: {features.shape}')
        if debug: print(f'captions shape: {captions.shape}')
        if debug: print(f'caption_lengths shape: {caption_lengths.shape}')

        # initialize the state of the LSTM with the features tensor
        h, c = self.init_LSTMCell(features)
        if debug: print(f'h (init) shape: {h.shape}')
        if debug: print(f'c (init) shape: {c.shape}')

        # sort batch entries by caption length
        caption_lengths, sorted_indices = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        features = features[sorted_indices]
        captions = captions[sorted_indices]

        # caption embedding
        embeddings = self.embedding_model(captions) # (batch_size, max_caption_length, embed_size)
        if debug: print(f'embeddings shape: {embeddings.shape}')
        
        # create tensor of output lengths for each entry in the batch. -1 is because we don't predict <EOS> token
        output_lengths = (caption_lengths).tolist()

        # create tensors to hold predictions and alphas
        predictions = torch.zeros(batch_size, max(output_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(output_lengths), self.num_pixels).to(device)


        for t in range(max(output_lengths)):
            # Retrieve the effective batch size which is a tensor representing which entries in the batch we want to use and which we want to exclude. In this case we want to exclude all trailing padding tokens
            effective_batch_size = sum([output_length > t for output_length in output_lengths])
            if debug: print(f'effective_batch_size: {effective_batch_size}')

            weighted_features, alpha = self.attention(features[:effective_batch_size],h[:effective_batch_size])
            if debug: print(f'weighted_features shape: {weighted_features.shape}')
            if debug: print(f'alpha shape: {alpha.shape}')

            h, c = self.lstm_cell(torch.cat([embeddings[:effective_batch_size, t, :], weighted_features], dim=1), (h[:effective_batch_size], c[:effective_batch_size]))
            if debug: print(f'h_{t} shape: {h.shape}')
            if debug: print(f'c_{t} shape: {c.shape}')

            predictions[:effective_batch_size, t, :] = self.top_layer(self.dropout(h))
            alphas[:effective_batch_size, t, :] = alpha

        return predictions, captions, output_lengths, alphas, sorted_indices


    def predict(self, features, debug=False):
        """
        :param features: tensor representing an encoded image with dimensions: (1, feature_size, feature_dim, feature_dim)
        """

        assert features.size(0) == 1, 'inference input assumes that batch size is 1'

        # flatten features tensor
        features = features.flatten(start_dim=2).permute((0, 2, 1)).to(device) # (1, num_pixels = 14*14, feature_size = 2048)

        # initialize the state of the LSTM with the features tensor
        h, c = self.init_LSTMCell(features) # (1, hidden_size) for both h and c
        if debug: print(f'h (init) shape: {h.shape}')
        if debug: print(f'c (init) shape: {c.shape}')

        # initialize predictions with <start>
        predictions = torch.tensor([[self.vocab.stoi[SOS]]]).to(device)
        alphas = torch.zeros(1, 1, self.num_pixels).to(device)


        # initialize index
        i = 0

        # forward loop
        while True:
            # retrieve weighted features
            weighted_features, alpha = self.attention(features,h)

            # embed the last predicted word (<start> will be embedded in first iteration)
            embedding = self.embedding_model(predictions[:, i])

            # step lstm
            h, c = self.lstm_cell(torch.cat([embedding, weighted_features], dim=1), (h, c))

            # predict a word
            pred = self.top_layer(self.dropout(h))
            pred_word = torch.argmax(pred, dim=1)
            
            predictions = torch.cat((predictions, pred_word.unsqueeze(dim=0)), dim=1)
            alphas = torch.cat((alphas, alpha.unsqueeze(dim=0)), dim=1)

            if pred_word.squeeze(dim=0).item() == self.vocab.stoi[EOS] or i >= 50:
                break
            else:
                i += 1
  
        return predictions, alphas










            
