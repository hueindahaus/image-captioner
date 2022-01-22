import math
import torch
import torch.nn as nn
import torchvision.models as models
from vocab import SOS, EOS, PAD
import config

device = config.DEVICE

class Encoder(nn.Module):
    def __init__(self, dropout = 0.2):
        super().__init__()
        self.resnet = self.build_resnet()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.dropout = nn.Dropout(dropout)


    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, 10, 10)
        features = self.dropout(features)
        features = self.adaptive_pool(features) # (batch_size, 2048, 14, 14)
        return features


    # fetches pretrained inception v3 network and freezes its hidden layers, then overrides default top layer with a custom one that has requires_grad = True
    def build_resnet(self, fine_tune=True):
        resnet = nn.Sequential(*list(models.resnet101(pretrained=True).to(device).children())[:-2])

        for param in resnet.parameters():
            param.requires_grad = False

        if(fine_tune):
            for layer in list(resnet)[5:]:
                for param in layer.parameters():
                    param.requires_grad = True

        return resnet


class AttentionModule(nn.Module):
    def __init__(self, feature_size, hidden_size, attention_size, dropout = 0.2):
        super().__init__()
        self.attention_encoder = nn.Linear(feature_size, attention_size)
        self.attention_decoder = nn.Linear(hidden_size, attention_size)
        self.attention_common = nn.Linear(attention_size, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, hidden_state):
        # compute context vectors for encoders and decoders (previous) output
        encoder_context = self.attention_encoder(features)                      # (batch_size, num_pixels, attention_size)
        decoder_context = self.attention_decoder(hidden_state).unsqueeze(1)       # (batch_size, num_pixels) --unsqueeze(1)--> (batch_size, 1, attention_size)

        # compute additive context vector. Note that decoders context vector is broadcasted at dim=1 when adding the vectors
        common_context = self.attention_common(self.dropout(self.relu(encoder_context + decoder_context))).squeeze(2) # (batch_size, num_pixels)
        
        # compute alpha and use it to weigh the features (and sum the values along the num_pixels dimension)
        alpha = self.softmax(common_context)   # (batch_size, num_pixels)
        weighted_features = (features * alpha.unsqueeze(2)).sum(dim=1)   # (batch_size, feature_size)
        
        
        return weighted_features, alpha


class Decoder(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, attention_size, num_layers=2, feature_size = 2048, num_pixels=196, dropout = 0.2, attention_dropout=0.2, fine_tune_embeddings = True,):
        super().__init__()

        if vocab.gensim_model:
            embed_size = vocab.vectors.shape[1]

        self.feature_size = feature_size
        self.vocab_size = len(vocab)
        self.num_pixels = num_pixels
        self.vocab = vocab

        self.embedding_model = vocab.make_embedding_layer(fine_tune = fine_tune_embeddings).to(device)
        self.lstm_l = nn.LSTMCell(feature_size + hidden_size, hidden_size, num_layers)
        self.init_h_l = nn.Linear(feature_size, hidden_size)  # used to compute an initial hidden state for the LSTMCell
        self.init_c_l = nn.Linear(feature_size, hidden_size)  # used to compute an initial cell state for the LSTMCell
        self.lstm_tda = nn.LSTMCell(feature_size + embed_size + hidden_size, hidden_size, num_layers)
        self.init_h_tda = nn.Linear(feature_size, hidden_size)  # used to compute an initial hidden state for the LSTMCell
        self.init_c_tda = nn.Linear(feature_size, hidden_size)  # used to compute an initial cell state for the LSTMCell
        self.top_layer = nn.Linear(hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionModule(feature_size, hidden_size, attention_size, dropout = attention_dropout)
        self.relu = nn.ReLU()

        self.f_beta = nn.Linear(hidden_size, feature_size)   # linear layer for creating a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()

    def init_LSTMCell(self, features, is_language_lstm):
        mean_features = features.mean(dim=1)
        if is_language_lstm:
            h = self.init_h_l(mean_features)  # (batch_size, hidden_size)
            c = self.init_c_l(mean_features)  # (batch_size, hidden_size)
        else:
            h = self.init_h_tda(mean_features)  # (batch_size, hidden_size)
            c = self.init_c_tda(mean_features)  # (batch_size, hidden_size)
        return h, c
    
    def forward(self, features, captions, caption_lengths, scheduled_sampling_p=0, debug=False):

        batch_size = features.size(0)

        # flatten feature images so that all pixels lie in the same dimension and switch dimensions to enable easier computations
        features = features.flatten(start_dim=2).permute((0, 2, 1)) # (batch_size, num_pixels = 14*14, feature_size = 2048)

        # sort batch entries by caption length
        caption_lengths, sorted_indices = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        features = features[sorted_indices]
        captions = captions[sorted_indices]

        # initialize the state of the LSTM with the features tensor
        h_l, c_l = self.init_LSTMCell(features, True)
        h_tda, c_tda = self.init_LSTMCell(features, False)

        # caption embedding
        embeddings = self.embedding_model(captions) # (batch_size, max_caption_length, embed_size)
        
        # create tensor of output lengths for each entry in the batch. -1 is because we don't predict on <EOS> token
        output_lengths = (caption_lengths-1).tolist()

        # create tensors to hold predictions and alphas
        predictions = torch.zeros(batch_size, max(output_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(output_lengths), self.num_pixels).to(device)

        for t in range(max(output_lengths)):
            # Retrieve the effective batch size which is a tensor representing which entries in the batch we want to use and which we want to exclude. In this case we want to exclude all trailing padding tokens
            effective_batch_size = sum([output_length > t for output_length in output_lengths])

            # scheduled sampling - with probability scheduled_sampling_p: take previously predicted word and embed it and input instead of gold standard token
            prev_embedding = embeddings[:effective_batch_size, t, :]
            if t > 0 and torch.rand(1).item() <= scheduled_sampling_p:
                prev_embedding = self.embedding_model(predictions[:effective_batch_size, t-1].argmax(dim=1))

            h_tda, c_tda = self.lstm_tda(torch.cat((h_l[:effective_batch_size],features.mean(dim=1)[:effective_batch_size], prev_embedding), dim=1), (h_tda[:effective_batch_size],c_tda[:effective_batch_size]))

            weighted_features, alpha = self.attention(features[:effective_batch_size],h_tda[:effective_batch_size])

            h_l, h_c = self.lstm_l(torch.cat((weighted_features[:effective_batch_size], h_tda[:effective_batch_size]), dim=1), (h_l[:effective_batch_size], c_l[:effective_batch_size]))
            
            predictions[:effective_batch_size, t, :] = self.top_layer(self.dropout(h_l))
            alphas[:effective_batch_size, t, :] = alpha

        return predictions, captions, output_lengths, alphas, sorted_indices

    def beam(self, features_batch, k_init=3, debug=True):
        
        preds = []
        alphas = []

        for index, features in enumerate(features_batch):
            k = k_init
            # flatten features tensor
            features = features.unsqueeze(dim=0).flatten(start_dim=2).permute((0, 2, 1)) # (1, num_pixels = 14*14, feature_size = 2048)
            #print(features.shape)

            # To run beam search efficiently we treat the problem as of having k sequences in k entries in the same batch
            features = features.expand(k, features.size(1), features.size(2)) # (k, num_pixels = 14*14, feature_size = 2048)

            # Tensor to hold top k prev words and initialize with SOS-token
            k_prev_words = torch.tensor([[self.vocab.stoi[SOS]] for i in range(k)]).to(device)   # (k, 1)

            # Tensor to hold the k best sequences
            k_seqs = k_prev_words

            # Tensor to hold the scores for the sequences
            k_scores = torch.zeros(k, 1).to(device)

            # Tensor to hold the alphas for the sequences
            k_seqs_alphas = torch.ones(k, 1, self.num_pixels).to(device)

            completed_seqs = []
            completed_seqs_alphas = []
            completed_scores = []

            # initialize the state of the LSTM with the features tensor
            h_l, c_l = self.init_LSTMCell(features, True) # (k, hidden_size) for both h and c
            h_tda, c_tda = self.init_LSTMCell(features, False) # (k, hidden_size) for both h and c

            step = 0

            while k > 0 and step < 50:
                step += 1

                # Embed the last predicted words
                embedding = self.embedding_model(k_prev_words).squeeze(1)

                # step top down attention lstm
                h_tda, c_tda = self.lstm_tda(torch.cat((h_l ,features.mean(dim=1), embedding), dim=1), (h_tda, c_tda))

                # retrieve weighted features
                weighted_features, alpha = self.attention(features,h_tda)

                # Step language lstm
                h_l, h_c = self.lstm_l(torch.cat((weighted_features, h_tda), dim=1), (h_l, c_l))

                # Predict and extract scores (probability outputs)
                scores = nn.functional.log_softmax(self.top_layer(self.dropout(h_l)), dim=1)
                scores = k_scores[scores.size(0)-1] + scores

                k_scores, k_next_words = scores.view(-1).topk(k, 0, True, True)

                # Convert unrolled indices to actual indices of scores
                k_next_words = k_next_words % self.vocab_size

                # Add new words and alphas to holding tensors
                k_seqs = torch.cat((k_seqs, k_next_words.unsqueeze(1)), dim=1)
                k_seqs_alphas = torch.cat((k_seqs_alphas, alpha.unsqueeze(1)), dim=1)

                # Check for EOS token to set sequences as complete
                incompleted_idcs = []
                completed_idcs = []
                if step == 50:
                    completed_idcs = [i for i in range(len(k_next_words))]
                else:
                    for i, next_word in enumerate(k_next_words):
                        if next_word == self.vocab.stoi[EOS]:
                            completed_idcs.append(i)
                        else:  
                            incompleted_idcs.append(i)

                if len(completed_idcs) > 0:
                    completed_seqs.extend(k_seqs[completed_idcs].tolist())
                    completed_seqs_alphas.extend(k_seqs_alphas[completed_idcs].tolist())
                    completed_scores.extend(k_scores[completed_idcs])
                    k -= len(completed_idcs)
                
                k_seqs = k_seqs[incompleted_idcs]
                k_seqs_alphas = k_seqs_alphas[incompleted_idcs]
                h_tda = h_tda[incompleted_idcs]
                c_tda = c_tda[incompleted_idcs]
                h_l = h_l[incompleted_idcs]
                c_l = c_l[incompleted_idcs]
                features = features[incompleted_idcs]
                k_scores = k_scores[incompleted_idcs].unsqueeze(1)
                k_prev_words = k_next_words[incompleted_idcs].unsqueeze(1)
                
            # simply argmax the best score in the list of scores and get the index with the best sequence and alphas and populate preds and alphas
            #print(completed_scores)
            best_index = completed_scores.index(max(completed_scores))
            preds.append(completed_seqs[best_index])
            alphas.append(torch.tensor(completed_seqs_alphas[best_index]))

        return preds, alphas
            











            
