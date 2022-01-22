import torch

# Flexible args
BATCH_SIZE              = 8
NUM_EPOCHS              = 20
START_EPOCH             = 0
HIDDEN_SIZE             = 512
EMBED_SIZE              = 512
ATTENTION_SIZE          = 512
NUM_HIDDEN_LAYERS       = 2
ENCODER_DROPOUT         = 0.5
DECODER_DROPOUT         = 0.5
ATTENTION_DROPOUT       = 0.5
NUM_WORKERS             = 1
FINE_TUNE_EMBEDDINGS    = True
# Learning rates
ENCODER_LR              = 1e-4
DECODER_LR              = 4e-4

DEVICE                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PRINT_EVERY_NTH_BATCH   = 1000


def export_config_dict():
    dict = {
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'ENCODER_LR': ENCODER_LR,
        'DECODER_LR': DECODER_LR,
        'START_EPOCH': START_EPOCH,
        'HIDDEN_SIZE': HIDDEN_SIZE,
        'EMBED_SIZE': EMBED_SIZE,
        'ATTENTION_SIZE': ATTENTION_SIZE,
        'NUM_HIDDEN_LAYERS': NUM_HIDDEN_LAYERS,
        'ENCODER_DROPOUT': ENCODER_DROPOUT,
        'DECODER_DROPOUT': DECODER_DROPOUT,
        'ATTENTION_DROPOUT':ATTENTION_DROPOUT,
        'NUM_WORKERS': NUM_WORKERS,
        'FINE_TUNE_EMBEDDINGS': FINE_TUNE_EMBEDDINGS,
        'ENCODER_LR': ENCODER_LR,
        'DECODER_LR': DECODER_LR,
        'DEVICE': DEVICE,
        'PRINT_EVERY_NTH_BATCH': PRINT_EVERY_NTH_BATCH
    }
    return dict