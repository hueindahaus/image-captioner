import torch

# Flexible args
BATCH_SIZE              = 8
NUM_EPOCHS              = 20
HIDDEN_SIZE             = 256
EMBED_SIZE              = 256
ATTENTION_SIZE          = 256
DROPOUT                 = 0.2
NUM_WORKERS             = 1
FINE_TUNE               = False

# Learning rates
ENCODER_LR              = 1e-4
DECODER_LR              = 4e-4

DEVICE                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PRINT_EVERY_NTH_BATCH   = 100