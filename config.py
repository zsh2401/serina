import os

import torch.cuda

DEVICE = "cpu"
SAMPLE_RATE = 16_000
BATCH_SIZE = 64


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

if "DEVICE" in os.environ:
    DEVICE = str(os.environ["DEVICE"])
