import os

import torch.cuda

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

if "DEVICE" in os.environ:
    device = str(os.environ["DEVICE"])
