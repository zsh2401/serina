import os

import torch.cuda

DEVICE = "cpu"
SAMPLE_RATE = 32_000
BATCH_SIZE = 64
# log_mel_spec, mel_spec, spec
S_TYPE = "mel_spec"
LEARN_RATE=0.001

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

if "DEVICE" in os.environ:
    DEVICE = str(os.environ["DEVICE"])

if "SAMPLE_RATE" in os.environ:
    DEVICE = int(os.environ["SAMPLE_RATE"])

if "S_TYPE" in os.environ:
    DEVICE = int(os.environ["S_TYPE"])

PTH_NAME = f"model.{S_TYPE}.{SAMPLE_RATE}hz.pth"
