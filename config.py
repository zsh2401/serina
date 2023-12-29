import os

import torch.cuda

DEVICE = "cpu"
SAMPLE_RATE = 32_000
BATCH_SIZE = 64
# log_mel, mel, spec
S_TYPE = "mel_spec"
LEARN_RATE = 0.001
# -1 is infinite
EPOCH = -1

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

if "DEVICE" in os.environ:
    DEVICE = str(os.environ["DEVICE"])

if "SAMPLE_RATE" in os.environ:
    SAMPLE_RATE = int(os.environ["SAMPLE_RATE"])

if "S_TYPE" in os.environ:
    S_TYPE = str(os.environ["S_TYPE"])

if "EPOCH" in os.environ:
    EPOCH = int(os.environ["EPOCH"])

PTH_NAME = f"model.{S_TYPE}.{SAMPLE_RATE}hz.pth"

print("==Serina Configration==")
print(
    f"DEVICE {DEVICE}, SAMPLE_RATE {SAMPLE_RATE}hz, S_TYPE {S_TYPE}, BATCH_SIZE {BATCH_SIZE}, EPOCH {'Inf' if EPOCH < 0 else str(EPOCH)}")
print("==Serina Configration==")
