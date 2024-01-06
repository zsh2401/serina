import torch

conf = {
    "device": "cpu",
    "sample_rate": 44100,
    "batch_size": 64,
    "spec": "mel",
    "optimizer": "adam",
    "learn_rate": 0.0001,
    "epoch": -1,
    "spec_h": 512,
    "spec_w": 512
}
# 冗余
if torch.cuda.is_available():
    conf["device"] = "cuda"
elif torch.backends.mps.is_available():
    conf["device"] = "mps"
else:
    conf["device"] = "cpu"


def get_pth_name():
    return "serina-" + ".".join([str(conf[k]) for k in conf if k != "device"]) + ".pth"


def print_conf():
    print("==Serina Configration==")
    print(conf)
