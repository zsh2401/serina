conf = {
    "device": "cpu",
    "sample_rate": 44100,
    "batch_size": 64,
    "spec": "mel",
    "learn_rate": 0.0001,
    "epoch": -1,
}


def get_pth_name():
    return "".join([key for key in conf]) + ".pth"


def print_conf():
    print("==Serina Configration==")
    print(conf)
