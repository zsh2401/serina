#!/usr/bin/env python
import argparse

import torch

parser = argparse.ArgumentParser(description='Serina Sound Classification')

parser.add_argument("--batch-size", type=int, help="The batch size for training, default is 64.", default=64)
parser.add_argument("--sample-rate", type=int, help="The sample rate.", default=44100)
parser.add_argument("--epoch", type=int, help="Training epochs, -1 stand for infinite.", default=-1)
parser.add_argument("--device", type=str, help="Running on certain device, default is auto.", default="auto")
parser.add_argument("--learn-rate", type=float, help="Initial learn rate for training", default=0.001)
parser.add_argument("--spec", type=str, help="The spectrogram type", choices=["mel", "normal", "log-mel"],
                    default='mel-spec')
subparser = parser.add_subparsers(dest="command")
subparser.add_parser("train", help="Train model")
subparser.add_parser("test", help="Test model")
subparser.add_parser("listen", help="Start classification with microphone and trained model")

args = parser.parse_args()

import serina.config

if args.device != "auto":
    serina.config.conf["device"] = args.device
elif torch.cuda.is_available():
    serina.config.conf["device"] = "cuda"
elif torch.backends.mps.is_available():
    serina.config.conf["device"] = "mps"
else:
    serina.config.conf["device"] = "cpu"

serina.config.conf["epoch"] = args.epoch
serina.config.conf["spec"] = args.spec
serina.config.conf["sample_rate"] = args.sample_rate
serina.config.conf["learn_rate"] = args.learn_rate
serina.config.print_conf()
if args.command == "train":
    import serina.train
