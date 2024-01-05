import os

import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as VT
import torchaudio.transforms as AT

from serina.config import conf
from serina.dataset.label import  label_to_index
from serina.dataset.audio import build_transform
import pandas as pd

df = pd.read_csv(os.path.dirname(__file__) + "/../../ESC-50/meta/esc50.csv")


def select(start_percent, end_percent):
    def __inner_select(group):
        start = int(len(group) * start_percent)  # 计算 60% 的位置
        end = int(len(group) * end_percent)  # 计算 80% 的位置
        return group.iloc[start:end]

    sampled_df = df.groupby('category').apply(__inner_select)

    # 重置索引
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df


# Load basic info of audio
class RawWaveSet(Dataset):
    def __init__(self, start_percent, end_percent):
        self.df = select(start_percent, end_percent)

    def __len__(self):
        return len(self.df)

    def to_data_loader(self):
        return DataLoader(self, shuffle=True, batch_size=conf["batch_size"])

    def __getitem__(self, item):
        row = self.df.take([item], axis=0)

        file_path = os.path.dirname(__file__) + "/../../ESC-50/audio/" + row.filename.values[0]
        waveform, sample_rate = torchaudio.load(file_path)

        return waveform, sample_rate, file_path, row.category.values[0]


class ResampledWaveSet(RawWaveSet):
    def __init__(self, target_sample_rate, start_percent, end_percent):
        super().__init__(start_percent, end_percent)
        self.target_sample_rate = target_sample_rate

    def __getitem__(self, item):
        waveform, sample_rate, path, category = super().__getitem__(item)
        if sample_rate != self.target_sample_rate:
            waveform = AT.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)(waveform)
        return waveform, sample_rate, path, category


class TransformedWaveSet(RawWaveSet):
    def __init__(self, transform, start_percent, end_percent):
        super().__init__(start_percent, end_percent)
        self.transform = transform

    def __getitem__(self, item):
        waveform, sample_rate, path, category = super().__getitem__(item)
        tensor = self.transform(waveform)
        return tensor, sample_rate, path, category


class EasySet(TransformedWaveSet):
    def __init__(self, start_percent, end_percent):
        super().__init__(transform=build_transform(), start_percent=start_percent, end_percent=end_percent)
        self.cache = {}

    def __getitem__(self, item):
        if item in self.cache:
            return self.cache[item]

        tensor, sample_rate, path, category = super().__getitem__(item)

        r = tensor, label_to_index(category)
        self.cache[item] = r
        return r
