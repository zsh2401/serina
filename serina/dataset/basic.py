import os.path
import time

import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader

from serina.util.audio import *
from serina.config import SAMPLE_RATE, BATCH_SIZE, S_TYPE
from serina.dataset.label import label_to_index

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


class SoundDataset(Dataset):
    def __init__(self, start_percent, end_percent):
        self.df = select(start_percent, end_percent)
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def to_data_loader(self):
        return DataLoader(self, shuffle=True, batch_size=BATCH_SIZE)

    def get_raw_info(self, item):
        row = self.df.take([item], axis=0)

        file_path = os.path.dirname(__file__) + "/../../ESC-50/audio/" + row.filename.values[0]
        waveform, sample_rate = torchaudio.load(file_path)

        return waveform, sample_rate, file_path, row.category.values[0]

    def get_item_with_file_path(self, item):
        if item in self.cache:
            # print("cahced")
            return self.cache[item]

        start = time.time()
        waveform, sample_rate, file_path, category = self.get_raw_info(item)
        waveform = standardize(waveform, sample_rate, SAMPLE_RATE)

        if S_TYPE == "mel":
            waveform = waveform_to_mel_spectrogram(waveform, sample_rate)
        elif S_TYPE == "log_mel":
            waveform = waveform_to_log_mel_spectrogram(waveform, sample_rate)
        else:
            waveform = waveform_to_spectrogram(waveform, sample_rate)

        waveform = spectrogram_to_image_tensor(waveform)

        result = waveform, label_to_index(category), file_path
        self.cache[item] = result
        # start = time.time()
        # print(f"used {time.time() - start}s")
        return result

    def __getitem__(self, item):
        w, l, f = self.get_item_with_file_path(item)
        return w, l


class TrainSet(SoundDataset):
    def __init__(self):
        super().__init__(0, 0.6)


class ValidationSet(SoundDataset):
    def __init__(self):
        super().__init__(0.6, 0.8)


class TestSet(SoundDataset):
    def __init__(self):
        super().__init__(0.8, 1)
