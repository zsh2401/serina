import torchaudio
import torchaudio.transforms as AT
from label import label_to_index
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as VT
df = pd.read_csv("./ESC-50/meta/esc50.csv")


def select(start_percent, end_percent):
    def __inner_select(group):
        start = int(len(group) * start_percent)  # 计算 60% 的位置
        end = int(len(group) * end_percent)  # 计算 80% 的位置
        return group.iloc[start:end]

    sampled_df = df.groupby('category').apply(__inner_select)

    # 重置索引
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df


def waveform_to_spectrogram(waveform, sample_rate):
    # 转换为梅尔频谱图
    spectrogram_transform = AT.MelSpectrogram(sample_rate=sample_rate)
    mel_spectrogram = spectrogram_transform(waveform)

    # 转换为对数尺度
    log_mel_spectrogram = AT.AmplitudeToDB()(mel_spectrogram)

    return log_mel_spectrogram


class SoundDataset(Dataset):
    def __init__(self, start_percent, end_percent):
        self.spectrogram_transform = AT.MelSpectrogram(sample_rate=16_000)
        # self.
        self.vision_transform = VT.Compose([
            VT.ToPILImage(),
            VT.Lambda(lambda x: x.convert('RGB')),
            VT.Resize(224),
            VT.RandomCrop(224),
            VT.ToTensor(),  # 将图片转换为Tensor
            VT.Normalize(mean=[0.485, 0.456, 0.406],  # 图像标准化
                         std=[0.229, 0.224, 0.225])
        ])
        self.target_sample_rate = 16_000
        self.df = select(start_percent, end_percent)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.take([item], axis=0)

        file_path = "./ESC-50/audio/" + row.filename.values[0]
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        spectrogram = self.spectrogram_transform(waveform)
        # spectrogram = waveform_to_spectrogram(waveform, self.target_sample_rate)
        waveform_conv_tensor = self.vision_transform(spectrogram)

        return waveform_conv_tensor, label_to_index(row.category.values[0])


class TrainSet(SoundDataset):
    def __init__(self):
        super().__init__(0, 0.6)


class ValidationSet(SoundDataset):
    def __init__(self):
        super().__init__(0.6, 0.8)


class TestSet(SoundDataset):
    def __init__(self):
        super().__init__(0.8, 1)
