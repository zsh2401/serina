import torchaudio.transforms as AT
import torchvision.transforms as VT


def standardize(waveform, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        resampler = AT.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform


def waveform_to_mel_spectrogram(waveform, sample_rate):
    # 转换为梅尔频谱图
    spectrogram_transform = AT.MelSpectrogram(sample_rate=sample_rate)
    mel_spectrogram = spectrogram_transform(waveform)
    return mel_spectrogram

    # # 转换为对数尺度
    # log_mel_spectrogram = AT.AmplitudeToDB()(mel_spectrogram)
    #
    # return log_mel_spectrogram


def waveform_to_spectrogram(waveform, sample_rate):
    spectrogram_transform = AT.Spectrogram()
    mel_spectrogram = spectrogram_transform(waveform)
    return mel_spectrogram


vision_transform = VT.Compose([
    VT.ToPILImage(),
    VT.Lambda(lambda x: x.convert('RGB')),
    VT.Resize((224, 224)),
    # VT.RandomCrop(224),
    VT.ToTensor(),  # 将图片转换为Tensor
    VT.Normalize(mean=[0.485, 0.456, 0.406],  # 图像标准化
                 std=[0.229, 0.224, 0.225])
])


def spectrogram_to_image_tensor(waveform):
    return vision_transform(waveform)
