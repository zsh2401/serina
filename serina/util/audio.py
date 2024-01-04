# This file includes
#
import torch
import torchaudio.transforms as AT
import torchvision.transforms as VT
from scipy.ndimage import zoom


def standardize(waveform, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        resampler = AT.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform


def waveform_to_log_mel_spectrogram(waveform, sample_rate):
    # 转换为梅尔频谱图
    spectrogram_transform = AT.MelSpectrogram(sample_rate=sample_rate)
    mel_spectrogram = spectrogram_transform(waveform)

    # 转换为对数尺度
    log_mel_spectrogram = AT.AmplitudeToDB()(mel_spectrogram)

    return log_mel_spectrogram


def waveform_to_mel_spectrogram(waveform, sample_rate):
    # 转换为梅尔频谱图
    spectrogram_transform = AT.MelSpectrogram(sample_rate=sample_rate, f_max=18000, n_mels=224, n_fft=4096,
                                              normalized=True, win_length=2205, hop_length=308)
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


# do mlcc transform
def get_mlcc(waveform, sample_rate):
    mfcc_transform = AT.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,  # 通常选择的MFCC数量
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    # 应用MFCC变换
    return mfcc_transform(waveform)


def combine_spectrogram_and_mlcc(transformed_spec_tensor, mfcc_tensor):
    h_scale = 224 / mfcc_tensor.shape[0]
    w_scale = 224 / mfcc_tensor.shape[1]
    scale_factor = [224 / mfcc_tensor.shape[0], 224 / mfcc_tensor.shape[1]]

    # 使用zoom进行重采样
    mfcc_resized = zoom(mfcc_tensor, (h_scale, w_scale))

    mfcc_tensor = torch.from_numpy(mfcc_resized).float()
    mfcc_tensor = mfcc_tensor.unsqueeze(0).unsqueeze(0)  # 添加两个新的维度

    return torch.stack([transformed_spec_tensor, mfcc_tensor], dim=0)


vision_transform = VT.Compose([
    VT.ToPILImage(),
    VT.Lambda(lambda x: x.convert('RGB')),
    # VT.Resize((224, 224)),
    VT.Resize((224, 224)),
    VT.ToTensor(),  # 将图片转换为Tensor
    VT.Normalize(mean=[0.485, 0.456, 0.406],  # 图像标准化
                 std=[0.229, 0.224, 0.225])
])


# vision_transform_without_resize = VT.Compose([
#     VT.ToPILImage(),
#     VT.Lambda(lambda x: x.convert('RGB')),
#     VT.Resize((224, 224 * 6)),
#     VT.ToTensor(),  # 将图片转换为Tensor
#     VT.Normalize(mean=[0.485, 0.456, 0.406],  # 图像标准化
#                  std=[0.229, 0.224, 0.225])
# ])


def spectrogram_to_image_tensor(waveform):
    return vision_transform(waveform)
