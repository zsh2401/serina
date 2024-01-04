import torch

from serina import get_categories, PTH_NAME, DEVICE, SAMPLE_RATE, standardize, waveform_to_mel_spectrogram, \
    waveform_to_log_mel_spectrogram, waveform_to_spectrogram, S_TYPE, spectrogram_to_image_tensor, index_to_label
from serina.model import create_model
import pyaudio
import wave
import numpy as np


class SerinaApplication:
    def __init__(self):
        self.model = create_model(get_categories())
        checkpoint = torch.load(PTH_NAME)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(DEVICE)

    def listen_to_microphone(self, chunk_size=1024):
        audio = pyaudio.PyAudio()
        frames = []
        stream = audio.open(format=pyaudio.paInt16, channels=1,
                            rate=SAMPLE_RATE, input=True,
                            frames_per_buffer=chunk_size)
        CHUNKS_PER_SECOND = int(1 / (chunk_size / SAMPLE_RATE))
        X_FRAMES = 4 * CHUNKS_PER_SECOND
        i = 0
        print("Listening")
        while True:
            data = stream.read(chunk_size)
            frames.append(data)
            if i % CHUNKS_PER_SECOND == 0 and len(frames) > X_FRAMES:
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                # 转换为 torch 张量
                waveform = torch.from_numpy(audio_data).float()
                result = self.check(waveform, SAMPLE_RATE)
                print(f"result is {result}")

    def check(self, waveform, sample_rate) -> str:
        waveform = standardize(waveform, sample_rate, SAMPLE_RATE)

        if S_TYPE == "mel":
            waveform = waveform_to_mel_spectrogram(waveform, sample_rate)
        elif S_TYPE == "log_mel":
            waveform = waveform_to_log_mel_spectrogram(waveform, sample_rate)
        else:
            waveform = waveform_to_spectrogram(waveform, sample_rate)

        waveform = spectrogram_to_image_tensor(waveform)
        with torch.no_grad():
            Y = self.model(waveform)
            predicted = torch.max(Y.data, 1)
            return index_to_label(predicted)
