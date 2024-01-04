import torch

from serina import get_categories, PTH_NAME, DEVICE, SAMPLE_RATE, standardize, waveform_to_mel_spectrogram, \
    waveform_to_log_mel_spectrogram, waveform_to_spectrogram, S_TYPE, spectrogram_to_image_tensor, index_to_label
from serina.model import create_model

import numpy as np


class SerinaApplication:
    def __init__(self):
        self.model = create_model(get_categories())
        # checkpoint = torch.load(PTH_NAME)
        # self.model.load_state_dict(checkpoint["model"])
        self.model.to(DEVICE)

    def listen_to_microphone(self, chunk_size=1024):
        import pyaudio
        audio = pyaudio.PyAudio()
        frames = []
        stream = audio.open(format=pyaudio.paInt16, channels=1,
                            rate=SAMPLE_RATE, input=True,
                            frames_per_buffer=chunk_size)
        CHUNKS_PER_SECOND = int(1 / (chunk_size / SAMPLE_RATE))
        X_CHUNKS = 4 * CHUNKS_PER_SECOND
        i = 0
        print("Listening")
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
            i += 1
            if i % CHUNKS_PER_SECOND == 0 and len(frames) > X_CHUNKS + CHUNKS_PER_SECOND:
                frames = frames[CHUNKS_PER_SECOND:]
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                # 转换为 torch 张量
                waveform = torch.from_numpy(audio_data).float()
                result = self.check(waveform, SAMPLE_RATE)
                print(f"result is {result[:3]}")

    def check(self, waveform, sample_rate) -> str:
        waveform = standardize(waveform, sample_rate, SAMPLE_RATE)

        if S_TYPE == "mel":
            waveform = waveform_to_mel_spectrogram(waveform, sample_rate)
        elif S_TYPE == "log_mel":
            waveform = waveform_to_log_mel_spectrogram(waveform, sample_rate)
        else:
            waveform = waveform_to_spectrogram(waveform, sample_rate)

        waveform = spectrogram_to_image_tensor(waveform)
        print(waveform.shape)
        X = torch.stack([waveform], 0).to(DEVICE)
        with torch.no_grad():
            Y = self.model(X)
            # print(Y)
            probabilities = torch.nn.functional.softmax(Y, 1)
            result = []
            for image_pro in probabilities:
                img_pros = []
                for i, p in enumerate(image_pro):
                    img_pros.append((index_to_label(i), float(p)))
                img_pros.sort(key=lambda v: v[1], reverse=True)
                img_pros = img_pros[:3]
                result.append(img_pros)

            return result
