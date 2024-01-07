import time

import torch

from serina import get_num_classes, index_to_label, \
    get_pth_name, conf
from serina.dataset.audio import standardize, build_transform
from serina.model import create_model
from playsound import playsound
import numpy as np
import vlc

from serina.player import Player


class SerinaApplication:
    def __init__(self):
        self.model = torch.nn.DataParallel(create_model(get_num_classes()))
        print(f"Loading {get_pth_name()}")
        checkpoint = torch.load(get_pth_name(), map_location=conf["device"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(conf["device"])
        self.model.eval()
        self.cont = 0
        self.playing = True
        self.endurance = 5
        # self.player = vlc.MediaPlayer("./声音玩具_你的城市.mp3")
        self.player = vlc.MediaPlayer("./xx.mp3")

    def listen_to_microphone(self, chunk_size=1024):
        import pyaudio
        audio = pyaudio.PyAudio()
        frames = []
        sample_rate = conf["sample_rate"]
        stream = audio.open(format=pyaudio.paInt16, channels=1,
                            rate=sample_rate, input=True,
                            frames_per_buffer=chunk_size)
        CHUNKS_PER_SECOND = int(1 / (chunk_size / sample_rate))
        X_CHUNKS = 4 * CHUNKS_PER_SECOND
        DURATION = X_CHUNKS * (chunk_size / sample_rate)
        print(f"{CHUNKS_PER_SECOND} chunks per second, check once per {X_CHUNKS} (near {DURATION}s)")
        i = 0
        print("Listening")
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
            i += 1
            if i % CHUNKS_PER_SECOND == 0 and len(frames) > X_CHUNKS + CHUNKS_PER_SECOND:
                # print(f"Before cut length {len(frames)} {frames[0]}")
                frames = frames[CHUNKS_PER_SECOND:]
                # print(f"After cut length {len(frames)} {frames[0]}")
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                # 转换为 torch 张量
                start = time.time()
                waveform = torch.from_numpy(audio_data).float()
                waveform /= 32768
                # print(waveform)
                result = self.check(waveform, sample_rate)
                # print(f"({time.time() - start:.2f}s) Past {DURATION:.2f}s result is {[f'{p[0]}:{p[1] * 100:.2f}%' for p in result]}")

                found_rain = False
                for p in result:
                    if p[0] == "rain" and self.player.get_state() != vlc.State.Playing:
                        if self.cont < 0:
                            self.cont = 0
                        self.cont += 1
                        found_rain = True
                        print("Serina felt the mistery wave and says: It's raining!!!")

                if found_rain is False:
                    if self.cont > 0:
                        self.cont = 0
                    self.cont -= 1

                if self.player.get_state() != vlc.State.Playing:
                    print(f"Current endurance {self.cont}")
                if self.cont > self.endurance and self.player.get_state() != vlc.State.Playing:
                    print("Let's Fall in Love for the Night")
                    self.player.play()
                    self.player.set_time(63 * 1000)
                # elif self.cont < -self.endurance and self.player.get_state() == vlc.State.Playing:
                #     self.player.pause()

    def check(self, waveform, sample_rate) -> str:
        waveform = standardize(waveform, sample_rate, conf["sample_rate"])
        waveform = build_transform()(waveform)
        inputs = torch.stack([waveform], 0).to(conf["device"])

        with torch.no_grad():
            outputs = self.model(inputs)

        probabilities = torch.nn.functional.softmax(outputs, 1)
        # result = []
        for image_pro in probabilities:
            img_pros = []

        for i, p in enumerate(image_pro):
            img_pros.append((index_to_label(i), float(p)))
        img_pros.sort(key=lambda v: v[1], reverse=True)
        img_pros = img_pros[:3]
        return img_pros

        # result.append(img_pros)
        #
        # return result
