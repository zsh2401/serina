#!/usr/bin/env python
import pyaudio
import wave

# 录音参数设置
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1              # 声道数
RATE = 44100              # 采样率
CHUNK = 1024              # 块大小
RECORD_SECONDS = 5        # 录音时间
WAVE_OUTPUT_FILENAME = "output.wav"  # 输出文件名

# 初始化 pyaudio
audio = pyaudio.PyAudio()

# 打开录音流
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# 录音过程
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording.")

# 停止录音
stream.stop_stream()
stream.close()
audio.terminate()

# 保存录音文件
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
