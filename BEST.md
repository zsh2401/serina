目前性能最好的组合是：
```python
transform = VT.Compose([
    AT.MelSpectrogram(sample_rate=sample_rate, f_max=18000, n_mels=224, n_fft=4096,
                      win_length=2205, hop_length=308),
    AT.AmplitudeToDB(),
    VT.Lambda(lambda x: x.repeat(3, 1, 1)),
    VT.Resize((224, 224)),
])
model = resnet18(pretrained=True)
```