#!/usr/bin/env python
from serina import SerinaApplication
import sys
import torchaudio

def predict(path:str):
    waveform, sample_rate = torchaudio.load(path)
    # os.arg
    app = SerinaApplication()
    print(app.check(waveform,sample_rate))