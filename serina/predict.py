#!/usr/bin/env python
from serina import SerinaApplication
import sys
import torchaudio

path = sys.argv[1]
waveform, sample_rate = torchaudio.load(path)
# os.arg
app = SerinaApplication()
print(app.check(waveform,sample_rate))