#!pip install musdb

import musdb
from config import *

mus = musdb.DB(subsets='test', root=WAV_PATH)

#! musdbconvert path/to/musdb-stems-root path/to/new/musdb-wav-root

## Convert from STEMS format to .wav format

for tracks in mus:
    print(tracks)

print(type(mus[0]))

print(mus[0].audio)

