import musdb
from config import *

mus = musdb.DB(subsets='test', root=WAV_PATH)

#!musdbconvert WAV_PATH

for tracks in mus:
    print(tracks)

print(type(mus[0]))

mus[0].audio

