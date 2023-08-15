import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import librosa
import os
import glob
import random
import soundfile as sf

from config import *

def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

    def mix(s1_dvec, s1_target, s2, train, num, out_dir):
    srate = 16000
    dir_ = os.path.join(out_dir, 'train' if train else 'test')
    
    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
        'wav files must be mono, not stereo'
    
    d, _ = librosa.effects.trim(d, top_db=20)
    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)
    
    #print(d,w1,w2)
    
    # if reference for d-vector is too short, discard it
    if d.shape[0] < 1.1 * 80 * 160:
        return
    # LibriSpeech dataset have many silent interval, so let's vad-merge them
    # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
    #if args.vad == 1:
    w1, w2 = vad_merge(w1), vad_merge(w2)
    
    #print("check")
        
    # if merged audio is shorter than `L`, discard it
    L = int(srate * 3.0)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1, w2 = w1[:L], w2[:L]
    
    mixed = w1 + w2
    
    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1/norm, w2/norm, mixed/norm
    
    print(w1)
    
    target_ext = '*-target.wav'
    mixed_ext = '*-mixed.wav'
    
    #s1_target_wav = s1_target.replace(".flac", ext)
    
    # save vad & normalized wav files
    #target_wav_path = os.path.join(dir_, target_ext)
    #mixed_wav_path = os.path.join(dir_, mixed_ext)
    target_wav_path = formatter(dir_, target_ext, num)
    mixed_wav_path = formatter(dir_, mixed_ext, num)
    
    #print(target_wav_path)
    
    sf.write(target_wav_path, np.ravel(w1), srate)
    sf.write(mixed_wav_path, np.ravel(mixed), srate)
    
    #print(mixed)
    return mixed
    
LIBRI_DIR = LIBRISPEECH_PATH
test_folders = [x for x in glob.glob(os.path.join(LIBRI_DIR, 'dev-clean', '*'))]

test_spk = [glob.glob(os.path.join(spk, '**', '*.flac'), recursive=True) for spk in test_folders]

test_spk = [x for x in test_spk if len(x) >= 2]

def test_wrapper():
    spk1, spk2 = random.sample(test_spk, 2)
    s1_dvec, s1_target = random.sample(spk1, 2)
    s2 = random.choice(spk2)
    aud = mix(s1_dvec, s1_target, s2, train=False, num=1, out_dir=MIXED_AUDIO_OUT_DIR)
    return aud
mixed_aud = test_wrapper()