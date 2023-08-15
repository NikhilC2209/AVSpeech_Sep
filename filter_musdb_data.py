import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import librosa
from IPython.display import Audio, display, clear_output
import soundfile as sf

from config import *

#MUSDB_PATH = MUSDB_PATH    (from config.py)
#DEST_PATH = DEST_PATH


def create_spec(src, dest):
    if not os.path.exists(dest):
        os.mkdir(dest)
        os.mkdir(os.path.join(dest, "mixture"))
        os.mkdir(os.path.join(dest, "vocals"))
        os.mkdir(os.path.join(dest, "drums"))
        os.mkdir(os.path.join(dest, "bass"))
        os.mkdir(os.path.join(dest, "other"))
        os.mkdir(os.path.join(dest, "accompaniment"))
        os.mkdir(os.path.join(dest, "linear_mixture"))
    
    dest_folder_list = sorted(['mixture', 'drums', 'bass', 'other', 'vocals', 'accompaniment', 'linear_mixture'])
    
    loader = tqdm(sorted(os.listdir(src)))
    for idx, name in enumerate(loader):
        norm=None
        #for i in os.listdir(os.path.join(src, name)):
        folder_path=os.path.join(src, name)
        folder=os.listdir(os.path.join(src, name))
        #print(folder)
        for i in range(len(folder)):
            #print(os.path.join(folder_path,folder[i]))
            y, sr = librosa.load(os.path.join(folder_path,folder[i]))
            #print(y)
            
            stft = librosa.stft(y)
            spectrum, phase = librosa.magphase(stft)
            spectrogram = np.abs(spectrum).astype(np.float32)
            norm = spectrogram.max() if norm is None else norm
            spectrogram /= norm
            
            np.save(os.path.join(dest, dest_folder_list[i], str(idx) + '_' + name[:-10] + '_spec'), spectrogram)
            np.save(os.path.join(dest, dest_folder_list[i], str(idx) + '_' + name[:-10] + '_phase'), phase)
    
create_spec(MUSDB_PATH, DEST_PATH)

#SINGLE_SONG_SRC = SINGLE_SONG_SRC      (from config.py)
#SINGLE_SONG_DEST = SINGLE_SONG_DEST

def create_single_spec(src, dest):
    if not os.path.exists(dest):
        os.mkdir(dest)
        os.mkdir(os.path.join(dest, "mixture"))
        os.mkdir(os.path.join(dest, "vocals"))
        os.mkdir(os.path.join(dest, "drums"))
        os.mkdir(os.path.join(dest, "bass"))
        os.mkdir(os.path.join(dest, "other"))
        os.mkdir(os.path.join(dest, "accompaniment"))
        os.mkdir(os.path.join(dest, "linear_mixture"))
        
    dest_folder_list = sorted(['mixture', 'drums', 'bass', 'other', 'vocals', 'accompaniment', 'linear_mixture'])
        
    folder_name = SINGLE_SONG_SRC.split('\\')
    folder_name.reverse()
        
    loader = sorted(os.listdir(src))
    norm=None
    for idx, name in enumerate(loader):
        print(name)
        y, sr = librosa.load(os.path.join(src,name))
            
        stft = librosa.stft(y)
        spectrum, phase = librosa.magphase(stft)
        spectrogram = np.abs(spectrum).astype(np.float32)
        norm = spectrogram.max() if norm is None else norm
        spectrogram /= norm
            
        print(os.path.join(dest, dest_folder_list[idx], str(0) + '_' + folder_name[0][:-10] + '_spec'))
            
        np.save(os.path.join(dest, dest_folder_list[idx], str(0) + '_' + folder_name[0][:-10] + '_spec'), spectrogram)
        np.save(os.path.join(dest, dest_folder_list[idx], str(0) + '_' + folder_name[0][:-10] + '_phase'), phase)
            
create_single_spec(SINGLE_SONG_SRC, SINGLE_SONG_DEST)

dest_folder_list = sorted(['mixture', 'drums', 'bass', 'other', 'vocals', 'accompaniment', 'linear_mixture'])

#TEST_PATH = TEST_PATH (from config.py) 

def spec_to_wav(src, dest):
    loader = tqdm(sorted(os.listdir(src)))
    for audio_idx, spec_name in enumerate(loader):
        if 'spec' in spec_name:
            # load data
            #phase_name = spec_name[:-8] + 'phase.npy'
            phase_name = spec_name[:-19] + 'phase.npy'
            mag = np.load(os.path.join(src, spec_name))
            phase = np.load(os.path.join(src, phase_name))

            #idx = slice(*librosa.time_to_frames([30, 35], sr=22050))
#             plt.figure(figsize=(12, 8))
#             plt.subplot(3, 1, 1)
#             librosa.display.specshow(librosa.amplitude_to_db(mag, ref=np.max),
#                          y_axis='log', sr=22050)
            
            #full_audio = librosa.istft(mag)
            
            # resize as the same size
            length = min(phase.shape[-1], mag.shape[-1])
            mag = mag[:, :length]
            phase = phase[:, :length]

            # De-normalization
            spectrogram = mag*phase
            mix_spec = np.load(os.path.join(src, spec_name))
            spectrogram *= mix_spec.max()

            plt.figure(figsize=(12, 16))
            plt.subplot(3, 1, 1)
            librosa.display.specshow(librosa.amplitude_to_db(mix_spec, ref=np.max),
                         y_axis='log', sr=22050)
            
            # reconstruct the audio
            #y = librosa.istft(spectrogram, win_length=1024, hop_length=768)
            y = librosa.istft(spectrogram)
            
            def play_audio(y, sr, autoplay=False):
                display(Audio(y, rate=sr, autoplay=autoplay))
    
            play_audio(y,22050)
            #file_path = os.path.join(src, str(audio_idx) + '.wav')
            #librosa.output.write_wav(dest, int(44100), norm=True)
            #sf.write(file_path, y, 22050)

spec_to_wav(TEST_PATH, TEST_PATH)