import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import librosa
import random
import torch
import os

from tqdm import tqdm
from matplotlib import pyplot as plt
from config import *

model = UNet()
model.load("result.pth")   ### Pre-trained weights

model.eval()

# Seperate!
with torch.no_grad():
    bar = tqdm([_ for _ in sorted(os.listdir(TEST_PATH)) if 'spec' in _])
    for idx, name in enumerate(bar):
        if idx > 5:
            break
        mix = np.load(os.path.join(TEST_PATH, name))
        print(mix)
        spec_sum = None
        for i in range(mix.shape[-1] // 128):
            # Get the fixed size of segment
            seg = mix[1:, i * 128:i * 128 + 128, np.newaxis]
            seg = np.asarray(seg, dtype=np.float32)
            seg = torch.from_numpy(seg).permute(2, 0, 1)
            seg = torch.unsqueeze(seg, 0)
            #seg = seg.cuda()

            # generate mask
            msk = model(seg)

            print(msk)
            # split the voice
            vocal_ = seg * (1 - msk)
            #vocal_ = seg * msk

            # accumulate the segment until the whole song is finished
            vocal_ = vocal_.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, 0]
            vocal_ = np.vstack((np.zeros((128)), vocal_))
            spec_sum = vocal_ if spec_sum is None else np.concatenate((spec_sum, vocal_), -1)
        np.save(os.path.join(TEST_PATH, str(idx) + '_' + name[:-4] + '_pred_vocal'), spec_sum)