# Speech Separation (Final Year Thesis)

Thesis project for Speech Separation using Deep Learning

## Installation & Dataset Setup

Installing Dependencies

```
pip install -r requirements.txt
```

Setting up MUSDB18 for training (optional)

Convert from STEMS format to .wav format
```
musdbconvert path/to/musdb-stems-root path/to/new/musdb-wav-root
```

Download LibriSpeech Corpus for creating Synthetic mixtures from https://www.openslr.org/12

## Creating Synthetic Audio for Training our Model

Each entry in Librispeech Corpus refers to a speaker, and each speaker folder contains multiple recordings with annotations included. We can use this individual speaker audio from these folders and overlap them using [pydub](https://github.com/jiaaro/pydub) to create synthetic audio mixtures and use them to train our model.

## Synthetic Audio Data Format:

```
+ data
    |
    + spk1_spk2
    |      |
    |      + sound1.wav
    |      + sound2.wav
    |      + mixed.wav
    + spk1_spk3
    |      |
    |      + sound1.wav
    |      + sound2.wav
    |      + mixed.wav
```
