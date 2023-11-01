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

## Setting up LibriMix for training

LibriMix is an open source dataset for source separation in noisy environments. It is derived from LibriSpeech signals (clean subset) and WHAM noise. It offers a free alternative to the WHAM dataset and complements it. It will also enable cross-dataset experiments.

[Generating LibriMix](https://github.com/JorisCos/LibriMix#readme)

## Features

In LibriMix you can choose :

- The number of sources in the mixtures.
- The sample rate of the dataset from 16 KHz to any frequency below.
- The mode of mixtures : min (the mixture ends when the shortest source ends) or max (the mixtures ends with the longest source)
- The type of mixture : mix_clean (utterances only) mix_both (utterances + noise) mix_single (1 utterance + noise)

By default, LibriMix will be generated for 2 and 3 speakers, at both 16Khz and 8kHz, for min max modes, and all mixture types will be saved (mix_clean, mix_both and mix_single). This represents around 430GB of data for Libri2Mix and 332GB for Libri3Mix. Alternatively if you want to generate a smaller subset you can look at the options below:

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

## Using MiniLibriMix

MiniLibriMix is a small version of LibriMix. 

It was made for demonstration purposes. 

It contains a train set of 800 mixtures and a validation set of 200 mixtures.

In each set, you will find :

- mix_clean a folder containing clean mixtures of 2 speakers.
- mix_both a folder containing clean mixtures of 2 speakers and a noise.
- s1, s2, noise three folders containing the raw signals in the mixture.
