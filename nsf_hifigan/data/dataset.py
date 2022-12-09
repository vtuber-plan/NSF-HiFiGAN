import os
import random
from typing import Optional

import torch
import torchaudio

from ..utils import load_filepaths, load_wav_to_torch

resamplers = {}

def load_audio(filename: str, sr: Optional[int] = None):
    global resamplers
    audio, sampling_rate = load_wav_to_torch(filename)

    if sr is not None and sampling_rate != sr:
        # not match, then resample
        if sr in resamplers:
            resampler = resamplers[(sampling_rate, sr)]
        else:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
            resamplers[(sampling_rate, sr)] = resampler
        audio = resampler(audio)
        sampling_rate = sr
        # raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
    return audio

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, audiopaths: str, hparams):
        self.audiopaths = load_filepaths(audiopaths)
        self.hparams = hparams
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.mel_fmin       = hparams.mel_fmin
        self.mel_fmax       = hparams.mel_fmax
        self.n_mel_channels = hparams.n_mel_channels

        self.resamplers = {}

        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_item(self, index: int):
        audio_path = self.audiopaths[index]
        
        audio_wav = load_audio(audio_path, sr=self.sampling_rate)
        audio_wav = audio_wav.unsqueeze(0)

        if audio_wav.shape[1] > self.sampling_rate * 20:
            l = self.sampling_rate * 10
            start = random.randrange(0, audio_wav.shape[1] - l)
            clip_audio_wav = audio_wav[:, start:start+l]
            audio_wav = clip_audio_wav

        return {
            "wav": audio_wav
        }

    def __getitem__(self, index):
        ret = self.get_item(index)
        return ret

    def __len__(self):
        return len(self.audiopaths)
