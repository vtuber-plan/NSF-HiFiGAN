import os
import random
from typing import Optional

import torch
import torchaudio

import numpy as np
import librosa
from librosa import pyin

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

def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch

def estimate_pitch(audio: np.ndarray, sr: int, n_fft: int, win_length: int, hop_length: int,
                    method='pyin', normalize_mean=None, normalize_std=None, n_formants=1):
    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':
        snd, sr = audio, sr
        pad_size = int((n_fft-hop_length)/2)
        snd = np.pad(snd, (pad_size, pad_size), mode='reflect')

        pitch_mel, voiced_flag, voiced_probs = pyin(
            snd,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=win_length,
            hop_length=hop_length,
            center=False,
            pad_mode='reflect')
        # assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        # pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError
    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel

def get_pitch(audio: str, sr: int, filter_length: int, win_length: int, hop_length: int):
    pitch_mel = estimate_pitch(
        audio=audio, sr=sr, n_fft=filter_length,
        win_length=win_length, hop_length=hop_length, method='pyin',
        normalize_mean=None, normalize_std=None, n_formants=1)

    return pitch_mel

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

        audio_pitch = get_pitch(
            audio_wav.numpy(),
            self.sampling_rate,
            self.hparams.filter_length,
            self.hparams.win_length,
            self.hparams.hop_length
        )

        return {
            "wav": audio_wav.unsqueeze(0),
            "pitch": audio_pitch,
        }

    def __getitem__(self, index):
        ret = self.get_item(index)
        return ret

    def __len__(self):
        return len(self.audiopaths)
