import torch
import torchaudio
import torchaudio.transforms as T

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import random

import numpy as np

from ..mel_processing import hann_window

class GaussianNoise(torch.nn.Module):
    def __init__(self, min_snr=0.0001, max_snr=0.01):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio):
        std = torch.std(audio)
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)

        norm_dist = torch.distributions.normal.Normal(0.0, noise_std)
        noise = norm_dist.rsample(audio.shape).type(audio.dtype).to(audio.device)

        return audio + noise

class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        freq=16000,
        n_fft=1024,
        n_mel=128,
        win_length=1024,
        hop_length=256
    ):
        super().__init__()

        self.freq=freq

        pad = int((n_fft-hop_length)/2)
        self.spec = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, power=None,center=False, pad_mode='reflect', normalized=False, onesided=True)

        # self.strech = T.TimeStretch(hop_length=hop_length, n_freq=freq)
        self.spec_aug = torch.nn.Sequential(
            GaussianNoise(min_snr=0.0001, max_snr=0.02),
            T.FrequencyMasking(freq_mask_param=80),
            # T.TimeMasking(time_mask_param=80),
        )

        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor, aug: bool=False) -> torch.Tensor:
        shift_waveform = waveform
        # Convert to power spectrogram
        spec = self.spec(shift_waveform)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        # Apply SpecAugment
        if aug:
            spec = self.spec_aug(spec)
        # Convert to mel-scale
        mel = self.mel_scale(spec)
        return mel