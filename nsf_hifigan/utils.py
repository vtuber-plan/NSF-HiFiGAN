
import logging
import sys
import torch
import torchaudio
from typing import Any, Dict, List, Tuple


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging

def load_filepaths(filename: str) -> List[List[str]]:
    with open(filename, encoding='utf-8') as f:
        filepaths = [line.rstrip() for line in f]
    return filepaths

def load_wav_to_torch(full_path: str) -> Tuple[torch.FloatTensor, int]:
    data, sampling_rate = torchaudio.load(full_path)
    if len(data.shape) >= 2:
        data = torch.mean(data, dim=0)
    return data, sampling_rate
