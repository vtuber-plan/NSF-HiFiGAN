
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

def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)

MATPLOTLIB_FLAG = False
def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data