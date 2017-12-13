###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import os
import os.path
import librosa
import soundfile as sf
import numpy as np

AUDIO_EXTENSIONS = [
    '.wav',
    '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_dataset(dir, opt):
    audios = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_audio_file(fname):
                path = os.path.join(root, fname)
                print(path)
                try:
                    wav, sr = sf.read(path, dtype='float32')
                except:
                    continue
                try:
                    assert sr == opt.SR
                except AssertionError:
                    wav = librosa.resample(wav, sr, opt.SR)

                wav = wav - np.mean(wav)
                wav = wav / np.max(np.abs(wav))
                # sf.write(path, wav, opt.SR)

                audios.append(wav.astype(np.float32))
    return audios
