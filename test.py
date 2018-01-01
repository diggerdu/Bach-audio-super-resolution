import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.audio_process import recovery_phase
import torch
import torch.nn as nn
from torch.autograd import Variable

from pdb import set_trace as st
from util import html



# audio process
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import decimate
from models.time_frequence import spline_up


def CalSNR(ref, sig):
    ref_p = np.mean(np.square(ref))
    noi_p = np.mean(np.square(sig - ref))
    return 10 * (np.log10(ref_p) - np.log10(noi_p))



def loadAudio(path, SR):
    data, sr = sf.read(path)
    print(path, data.shape, sr)
    try:
        assert sr == SR
    except AssertionError:
        data = librosa.resample(data, sr, opt.SR)

    data = data / np.max(np.abs(data))
    return data - np.mean(data)

def eval(model, opt):
    cleanPath = opt.PathClean
    assert os.path.isfile(cleanPath)
    leng = opt.nfft + (opt.nFrames - 1) * opt.hop
    clean = loadAudio(cleanPath, opt.SR)
    assert clean.shape[0] > leng

    clean = clean[:leng]
    degraded = decimate(clean, 2, zero_phase=0)
    interpolated = spline_up(degraded, 2)

    print(opt.nfft, opt.nFrames, opt.hop)

    # cal spectrogram and mfcc
    mfcc = librosa.feature.mfcc(
            interpolated, opt.SR, n_mfcc=opt.nmfcc, n_fft=opt.nfft, hop_length=opt.hop)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta)
    feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), 0)
    
    phase = np.angle(librosa.core.stft(interpolated, opt.nfft, opt.hop))

#    input_ = {'A' : torch.from_numpy(interpolated[None, None, :]).float()}
    input_ = {'A': torch.from_numpy(feature).float()}
    model.set_input(input_)
    model.test()
    
    amp = model.get_current_visuals().data.cpu().numpy()
    spec = amp * np.cos(phase) + amp * np.sin(phase) * 1j
    output = librosa.core.istft(spec, hop_length=opt.nfft//2)
    
    print(CalSNR(clean, output), 'dB')

    sf.write('clean.wav', clean, opt.SR)
    sf.write('enhanced.wav', output, opt.SR)
    sf.write('degraded.wav', degraded, opt.SR//2)
    sf.write('interpolated.wav', interpolated, opt.SR)


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.model = "test"

model = create_model(opt)

# test

#cleanPath = "/home/alan/Large/clean/p225/p225_001.wav"
# cleanPath = "/home/diggerdu/dataset/VCTK-Corpus/wav48/p315/p315_013.wav"
#noisePath = "/home/alan/Large/babble/59899-robinhood76-00309crowd2-116.wav"
#noisePath = "/home/diggerdu/dataset/speech/14.wav"
#cleanPath = "/home/diggerdu/dataset/speech/14.wav"

eval(model, opt)

