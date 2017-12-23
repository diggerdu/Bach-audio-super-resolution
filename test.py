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

def eval(model, cleanPath, noisePath, opt):
    leng = opt.nfft + (opt.nFrames - 1) * opt.hop
    clean = loadAudio(cleanPath, opt.SR)
    assert clean.shape[0] > leng

#    clean = clean[:leng]
    clean = [x for x in range(leng)]
    target = clean
#    degraded = decimate(clean, 2, zero_phase=0)
#    interpolated = spline_up(degraded, 2)
    degraded = []
    for index, point in enumerate(clean):
        if index % 2 == 0:
            degraded.append(point)

    print(opt.nfft, opt.nFrames, opt.hop)

#    input_ = {'A' : torch.from_numpy(interpolated[None, None, :]).float()}
    input_ = {'A': torch.from_numpy(degraded[None, :]).float()}
    model.set_input(input_)
    model.test()
    res = model.get_current_visuals().data.cpu().numpy().flatten()
    
    output = []
    for index, point1, point2 in zip(input_, res):
        __import__('pdb').set_trace()
        output.append(point1)
        output.append(point2)
    output = np.array(output)
    __import__('pdb').set_trace()
    print(CalSNR(target, output), 'dB')

    sf.write('clean.wav', target, opt.SR)
    sf.write('enhanced.wav', output, opt.SR)
    sf.write('degraded.wav', degraded, opt.SR//2)
#    sf.write('interpolated.wav', interpolated, opt.SR)


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.model = "test"

model = create_model(opt)

# test

cleanPath = "/home/alan/Large/clean/p225/p225_001.wav"
# cleanPath = "/home/diggerdu/dataset/VCTK-Corpus/wav48/p315/p315_013.wav"
noisePath = "/home/alan/Large/babble/59899-robinhood76-00309crowd2-116.wav"
#noisePath = "/home/diggerdu/dataset/speech/14.wav"
#cleanPath = "/home/diggerdu/dataset/speech/14.wav"

eval(model, cleanPath, noisePath, opt)

