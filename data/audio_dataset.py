import os.path
from data.base_dataset import BaseDataset
from data.audio_folder import make_dataset
import librosa
import soundfile as sf
import numpy as np
import random
from models import time_frequence as tf
from scipy.signal import decimate

class AudioDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.DirClean = opt.PathClean
        self.snr = opt.snr

        self.Clean = make_dataset(self.DirClean, opt)
        self.Noise = []

        self.SR = opt.SR
        self.hop = opt.hop
        self.nfft = opt.nfft
        self.scale = opt.scale

    def __getitem__(self, index):
        CleanData = self.Clean[index % len(self.Clean)]

        CleanAudio = self.load_audio(CleanData)
        A = decimate(CleanAudio, self.scale).astype(np.float32)
#        A = tf.spline_up(A, self.scale).astype(np.float32)

        assert A.dtype==np.float32 and CleanAudio.dtype==np.float32

        return {
                'A': A,
                'B': CleanAudio,
        }

    def __len__(self):
        # return len(self.FilesClean)
        # return 64
        return max(len(self.Clean), len(self.Noise))

    def addnoise(self, clean, noise):
        # print(clean.dtype, noise.dtype)
        assert clean.shape == noise.shape
        noiseAmp = np.mean(np.square(clean)) / np.power(10, self.snr / 10.0)
        scale = np.sqrt(noiseAmp / np.clip(np.mean(np.square(noise)), a_min=1e-7, a_max=1e8))
        return clean + scale * noise

    def name(self):
        return "AudioDataset"

    def load_audio(self, data):

        target_len = self.opt.len
        if data.shape[0] >= target_len:
            head = random.randint(0, data.shape[0] - target_len)
            data = data[head:head + target_len]
        if data.shape[0] < target_len:
            ExtraLen = target_len - data.shape[0]
            PrevExtraLen = np.random.randint(ExtraLen)
            PostExtraLen = ExtraLen - PrevExtraLen
            PrevExtra = np.zeros((PrevExtraLen, ), dtype=np.float32)
            PostExtra = np.zeros((PostExtraLen, ), dtype=np.float32)
            data = np.concatenate((PrevExtra, data, PostExtra))

        data = data - np.mean(data)
        assert data.dtype == np.float32
        assert data.shape[0] == self.opt.len
        return data
