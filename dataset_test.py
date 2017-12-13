import torch
import numpy as np
from data.audio_dataset import AudioDataset


if __name__ == "__main__":
    class OPT:
        def __init__(self):
            self.PathClean = '/home/alan/Documents/once'
            self.snr = 0
            self.scale = 2
            self.SR = 16000
            self.nfft = 1024
            self.hop = 512
            self.len = 16000

    opt = OPT()
    dataset = AudioDataset()
    dataset.initialize(opt)
    print('A', dataset[0]['A'], dataset[10]['A'].shape)
    print('B', dataset[2]['B'], dataset[10]['B'].shape)
