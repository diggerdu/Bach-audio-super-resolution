import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import scipy.stats


#TODO weight init
class SeganDiscriminator(nn.Module):
    def __init__(self, opt):
        super(SeganDiscriminator, self).__init__()
        self.gpu_ids = opt.gpu_ids
        self.fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.model_list = nn.ModuleList()
        self.model_list.append(DiscBlock(31, 2, self.fmaps[0]))
        for index in range(1, len(self.fmaps), 1):
            self.model_list.append(DiscBlock(31, self.fmaps[index-1], self.fmaps[index]))
#        self.model_list.append(GaussianLayers(opt.disc_noise_std))
        self.model_list.append(nn.Conv1d(self.fmaps[index], 1, 1))
        self.model_list.append(Squeeze())
        self.model_list.append(nn.Linear(8, 1))


    def forward(self, x):
        for index, model in enumerate(self.model_list):
            x = model(x)

        return x

    def weights_init(self):
        for name, para in self.named_parameters():
            if 'weight' in name:
                shape = tuple(para.shape)
                data = scipy.stats.truncnorm.rvs(-0.04, 0.04, loc=0, scale=0.02, size=shape)
                data = torch.from_numpy(data.astype(np.float32))
                para.data.copy_(data)
            elif 'bias' in name:
                init.constant(para, 0)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, input_):
        return torch.squeeze(input_)

class DiscBlock(nn.Module):
    def __init__(self, kwidth, in_nfmaps, out_nfmaps):
        super(DiscBlock, self).__init__()
        self.model_list = nn.ModuleList()
        self.model_list.append(DownConv(in_nfmaps, out_nfmaps, (kwidth, 1), ((kwidth-1)//2, 0)))

    def forward(self, x):
        for index, model in enumerate(self.model_list):
            unsqueezed_x = torch.unsqueeze(x, 3) 
            output = model(unsqueezed_x)
            x = torch.squeeze(output, 3)
        return x

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DownConv, self).__init__()
        self.model_list = nn.ModuleList()
        self.model_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, (2, 1), padding,))
        self.model_list.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

    def forward(self, x):
        for index, model in enumerate(self.model_list):
            x = model(x)
        return x


class GaussianLayer(nn.Module):
    def __init__(self, std):
        super(GaussianLayer, self).__init__()
        self.std = std

    def forward(self, input_):
        noise = Variable(torch.Tensor(input_.shape).normal_(0, self.std))

        return input_ + noise
