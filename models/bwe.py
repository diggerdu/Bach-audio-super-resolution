import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from scipy import interpolate
from collections import OrderedDict

n_filters = [  128,  256,  512, 512, 512, 512, 512, 512 ]
n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]

class SubPixel1d(nn.Module):
    def __init__(self, scale):
        super(SubPixel1d, self).__init__()
        self.scale = scale
    
    def forward(self, I):
        assert(I.data.shape[1] % self.scale == 0)
        n_b = I.data.shape[0]
        n_c = I.data.shape[1] / self.scale
        n_l = I.data.shape[2] * self.scale
        transformed = I.permute(0, 2, 1)
        transformed = transformed.contiguous().view(-1)
        transformed = torch.stack(torch.chunk(transformed, n_b), dim=0)[:, :]
        transformed = torch.stack(torch.chunk(transformed, n_l, dim=1), dim=1)
        transformed = transformed.permute(0, 2, 1)

        return transformed

    def __repr__(self):
        return 'Subpixel1d({})'.format(self.scale)


class AudioUnet(nn.Module):
    def __init__(self, opt):
        super(AudioUnet, self).__init__()
        self.scale = opt.scale
        self.layers = opt.layers

        self.downsampling = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.iter = iter(zip(n_filters, n_filtersizes))


        #downsamplinglayers
        nf_buff = 1
        for l in range(self.layers):
            nf, fs = next(self.iter)
            x = nn.Sequential(OrderedDict([
                ('conv_orth', nn.Conv1d(in_channels=nf_buff, out_channels=nf, 
                    kernel_size=fs, stride=2, padding=(fs-1)//2)),
                ('leakyrelu', nn.LeakyReLU(0.2, inplace=True))
                ]))
            self.downsampling.extend([x])        
            nf_buff = nf

        #bottleneck
        self.bottleneck = nn.Sequential(OrderedDict([
            ( 'conv_orth', nn.Conv1d(in_channels=n_filters[-1], out_channels=n_filters[-1], 
                kernel_size=n_filtersizes[-1], stride=2, 
                padding=(n_filtersizes[-1]-1)//2) ),
            ( 'dropout', nn.Dropout(0.5)),
            ( 'leakyrelu', nn.LeakyReLU(0.2, inplace=True) )
            ]))
        
        #upsampling layers
        nf_buff = n_filters[self.layers]
        for l in range(self.layers):
            nf, fs = next(self.iter)
            x = nn.Sequential(OrderedDict([
                ('conv_orth', nn.Conv1d(in_channels=nf_buff, 
                    out_channels=2*nf, kernel_size=fs, padding=(fs-1)//2)),
                ('dropout', nn.Dropout(0.5)),
                ('relu', nn.ReLU(True)),
                ('subpixel', SubPixel1d(2))
                ]))
          
            self.upsampling.extend([x])
            nf_buff = n_filters[self.layers-1-l] + nf 
        
        #last convolution
        self.lastconv = nn.Sequential(OrderedDict([
            ( 'conv_norm', nn.Conv1d(in_channels=n_filters[-1]+n_filters[0], 
                                out_channels=2, kernel_size=9, padding=4) ),
            ( 'subpixel', SubPixel1d(2) ),
            ]))

    def forward(self, sample):
        #downsampling
        sample = torch.unsqueeze(sample, 1)
        downsampling_l = []
        for index, model in enumerate(self.downsampling):
            if index == 0:
                pred = model(sample)
                downsampling_l.append(pred)
            else:
                pred = model(pred)
                downsampling_l.append(pred)
        
        #bottleneck
        pred = self.bottleneck(pred)

        #upsampling
        for index, u_model in enumerate(self.upsampling):
            pred = torch.cat((u_model(pred), downsampling_l[-index-1]), 1)

        #last convolution
        pred = self.lastconv(pred)
        pred = torch.squeeze(pred, 1)
        #fancy interpolation
#        pred = torch.cat((sample.permute(0, 2, 1), pred.permute(0, 2, 1)), 1)
#        pred = pred.contiguous().view(pred.data.shape[0], -1)

        return pred



if __name__ == "__main__":
# #   I = torch.Tensor([[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12], [13, 14], [15, 16]]])
# #   I = I.permute(0, 2, 1)
# #   I = Variable(I)
# #   Subpixel = SubPixel1d(2)
# #   print('origin\n', I.data)
# #   print('transformed\n', Subpixel(I))\
    class OPT:
        def __init__(self):
            self.scale = 2
            self.layers = 4

    opt = OPT()
    model = AudioUnet(opt).cuda()
    print(model)
    for name, para in model.named_parameters():
        print(name)
        print(para.__class__.__name__)
    first_t = torch.Tensor(5, 1, 16000).cuda()
    first_t = Variable(first_t)
    print('result', model(first_t))
