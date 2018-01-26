import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import Module
import numpy as np
from collections import OrderedDict
#from . import densenet_efficient as dens
from . import time_frequence as tf
from . import bwe 
from . import segan
import torch.nn.init as init
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find(
            'InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_v2(m):
    classname = m.__class__.__name__


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer
    # return None


def define_G(opt):
    netG = None
    use_gpu = len(opt.gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    netG = AuFCNWrapper(opt)

    if len(opt.gpu_ids) > 0:
        netG.cuda(device=opt.gpu_ids[0])
    netG.weights_init()
    return netG


def define_D(opt):
    netD = None
    use_gpu = len(opt.gpu_ids) > 0
#    norm_layer = get_norm_layer(norm_type=norm)
    which_model_netD = opt.which_model_netD
    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'seganDiscriminator':
        netD = segan.SeganDiscriminator(opt)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device=opt.gpu_ids[0])
    netD.weights_init()
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self,
                 use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input_, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input_.numel()))
            if create_label:
                real_tensor = self.Tensor(input_.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input_.numel()))
            if create_label:
                fake_tensor = self.Tensor(input_.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input_, target_is_real):
        target_tensor = self.get_target_tensor(input_, target_is_real)
        return self.loss(input_, target_tensor)


class AuFCNWrapper(nn.Module):
    def __init__(self, opt):
        super(AuFCNWrapper, self).__init__()
        self.gpu_ids = opt.gpu_ids
        self.model = AuFCN(opt)
        try:
            assert opt.scale == 2
        except Exception:
            print('scale!=2 not supported')

    def forward(self, input_):
        if self.gpu_ids and isinstance(input_.data,
                                       torch.cuda.FloatTensor) and False:
            output = nn.parallel.data_parallel(self.model, input_, self.gpu_ids)
            print("network G output", output.size())
            return output
        else :
            return self.model(input_)

    def weights_init(self):
        for name, para in self.named_parameters():
            assert 'weight' in name or 'bias' in name
            if 'weight' in name:
                if 'orth' in name:
                    init.orthogonal(para.data)
                else:
                    init.normal(para.data, mean=0, std=1e-3)
            elif 'bias' in name:
                init.constant(para, 0)
    

# TODO robust
# TODO requires gradient
# TODO assert AC == 0
class AuFCN(nn.Module):
    def __init__(self, opt):
        super(AuFCN, self).__init__()
#        self.stft_model = tf.stft(opt.nfft, opt.hop)
#        self.istft_model = tf.istft(opt.nfft, opt.hop)
        self.AudioUnet = bwe.AudioUnet(opt)

    def forward(self, sample):
        pred = self.AudioUnet(sample)
        return {'time':pred}


class Tanh_rescale(Module):
    def forward(self, input_):
        return torch.div(
            torch.add(torch.tanh(torch.mul(input_, 2.0)), 1.0), 2.0)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
