import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import time_frequence as tf


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.gan_loss = opt.gan_loss
        self.isTrain = opt.isTrain
        self.scale = opt.scale

        # define tensors self.Tensor has been reloaded
        self.input_A = self.Tensor(opt.batchSize, opt.len).cuda(device=self.gpu_ids[0])
        self.input_B = self.Tensor(opt.batchSize, opt.nfft).cuda(device=self.gpu_ids[0])
        # load/define networks
        self.netG = networks.define_G(opt)
        self.stft = tf.stft().cuda()

        if opt.specLoss:
            self.specModel = tf.Spectrogram().cuda()

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
            #                              opt.which_model_netD,
            #                              opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            # if self.isTrain:
            #    self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterion = torch.nn.L1Loss()

            # initialize optimizers

            self.TrainableParam = list()
            param = self.netG.named_parameters()
            IgnoredParam = [id(P) for name, P in param if 'stft' in name]

            if self.opt.optimizer == 'Adam':
                self.optimizer_G = torch.optim.Adam(
                    filter(lambda P: id(P) not in IgnoredParam,
                       self.netG.parameters()),
                        lr=opt.lr,
                        betas=(opt.beta1, 0.999))

            if self.opt.optimizer == 'sgd':
                self.optimizer_G = torch.optim.SGD(
                    filter(lambda P: id(P) not in IgnoredParam,
                       self.netG.parameters()),
                        lr=opt.lr)
    
            if self.opt.optimizer == 'lbfgs':
                self.optimizer_G = torch.optim.LBFGS(
                        filter(lambda P:id(P) not in IgnoredParam, self.netG.parameters()),
                        lr=opt.lr,
                        history=50,
                        )
                def closure():
                    self.optimizer_G.zero_grad()
                    self.forward()
                    self.loss_G = self.criterion(self.fakeB, self.realB)
                    self.loss_G.backward()

                    return self.loss_G
                self.closure = closure



            print('---------- Networks initialized ---------------')
            networks.print_network(self.netG)
            # networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        input_B = input['B']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        self.image_paths = 'NOTIMPLEMENT'
    

    def spec(self, signal):
        mag, phase, ac = self.stft(signal)    
        spec = torch.rsqrt((torch.pow(mag * torch.cos(phase), 2) + 
            torch.pow(mag * torch.sin(phase), 2)))

        return spec

    def forward(self):
        self.real_A = Variable(self.input_A)
        output = self.netG.forward(self.real_A)
        self.fakeB = output['time']
        self.realB = Variable(self.input_B)

        self.realB = self.spec(self.realB)
        self.fakeB = self.spec(self.fakeB)
        self.realB.detach_()

        if self.opt.specLoss:
            self.fakeBSpec = output['spec']
            self.realBSpec = Variable(self.specModel(self.realB).data, requires_grad=False)


    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fakeB = self.netG.forward(self.real_A)
        self.realB = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fakeB
        fake_AB = self.fake_AB_pool.query(
            torch.cat((self.real_A, self.fakeB), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.realB), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # self.loss_G = self.criterionL1(self.fakeB, Variable(torch.cuda.FloatTensor(self.fakeB.size()).zero_()))
        nfft = self.opt.nfft
        if self.opt.wav_db:
            self.fakeB = tf.amp2db(self.fakeB)
            self.realB = tf.amp2db(self.realB)
        
        self.loss_G = self.criterion(self.fakeB, self.realB)
        if self.scale is 0:
            self.loss_G = self.criterion(self.fakeB[:, nfft:-nfft], self.realB[:, nfft:-nfft])
        

        if self.opt.specLoss:
            start = self.opt.specLossStart
            self.loss_GSpec = self.criterion(self.fakeBSpec[:,:,start:,:], self.realBSpec[:,:,start:,:])

            self.loss_G = self.loss_G + self.opt.specLossRatio * self.loss_GSpec
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        if self.gan_loss:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        
         
        if self.opt.optimizer == 'lbfgs':
            self.optimizer_G.step(self.closure)
        else:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

    def get_current_errors(self):
        if self.gan_loss:
            return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                                ('G_L1', self.loss_G_L1.data[0]),
                                ('D_real', self.loss_D_real.data[0]),
                                ('D_fake', self.loss_D_fake.data[0])])
        else:
            # print("#############clean sample mean#########")
            # sample_data = self.input_B.cpu().numpy()

            # print("max value", np.max(sample_data))
            # print("mean value", np.mean(np.abs(sample_data)))
            return OrderedDict([('G_LOSS', self.loss_G.data[0])])
            # return self.loss_G.data[0]

    def get_current_visuals(self):
        real_A = self.real_A.data.cpu().numpy()
        fakeB = self.fakeB.data.cpu().numpy()
        realB = self.realB.data.cpu().numpy()
        clean = self.clean.cpu().numpy()
        noise = self.noise.cpu().numpy()
        return OrderedDict([
            ('est_ratio', fakeB),
            ('clean', clean),
            ('ratio', realB),
            ('noise', noise),
        ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.gan_loss:
            self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        # lrd = self.opt.lr / self.opt.niter_decay
        # lr = self.old_lr - lrd
        lr = self.old_lr * 0.6
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
