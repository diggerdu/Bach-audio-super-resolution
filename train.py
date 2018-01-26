import time
import torch.backends.cudnn as cudnn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import sys
import numpy as np

cudnn.benchmark = True

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
# visualizer = Visualizer(opt)

total_steps = 0

#TODO able to print error more easily
class ERROR:
    def __init__(self):
        self.G_GANs = list()
        self.G_L1s = list()
        self.D_reals = list()
        self.D_fakes = list()

    def update(G_GAN, G_L1, D_real, D_fake):
        self.G_GANs.append(G_GAN)
        self.G_L1s.append(G_L1)
        self.D_reals.append(D_reals)
        self.D_fakes.append(D_fakes)

    def get_current_errors(self):
        for attr, value in self.__dict__.items():
            __import__('pdb').set_trace()
            if isinstance(attr, list):
                print('current error is:')
                print(attr, ':', value, end=' ')

    def __repr__(self):
        for attr, value in self.__dict__.items():
            print('current error is:')
            print(attr, ':', np.mean(value), end=' ')

embark_time = time.time()
for epoch in range(1, opt.niter + 1):
    epoch_start_time = time.time()
    error_G_GAN = list()
    error_G_L1 = list()
    error_Dreal = list()
    error_Dfake = list()
    for i, data in enumerate(dataset):
        total_steps += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()
        loss_G_GAN = model.get_current_errors()['G_GAN']
        loss_G_L1 = model.get_current_errors()['G_L1']
        loss_D_real = model.get_current_errors()['D_real']
        loss_D_fake = model.get_current_errors()['D_fake']

#        errorEpoch.append()
        print('current loss:')
        print('loss_G_GAN: {}, loss_G_L1: {}'.format(loss_G_GAN, loss_G_L1))
        print('loss_D_real: {}, loss_D_fake: {}'.format(loss_D_real, loss_D_fake))
        if time.time() - embark_time > 60 * 15:
            model.save('latest')
    t = time.time() - epoch_start_time
#    print('==============================================================================')
#    print('epoch ', epoch, ', current error is ', np.mean(errorEpoch), ' cost time is ', t)
#    print('==============================================================================')
    embark_time = time.time()

