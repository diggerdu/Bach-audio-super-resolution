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
opt.dictSize = opt.nFrames * dataset_size 
# visualizer = Visualizer(opt)

total_steps = 0

embark_time = time.time()
nData = 0 
for i, data in enumerate(dataset):
    total_steps += opt.batchSize
    model.set_input(data)
    model.optimize_parameters()
    nData = nData + data['A'].shape[0] 
    print('{} samples completed!'.format(nData))
    model.save('latest')

