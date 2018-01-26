import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class VBN(nn.Module):
    def __init__(self, ref_batch, name, epsilon=1e-5):
        super(VBN, self).__init__()
        self.epsilon = epsilon
        self.ref_mean = 

    def initialize(self, ref_batch):
        ref_mean = torch.mean(torch.mean(ref_batch, dim=0, keep_dims=True), dim=1, keep_dims=True)
        ref_mean_sq = torch.mean(torch.mean(torch.square(ref_batch), dim=0,keep_dims=True), 
                                    dim=1, keep_dims=True)
        ref_output = self._normalize(x, self.mean, self.mean_sq)

    def forward(self, input_):

    def _normalize(self, x, mean, mean_sq):
        pass

if __name__ == "__main__":
    test_data = np.random.normal(0, 1, (20, 100))
    test_data = torch.from_numpy(test_data.float()).cuda()
    test_data = Variable(test_data)
    model = VBN().cuda()
    print(model(test_data))
    model.backward()


