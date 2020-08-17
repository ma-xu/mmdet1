import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        print("norm_x is {}".format(norm_x))
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        print("ex is {}".format(ex))
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        print("ew is {}".format(ew))
        return torch.mm(self.scale * ex, ew.t())
        # return torch.tensor()

# def create_model(in_dims=512, out_dims=1000):
#     print('Loading Cosine Norm Classifier.')
#     return CosNorm_Classifier(in_dims=in_dims, out_dims=out_dims)