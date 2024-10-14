import numpy as np
import torch
import torch.optim as optim
from utils import *
import torch.nn as nn
from abc import ABC


class bnnr(nn.Module, ABC):
    def __init__(self, mic_feature, dis_feature, res, maxiter, alpha, beta, layer_size, device):
        super(bnnr, self).__init__()
        self.maxiter = maxiter
        self.mic = mic_feature
        self.dis = dis_feature
        self.res = res
        self.alpha = alpha
        self.beta = beta
        self.tol1 = 1e-1
        self.tol2 = 1e-2
        self.a = 0.0
        self.b = 1.0
        self.device = device
        self.bnnr = BNNR(self.mic, self.res, self.dis, self.alpha, self.beta, self.tol1, self.tol2, self.maxiter, self.a, self.b)
        self.bnnr1 = BNNR(self.mic, self.bnnr, self.dis, self.alpha, self.beta, self.tol1, self.tol2, self.maxiter, self.a, self.b)
        self.bnnr2 = BNNR(self.mic, self.bnnr1, self.dis, self.alpha, self.beta, self.tol1, self.tol2, self.maxiter, self.a, self.b)

    def forward(self):
        out = self.bnnr2
        return out
