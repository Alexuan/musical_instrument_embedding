import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleLoss(nn.Module):
    # source: https://github.com/clcarwin/sphereface_pytorch
    # AngleLoss class
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.bool().detach()
        #index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.01*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp().detach()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        loss = F.cross_entropy(input, target, weight,
            reduction=self.reduction)
        return loss


class NLLLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        loss = F.nll_loss(input, target, weight,
            reduction=self.reduction)
        return loss
