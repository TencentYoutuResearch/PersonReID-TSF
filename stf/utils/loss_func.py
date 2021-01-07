from __future__ import absolute_import
import sys

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import random
import math
import torch.nn.functional as F

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""


class LSR(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LSR, self).__init__()
        self.epsilon = epsilon
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, inputs, targets):
        num_class = inputs.size()[1]
        targets = self._class_to_one_hot(targets.data.cpu(), num_class)
        targets = Variable(targets.cuda())
        #print (targets, 'lsr')
        outputs = self.log_softmax(inputs)
        loss = - (targets * outputs)
        #print (loss.size(), 'lsr')
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        #print (loss, 'lsr')
        return loss

    def _class_to_one_hot(self, targets, num_class):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], num_class)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets, 1 - self.epsilon)
        targets_onehot.add_(self.epsilon / num_class)
        return targets_onehot


if __name__ == '__main__':
    pass
