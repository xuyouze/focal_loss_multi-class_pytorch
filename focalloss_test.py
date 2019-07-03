from __future__ import print_function

import torch
from torch.autograd import Variable

import os, sys, random, time
import argparse

from focal_loss import *

class_num = 10
batch_size = 100

# for example , N = 12800, C = 10
x = torch.rand(batch_size, class_num)
x = Variable(x.cuda())
# the label = 0 or 1
label = torch.randint(0, 2, (batch_size, class_num)).float()
label = Variable(label.cuda())

output0 = FocalLoss(gamma=2)(x, label)
