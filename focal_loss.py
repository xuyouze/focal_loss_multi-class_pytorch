# coding:utf-8
# @Time         : 2019/7/2 
# @Author       : xuyouze
# @File Name    : focal_loss.py

import torch
from torch import nn
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        """

        :param input: shape N * C
        :param target: shape N * C
        :return: loss
        """
        if isinstance(self.alpha, (float, int)):
            self.alpha = (self.alpha * torch.ones((target.size(1), 2))).cuda()
        if isinstance(self.alpha, list):
            self.alpha = torch.stack((torch.tensor(self.alpha), 1 - torch.tensor(self.alpha)), dim=1).cuda()
        pt = Variable(torch.sigmoid(input)).cuda()
        # loss = nn.BCELoss(reduction="none")(pt, target)
        loss = nn.BCEWithLogitsLoss(reduction="none")(input, target)
        loss = self.alpha[:, 0] * target * torch.pow(1 - pt, self.gamma) * loss + self.alpha[:, 1] * (
                    1 - target) * torch.pow(pt, self.gamma) * loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
