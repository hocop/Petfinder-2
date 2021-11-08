import numpy as np
import torch
import torch.nn as nn


class RegressionLogLossWithMargin(nn.Module):
    def __init__(self, vmin, vmax, regression_margin):
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.regression_margin = regression_margin

        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        m = self.regression_margin
        z = (self.vmax - self.vmin + 2 * self.regression_margin)
        pred_0_1 = (pred - self.vmin + m) / z
        target_0_1 = (target - self.vmin + m) / z

        return self.bce_loss(pred_0_1, target_0_1) * z


class HingeLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()

        self.margin = margin

    def forward(self, pred, target):
        upper = (pred - target - self.margin).clamp(min=0)
        lower = (target - self.margin - pred).clamp(min=0)
        return (upper + lower)


class WeightedLoss(nn.Module):
    def __init__(self, beta, loss_1, loss_2):
        super().__init__()

        self.beta = beta
        self.loss_1 = loss_1
        self.loss_2 = loss_2

    def forward(self, pred, target):
        return self.beta * self.loss_1(pred, target) + (1 - self.beta) * self.loss_2(pred, target)
