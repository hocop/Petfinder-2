import numpy as np
import torch
import torch.nn as nn


class RegressionLogLossWithMargin(nn.Module):
    def __init__(
        self,
        vmin,
        vmax,
        regression_margin_bot,
        regression_margin_top,
    ):
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.regression_margin_bot = regression_margin_bot
        self.regression_margin_top = regression_margin_top

        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        a = self.regression_margin_bot
        b = self.regression_margin_top
        z = (self.vmax - self.vmin + a + b)
        pred_0_1 = (pred - self.vmin + a) / z
        target_0_1 = (target - self.vmin + a) / z

        # Clip out-of-range predictions
        mask_clip_top = (target >= self.vmax) & (pred.detach() >= self.vmax)
        # mask_clip_bot = (target <= self.vmin) & (pred.detach() <= self.vmin)
        # mask = 1 - (mask_clip_top | mask_clip_bot).to(torch.float32)
        mask = 1 - mask_clip_top.to(torch.float32)

        if (1 - mask).sum() > 0:
            print('pred', pred)

        return self.bce_loss(pred_0_1, target_0_1) * z * mask


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
