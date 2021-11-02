import numpy as np
import torch
import torch.nn as nn


class RegressionLogLossWithMargin(nn.Module):
    def __init__(self, vmin, vmax, regression_margin):
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.regression_margin = regression_margin

        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        m = self.regression_margin
        z = (self.vmax - self.vmin + 2 * self.regression_margin)
        pred_0_1 = (pred + m) / z
        target_0_1 = (target + m) / z

        return self.bce_loss(pred_0_1, target_0_1)