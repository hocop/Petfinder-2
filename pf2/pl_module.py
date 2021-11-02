'''
Model class using pytorch lightning
'''

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
import timm


from .networks import SWIN
from .losses import RegressionLogLossWithMargin


class LitPet(pl.LightningModule):
    def __init__(
        self,
        fold,
        model_path=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.fold = fold
        self.model_path = model_path

        # Make net
        self.pet_net = SWIN(
            **self.hparams.model_params,
        )

        self.loss_fn = RegressionLogLossWithMargin(0, 100, self.hparams.regression_margin)

        self.rmse_score = 1000.0

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model Parameters")
        parser.add_argument('--image_size', type=int, default=None)
        parser.add_argument('--model_params', type=dict, default=None)
        parser.add_argument('--mixup_proba', type=float, default=None)
        parser.add_argument('--learning_rate', type=float, default=None)
        parser.add_argument('--regression_margin', type=float, default=None)
        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        # )
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': lr_scheduler,
        }

    def make_mixup(self, batch_size: int, alpha: float=0.5):
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(batch_size)
        return lam, rand_index

    def mixup(self, x: torch.Tensor, y: torch.Tensor, lam, rand_index):
        mixed_x = lam * x + (1 - lam) * x[rand_index, :]
        target_a, target_b = y, y[rand_index]
        return mixed_x, target_a, target_b, lam

    def forward(self, image, features):
        x = self.pet_net(image, features)
        out = torch.sigmoid(x[:, 0]) * (100 + self.hparams.regression_margin * 2) - self.hparams.regression_margin
        return out

    def training_step(self, batch, batch_idx):
        # Dict for metrics
        log_dict = {}

        if np.random.random() < self.hparams.mixup_proba:
            # Make Mixup
            lam, rand_index = self.make_mixup(batch['image'].shape[0])
            # Predict pawplarity
            pred = self(batch['image'], batch['features'], mixup=(lam, rand_index))
            # Compute mixup loss
            target_a, target_b = batch['target'], batch['target'][rand_index]
            loss_a = self.loss_fn(pred, target_a)
            loss_b = self.loss_fn(pred, target_b)
            loss = loss_a * lam + loss_b * (1 - lam)
        else:
            # Predict pawplarity
            pred = self(batch['image'], batch['features'])
            loss = self.loss_fn(pred, batch['target'])
            # Measure MSE
            log_dict['train/mse'] = ((pred - batch['target'])**2).mean()

        if loss != loss:
            print('ERROR: NaN loss!')

        log_dict['train/loss'] = loss

        self.log_dict(
            log_dict,
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0, mode='valid'):
        # Make darker and lighter versions
        dark = batch['image'] * 0.95
        light = (batch['image'] * 1.05).clip(0, 1)
        # Predict pawplarity
        pred_dark = self(
            dark,
            batch['features']
        )
        pred_dark_flip = self(
            dark.flip(-1),
            batch['features']
        )
        pred_light = self(
            light,
            batch['features']
        )
        pred_light_flip = self(
            light.flip(-1),
            batch['features']
        )
        pred = 0.25 * (pred_dark + pred_dark_flip + pred_light + pred_light_flip)
        pred = pred.clamp(0, 100)
        return pred.detach().cpu().numpy(), batch['target'].cpu().numpy(), batch['dog_detected'].cpu().numpy()

    def validation_epoch_end(self, validation_step_outputs, mode='valid'):
        '''
        Evaluate MSE and R2
        '''
        # Concatenate predictions
        predictions = np.concatenate([
            out[0]
            for out in validation_step_outputs
        ], 0)
        targets = np.concatenate([
            out[1]
            for out in validation_step_outputs
        ], 0)
        dog_detected = np.concatenate([
            out[2]
            for out in validation_step_outputs
        ], 0)

        # Compute scores
        mse_score = ((predictions - targets)**2).mean()
        rmse_score = np.sqrt(mse_score)
        target_variance = ((targets - targets.mean())**2).mean()
        r2_score = (target_variance - mse_score) / target_variance

        # Compute score for dogs separately
        mse_score_dogs = ((predictions - targets)**2)[dog_detected == 1].mean()
        rmse_score_dogs = np.sqrt(mse_score_dogs)

        # Save best model
        if rmse_score < self.rmse_score:
            if self.model_path is not None:
                model = self.pet_net.eval().cpu()
                with torch.jit.optimized_execution(True):
                    traced_graph = torch.jit.script(
                        model,
                        torch.randn(1, 3, self.hparams.image_size, self.hparams.image_size, device='cpu')
                    )
                traced_graph.save(self.model_path)
                self.pet_net.cuda()
                torch.cuda.empty_cache()
                print('Saved model to', self.model_path)

            # Remember RMSE
            self.rmse_score = rmse_score

        # Log average pawpularity metrics
        metrics = {
            'cv_step': self.fold * self.hparams.max_epochs + self.current_epoch,
            f'{mode}/MSE': mse_score,
            f'{mode}/RMSE': rmse_score,
            f'{mode}/Dog_RMSE': rmse_score_dogs,
            f'{mode}/R2': r2_score,
        }
        if self.current_epoch == self.hparams.max_epochs - 1:
            metrics['Best RMSE'] = self.rmse_score
            metrics['fold'] = self.fold
        self.log_dict(
            metrics,
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, mode='test')

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs, mode='test')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict