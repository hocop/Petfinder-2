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


from .networks import SWIN, PetDETR
from .losses import RegressionLogLossWithMargin, HingeLoss, WeightedLoss


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
        if self.hparams.model_type == 'swin':
            self.pet_net = SWIN(
                model_name=self.hparams.swin_model_name,
                num_image_neurons=self.hparams.swin_num_image_neurons,
                dropout_1=self.hparams.swin_dropout_1,
                dropout_2=self.hparams.swin_dropout_2,
                attn_drop=self.hparams.swin_attn_drop,
                freeze_layers=self.hparams.swin_freeze_layers,
                regression_margin=self.hparams.regression_margin,
            )
        elif self.hparams.model_type == 'detr':
            self.pet_net = PetDETR(
                num_boxes=self.hparams.detr_num_boxes,
                regression_margin=self.hparams.regression_margin,
            )

        # Domain adaptation GAN
        if self.hparams.discrim_layers == 1:
            self.discriminator = nn.Linear(
                self.hparams.swin_num_image_neurons,
                1
            )
        elif self.hparams.discrim_layers == 2:
            self.discriminator = nn.Sequential(
                nn.Linear(self.hparams.swin_num_image_neurons, 100),
                nn.ReLU(True),
                nn.Linear(100, 1),
            )
        self.loss_gan = nn.BCEWithLogitsLoss(reduction='none')

        self.loss_log = RegressionLogLossWithMargin(1, 100, self.hparams.regression_margin)
        self.loss_hinge = HingeLoss(self.hparams.hinge_margin)

        self.loss_fn_cat = WeightedLoss(
            self.hparams.bce_weight_cat,
            self.loss_log, self.loss_hinge,
        )
        self.loss_fn_dog = WeightedLoss(
            self.hparams.bce_weight_dog,
            self.loss_log, self.loss_hinge,
        )

        self.rmse_score = 1000.0
        self.rmse_list = []

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model Parameters")
        parser.add_argument('--image_size', type=int, default=None)
        parser.add_argument('--mixup_proba', type=float, default=None)
        parser.add_argument('--learning_rate', type=float, default=None)
        parser.add_argument('--weight_decay', type=float, default=None)
        parser.add_argument('--regression_margin', type=float, default=None)
        parser.add_argument('--model_type', type=str, default=None)
        parser.add_argument('--swin_model_name', type=str, default=None)
        parser.add_argument('--swin_num_image_neurons', type=int, default=None)
        parser.add_argument('--swin_dropout_1', type=float, default=None)
        parser.add_argument('--swin_dropout_2', type=float, default=None)
        parser.add_argument('--swin_attn_drop', type=float, default=None)
        parser.add_argument('--swin_freeze_layers', type=int, default=None)
        parser.add_argument('--detr_num_boxes', type=float, default=None)
        parser.add_argument('--bce_weight_cat', type=float, default=None)
        parser.add_argument('--bce_weight_dog', type=float, default=None)
        parser.add_argument('--prediction_lightness_delta', type=float, default=None)
        parser.add_argument('--prediction_pet_crop_weight', type=float, default=None)
        parser.add_argument('--prediction_glob_crop_weight', type=float, default=None)
        parser.add_argument('--hinge_margin', type=float, default=None)
        # GAN
        parser.add_argument('--discrim_learning_rate', type=float, default=None)
        parser.add_argument('--generator_loss_weight', type=float, default=None)
        parser.add_argument('--discrim_layers', type=int, default=None)
        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.pet_net.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.old_domain_prob <= 0:
            return optimizer
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.discrim_learning_rate)
        return [optimizer, optimizer_d]

    def forward(self, image, features):
        x, image_features = self.pet_net(image, features)
        return x, image_features

    def mixup(self, x: torch.Tensor, feats: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
        assert alpha > 0, "alpha should be larger than 0"
        assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(x.size()[0])
        mixed_x = lam * x + (1 - lam) * x[rand_index]
        mixed_feats = lam * feats + (1 - lam) * feats[rand_index]
        target_a, target_b = y, y[rand_index]
        return mixed_x, mixed_feats, target_a, target_b, lam

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # Dict for metrics
        log_dict = {}

        # Predict pawplarity
        pred, image_features = self(batch['image'], batch['features'])

        # Old domain mask
        old_mask = (batch['target'] < 0).to(batch['target'].dtype)
        new_mask = 1 - old_mask

        if optimizer_idx == 0:
            # Supervision loss (for pf2 images)
            cat_detected = batch['cat_detected'].to(pred.dtype)
            dog_detected = 1 - cat_detected
            loss = (
                self.loss_fn_cat(pred.mean(1), batch['target']) * cat_detected
                +
                self.loss_fn_dog(pred.mean(1), batch['target']) * dog_detected
            )
            loss = (loss * new_mask).sum() / (new_mask.sum() + 1e-9)

            # Domain adaptation generator loss (for old images)
            if self.hparams.old_domain_prob > 0:
                loss_gen = self.loss_gan(
                    self.discriminator(image_features)[:, 0],
                    torch.zeros_like(old_mask),
                )
                loss_gen = (loss_gen * old_mask).sum() / (old_mask.sum() + 1e-9)
                loss = loss + loss_gen * self.hparams.generator_loss_weight

            # Weight decay
            loss = loss + self.pet_net.l2() * self.hparams.weight_decay

            log_dict['train/loss'] = loss

        if optimizer_idx == 1:
            # Domain adaptation discriminator loss
            loss = self.loss_gan(
                self.discriminator(image_features.detach())[:, 0],
                old_mask,
            )
            loss = loss.mean()

            log_dict['train/discrim_loss'] = loss

        # Measure MSE
        log_dict['train/mse'] = (((pred.mean(1) - batch['target'])**2) * new_mask).sum() / new_mask.sum()

        if loss != loss:
            print('ERROR: NaN loss!')

        self.log_dict(
            log_dict,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0, mode='valid'):
        # Make darker and lighter versions
        dark = batch['image'] * (1 - self.hparams.prediction_lightness_delta)
        light = (batch['image'] * (1 + self.hparams.prediction_lightness_delta)).clip(0, 1)
        # Predict pawplarity
        inp_image = torch.cat([
            dark, dark.flip(-1),
            light, light.flip(-1),
            batch['image_pet'], batch['image_pet'].flip(-1),
            batch['image_glob'], batch['image_glob'].flip(-1),
        ], 0)
        int_feats = batch['features'].repeat([8, 1])
        with torch.no_grad():
            pred, image_features = self(inp_image, int_feats)
            pred = pred.mean(1)
        pred = pred.view([8, batch['image'].shape[0]])
        # Average
        b = self.hparams.prediction_pet_crop_weight
        c = self.hparams.prediction_glob_crop_weight
        s = 1 + b + c
        a = 1 / s
        b = b / s
        c = c / s
        pred = a * pred[:4].mean(0) + b * pred[4:6].mean(0) + c * pred[6:].mean(0)
        pred = pred.clamp(1, 100)
        return {
            'pred': pred.detach().cpu().numpy(),
            'target': batch['target'].cpu().numpy(),
            'dog_detected': batch['dog_detected'].cpu().numpy(),
            'cat_detected': batch['cat_detected'].cpu().numpy(),
        }

    def validation_epoch_end(self, validation_step_outputs, mode='valid'):
        '''
        Evaluate MSE and R2
        '''
        # Concatenate predictions
        predictions = np.concatenate([
            out['pred']
            for out in validation_step_outputs
        ], 0)
        targets = np.concatenate([
            out['target']
            for out in validation_step_outputs
        ], 0)
        cat_detected = np.concatenate([
            out['cat_detected']
            for out in validation_step_outputs
        ], 0)
        dog_detected = 1 - cat_detected

        # Compute scores
        mse_score = ((predictions - targets)**2).mean()
        mae_score = np.abs(predictions - targets).mean()
        rmse_score = np.sqrt(mse_score)
        self.rmse_list.append(rmse_score)
        target_variance = ((targets - targets.mean())**2).mean()
        r2_score = (target_variance - mse_score) / target_variance

        # Compute score for dogs separately
        mse_score_dogs = (((predictions - targets)**2) * dog_detected).sum() / dog_detected.sum()
        rmse_score_dogs = np.sqrt(mse_score_dogs)

        # Compute score for cats separately
        mse_score_cats = (((predictions - targets)**2) * cat_detected).sum() / cat_detected.sum()
        rmse_score_cats = np.sqrt(mse_score_cats)

        # Save best model
        if rmse_score < self.rmse_score and not self.trainer.sanity_checking:
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
            f'{mode}/MAE': mae_score,
            f'{mode}/RMSE': rmse_score,
            f'{mode}/Dog_RMSE': rmse_score_dogs,
            f'{mode}/Cat_RMSE': rmse_score_cats,
            f'{mode}/R2': r2_score,
        }
        if self.current_epoch == self.hparams.max_epochs - 1:
            metrics['Best RMSE'] = self.rmse_score
            metrics['Avg RMSE over steps'] = np.mean(self.rmse_list)
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