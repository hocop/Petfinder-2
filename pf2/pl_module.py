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
import copy

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
        torch.manual_seed(self.hparams.model_seed)
        if self.hparams.model_type == 'swin':
            self.pet_net = SWIN(
                model_name=self.hparams.swin_model_name,
                num_image_neurons=self.hparams.swin_num_image_neurons,
                dropout_1=self.hparams.swin_dropout_1,
                dropout_2=self.hparams.swin_dropout_2,
                attn_drop=self.hparams.swin_attn_drop,
                freeze_layers=self.hparams.swin_freeze_layers,
                regression_margin_bot=self.hparams.regression_margin_bot,
                regression_margin_top=self.hparams.regression_margin_top,
            )
        elif self.hparams.model_type == 'detr':
            self.pet_net = PetDETR(
                num_boxes=self.hparams.detr_num_boxes,
                regression_margin_bot=self.hparams.regression_margin_bot,
                regression_margin_top=self.hparams.regression_margin_top,
            )
        torch.manual_seed(self.hparams.random_seed)

        self.loss_log = RegressionLogLossWithMargin(
            1, 100,
            self.hparams.regression_margin_bot,
            self.hparams.regression_margin_top,
        )
        self.loss_hinge = HingeLoss(self.hparams.hinge_margin)

        self.loss_fn = WeightedLoss(
            self.hparams.bce_weight,
            self.loss_log, self.loss_hinge,
        )

        self.rmse_score = 1000.0
        self.rmse_list = []
        self.top3_avg = None
        self.best_weights = None

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model Parameters")
        parser.add_argument('--image_size', type=int, default=None)
        parser.add_argument('--weight_decay', type=float, default=None)
        parser.add_argument('--regression_margin_top', type=float, default=None)
        parser.add_argument('--regression_margin_bot', type=float, default=None)
        parser.add_argument('--model_type', type=str, default=None)
        parser.add_argument('--swin_model_name', type=str, default=None)
        parser.add_argument('--swin_num_image_neurons', type=int, default=None)
        parser.add_argument('--swin_dropout_1', type=float, default=None)
        parser.add_argument('--swin_dropout_2', type=float, default=None)
        parser.add_argument('--swin_attn_drop', type=float, default=None)
        parser.add_argument('--swin_freeze_layers', type=int, default=None)
        parser.add_argument('--detr_num_boxes', type=float, default=None)
        parser.add_argument('--model_seed', type=int, default=None)
        # Loss
        parser.add_argument('--bce_weight', type=float, default=None)
        parser.add_argument('--hinge_margin', type=float, default=None)
        parser.add_argument('--augtgt_coef', type=float, default=None)
        # Prediction
        parser.add_argument('--prediction_lightness_delta', type=float, default=None)
        parser.add_argument('--prediction_orig_weight', type=float, default=None)
        parser.add_argument('--prediction_pet_crop_weight', type=float, default=None)
        parser.add_argument('--prediction_glob_crop_weight', type=float, default=None)
        # Optimizer
        parser.add_argument('--optimizer_type', type=str, default=None)
        parser.add_argument('--learning_rate', type=float, default=None)
        parser.add_argument('--sgd_momentum', type=float, default=None)
        parser.add_argument('--sgd_dampening', type=float, default=None)
        parser.add_argument('--sgd_nesterov', type=bool, default=None)
        # Schedule
        parser.add_argument('--freeze_backend_steps', type=int, default=None)
        parser.add_argument('--freeze_backend_last_epochs', type=int, default=None)
        parser.add_argument('--scheduler_type', type=str, default=None)
        parser.add_argument('--scheduler_cos_t_max', type=int, default=None)
        # Mixup
        parser.add_argument('--mix_proba', type=float, default=None)
        parser.add_argument('--mixup_alpha', type=float, default=None)
        parser.add_argument('--cutmix_alpha', type=float, default=None)
        parser.add_argument('--cutmix_proba', type=float, default=None)
        return parent_parser

    def configure_optimizers(self):
        if self.hparams.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.pet_net.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.pet_net.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.sgd_momentum,
                dampening=self.hparams.sgd_dampening * float(not self.hparams.sgd_nesterov),
                nesterov=self.hparams.sgd_nesterov,
            )
        if self.hparams.scheduler_type == 'none':
            return optimizer
        if self.hparams.scheduler_type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.scheduler_cos_t_max,
            )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

    def forward(self, image, features, freeze_backend=torch.tensor(False)):
        x, image_features = self.pet_net(image, features, freeze_backend=freeze_backend)
        return x, image_features

    def rand_index(self, cat_detected, target):
        # Don't mixup label 100 with others
        # is_border = (target == 100).to(torch.float32)
        # not_border = 1 - is_border
        # mask_border = is_border[:, None] * is_border[None, :] + not_border[:, None] * not_border[None, :]

        # Don't mixup cats with dogs
        dog_detected = 1 - cat_detected
        mask_pet = cat_detected[:, None] * cat_detected[None, :] + dog_detected[:, None] * dog_detected[None, :]

        # Generate random indeces
        mask = mask_pet# * mask_border
        rnd = torch.rand(mask.shape, device=mask.device) * mask
        rnd_idx = torch.argmax(rnd, dim=1)

        return rnd_idx

    def mixup(self, x: torch.Tensor, feats: torch.Tensor, y: torch.Tensor, rand_index=None):

        if rand_index is None:
            rand_index = torch.randperm(x.size()[0])

        # Mix images
        if np.random.random() < self.hparams.cutmix_proba:
            # Use cutmix
            lam = np.random.beta(self.hparams.cutmix_alpha, self.hparams.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            mixed_x = x.detach().clone()
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # Adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        else:
            # Use mixup
            lam = np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
            mixed_x = lam * x + (1 - lam) * x[rand_index]

        # Mixup feats
        mixed_feats = lam * feats + (1 - lam) * feats[rand_index]

        # Shuffle targets
        target_a, target_b = y, y[rand_index]

        return mixed_x, mixed_feats, target_a, target_b, lam

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # Dict for metrics
        log_dict = {}

        cat_detected = batch['cat_detected'].to(batch['target'].dtype)
        # dog_detected = 1 - cat_detected

        if self.hparams.scheduler_type != 'none':
            self.log_dict(
                {'lr': self.lr_schedulers().get_last_lr()[0]},
                on_step=True, on_epoch=False, prog_bar=True, logger=False
            )

        freeze_backend = (
            self.global_step < self.hparams.freeze_backend_steps
            or
            self.current_epoch >= self.hparams.max_epochs - self.hparams.freeze_backend_last_epochs
        )

        # Maybe augment target
        rate = torch.ones_like(batch['target']) / (self.hparams.regression_margin_top * max(self.hparams.augtgt_coef, 0.01))
        m = torch.distributions.exponential.Exponential(rate)
        addition = m.sample().clip(0, self.hparams.regression_margin_top)
        if self.hparams.augtgt_coef > 0:
            mask_100 = (batch['target'] == 100).to(torch.float32)
            batch['target'] = batch['target'] + addition * mask_100

        if np.random.random() < self.hparams.mix_proba and batch['image'].size()[0] > 1:
            # Mixup
            mixed_x, mixed_feats, target_a, target_b, lam = self.mixup(
                batch['image'], batch['features'], batch['target'],
                rand_index=self.rand_index(cat_detected, batch['target']),
            )
            # Predict pawplarity
            pred, image_features = self(mixed_x, mixed_feats, freeze_backend=freeze_backend)
            # Distribute predictions
            if lam > 0.5:
                pred_a = pred[:, 0]
                pred_b = pred[:, -1]
            else:
                pred_a = pred[:, -1]
                pred_b = pred[:, 0]
            # Supervision loss
            loss_a = self.loss_log(pred_a, target_a).mean()
            loss_b = self.loss_log(pred_b, target_b).mean()
            loss = lam * loss_a + (1 - lam) * loss_b
            pred = pred[:, 0]
        else:
            # Predict pawplarity
            pred, image_features = self(batch['image'], batch['features'], freeze_backend=freeze_backend)
            pred = pred[:, 0]
            # Supervision loss
            loss = self.loss_fn(pred, batch['target']).mean()

        log_dict['train/loss'] = loss

        # Measure MSE
        log_dict['train/mse'] = ((pred - batch['target'])**2).mean()

        if loss != loss:
            print('ERROR: NaN loss!')

        self.log_dict(
            log_dict,
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        return loss

    def make_prediction_fast(self, batch):
        # Predict pawplarity
        inp_image = torch.cat([
            batch['image'],# batch['image'].flip(-1),
            batch['image_pet'],# batch['image_pet'].flip(-1),
            batch['image_glob'],# batch['image_glob'].flip(-1),
        ], 0)
        int_feats = batch['features'].repeat([3, 1])
        with torch.no_grad():
            pred, image_features = self(inp_image, int_feats)
            pred = pred[:, 0]
        pred = pred.view([3, batch['image'].shape[0]])

        # Average
        a = self.hparams.prediction_orig_weight
        b = self.hparams.prediction_pet_crop_weight
        c = self.hparams.prediction_glob_crop_weight
        s = a + b + c
        a = a / s
        b = b / s
        c = c / s
        # pred = a * pred[:2].mean(0) + b * pred[2:4].mean(0) + c * pred[4:].mean(0)
        pred = a * pred[0] + b * pred[1] + c * pred[2]

        return pred

    def make_prediction(self, batch):
        # Concatenate images
        inp_image = torch.cat([
            batch['image'], batch['image'].flip(-1),
            batch['image_pet'], batch['image_pet'].flip(-1),
            batch['image_glob'], batch['image_glob'].flip(-1),
        ], 0)
        int_feats = batch['features'].repeat([6, 1])
        
        # Make darker and lighter versions
        dark = (inp_image * (1 - self.hparams.prediction_lightness_delta)).clip(0, 1)
        light = (inp_image * (1 + self.hparams.prediction_lightness_delta)).clip(0, 1)
        
        # Predict pawpularity
        with torch.no_grad():
            # Predict with different lightness
            dark_pred, image_features_dark = self(dark, int_feats)
            light_pred, image_features_light = self(light, int_feats)
            # Average
            pred = (dark_pred[:, 0] + light_pred[:, 0]) / 2
        pred = pred.view([6, batch['image'].shape[0]])

        # Average
        a = self.hparams.prediction_orig_weight
        b = self.hparams.prediction_pet_crop_weight
        c = self.hparams.prediction_glob_crop_weight
        s = a + b + c
        a = a / s
        b = b / s
        c = c / s
        pred = a * pred[:2].mean(0) + b * pred[2:4].mean(0) + c * pred[4:].mean(0)

        return pred

    def validation_step(self, batch, batch_idx, dataloader_idx=0, mode='valid'):
        # pred = self.make_prediction_fast(batch)
        with torch.no_grad():
            pred = self(batch['image'], batch['features'])[0][:, 0]
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
        if not self.trainer.sanity_checking:
            self.rmse_list.append(rmse_score)
        target_variance = ((targets - targets.mean())**2).mean()
        r2_score = (target_variance - mse_score) / target_variance

        # Compute score for dogs separately
        mse_score_dogs = (((predictions - targets)**2) * dog_detected).sum() / dog_detected.sum()
        rmse_score_dogs = np.sqrt(mse_score_dogs)

        # Compute score for cats separately
        mse_score_cats = (((predictions - targets)**2) * cat_detected).sum() / cat_detected.sum()
        rmse_score_cats = np.sqrt(mse_score_cats)

        # Remember best model
        if rmse_score < self.rmse_score and not self.trainer.sanity_checking:
            # Remember RMSE
            self.rmse_score = rmse_score

            # Remember weights
            self.pet_net.cpu()
            self.best_weights = copy.deepcopy(self.pet_net.state_dict())
            self.pet_net.cuda()

        # Restore best model
        if (
            (self.current_epoch == self.hparams.max_epochs - self.hparams.freeze_backend_last_epochs - 1)
            and
            self.hparams.freeze_backend_last_epochs > 0
            and
            not self.trainer.sanity_checking
        ):
            self.pet_net.load_state_dict(self.best_weights)
            self.pet_net.pet_net.requires_grad_(False)
            print()
            print('Loaded model from previous best epoch')

        # Save best model to disk
        if self.model_path is not None and self.current_epoch == self.hparams.max_epochs - 1:
            self.pet_net.load_state_dict(self.best_weights)
            model = self.pet_net.eval().cpu()
            with torch.jit.optimized_execution(True):
                traced_graph = torch.jit.script(
                    model,
                    torch.randn(1, 3, self.hparams.image_size, self.hparams.image_size, device='cpu')
                )
            traced_graph.save(self.model_path)
            self.pet_net.cuda()
            torch.cuda.empty_cache()
            print()
            print('Saved model to', self.model_path)

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
            self.top3_avg = np.array(self.rmse_list)[np.argsort(self.rmse_list)[:3]].mean()
            metrics['Best RMSE'] = self.rmse_score
            metrics['Avg RMSE over steps'] = np.mean(self.rmse_list)
            metrics['RMSE top3 avg'] = self.top3_avg
            metrics['fold'] = self.fold
        self.log_dict(
            metrics,
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, mode='test')

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs, mode='test')

    # def get_progress_bar_dict(self):
    #     tqdm_dict = super().get_progress_bar_dict()
    #     tqdm_dict.pop("v_num", None)
    #     return tqdm_dict



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
