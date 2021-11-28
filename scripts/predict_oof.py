'''
Prediction script
'''

import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import matplotlib
matplotlib.use("agg")
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import os
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm

import pf2


def main(config):
    # Initialize config
    logger = pf2.initialize(config)

    # Load table
    data_train = pd.read_csv(os.path.join(config.data_path, 'train.csv'))

    # Load boxes
    with open(os.path.join(config.data_path, 'boxes_train.txt')) as boxes:
        data_train['boxes'] = [l.strip() for l in boxes]

    # Sturges' rule
    num_bins = int(1 + np.log2(len(data_train))) // 2
    bins = pd.cut(data_train['Pawpularity'], bins=num_bins, labels=False)
    boxes = np.array([
        'cat' in l for l in
        data_train['boxes']
    ]).astype(int)
    bins += boxes * num_bins

    kf = StratifiedKFold(config.num_folds, shuffle=True, random_state=config.random_seed)

    predictions = data_train['Pawpularity'].copy()
    predictions.iloc[:] = 0.0

    for fold, (train_index, val_index) in enumerate(kf.split(data_train, bins)):
        print('#' * 50)
        print(f'Fold {fold + 1} / {config.num_folds}')
        print('#' * 50)

        # Create train, valid, and test datasets
        data_module = pf2.datasets.PetDataModule(
            config,
            data_train.iloc[train_index],
            data_train.iloc[val_index],
        )

        # Load path
        model_path = os.path.join('saved_model', config.name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, f'fold_{fold}.pt')

        # Create lit module
        lit_module = pf2.pl_module.LitPet(
            fold=fold,
            **config.__dict__
        )

        # Predict
        lit_module.pet_net = torch.jit.load(model_path).eval().cuda()
        data_module.setup()
        val_dataloader = data_module.val_dataloader()
        fold_preds = []
        for batch in tqdm(val_dataloader):
            batch = {key: batch[key].cuda() for key in batch}
            pred = lit_module.make_prediction(batch)
            fold_preds.append(pred.detach().cpu().numpy())
        fold_preds = np.concatenate(fold_preds, 0)

        # Record predictions
        predictions.iloc[val_index] = fold_preds

    # print(predictions)
    residual = predictions.values - data_train['Pawpularity'].values
    rmse_score = np.sqrt(np.mean(residual**2))
    print('RMSE', rmse_score)

    # Save predictions
    submission = pd.DataFrame(columns=['Id', 'Pawpularity'])
    submission['Id'] = data_train['Id']
    submission['Pawpularity'] = predictions
    submission.to_csv('oof_prediction.csv', index=False)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train visual pawpularity and depth at the same time')

    parser.add_argument(
        '--config', type=str,
        help='configuration file in yaml format (ex.: config_kitti.yaml)')
    parser.add_argument(
        '--data_config', type=str, default=None,
        help='dataset configuration file in yaml format (ex.: datasets/datasets[dol-ml5].yaml)')
    parser.add_argument(
        '--name', type=str, default=None,
        help='name of the experiment for logging')

    parser = pf2.pl_module.LitPet.add_argparse_args(parser)
    parser = pf2.datasets.PetDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser = pf2.utils.initialization.set_argparse_defaults_from_yaml(parser)

    args = parser.parse_args()

    # Run program
    main(args)
