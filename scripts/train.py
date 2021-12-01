'''
Training script
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

    rmse_scores, rmse_avg_scores, rmse_top3_scores = [], [], []

    for fold, (train_index, val_index) in enumerate(kf.split(data_train, bins)):
        print('#' * 50)
        print(f'Fold {fold + 1} / {config.num_folds}')
        print('#' * 50)

        if fold == config.max_folds:
            print('max_folds reached')
            break

        # Create train, valid, and test datasets
        data_module = pf2.datasets.PetDataModule(
            config,
            data_train.iloc[train_index],
            data_train.iloc[val_index],
        )

        # Save path
        if config.save == 1:
            model_path = os.path.join('saved_model', config.name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, f'fold_{fold}.pt')
        else:
            model_path = None

        # Create lit module
        lit_module = pf2.pl_module.LitPet(
            fold=fold,
            model_path=model_path,
            **config.__dict__
        )
        if config.load_from is not None:
            pet_net = torch.jit.load(config.load_from)
            lit_module.pet_net.pet_net.load_state_dict(pet_net.pet_net.state_dict())
            # lit_module.pet_net.load_state_dict(pet_net.state_dict())
            print()
            print('Loaded', config.load_from)
            print()
            del pet_net

        # Create trainer
        trainer = pl.Trainer.from_argparse_args(
            config,
            logger=logger,
            reload_dataloaders_every_n_epochs=1,
        )

        # Train model
        # pylint: disable=no-member
        trainer.fit(
            lit_module,
            data_module,
        )

        # TODO Train on val for one more epoch
        if config.save == 1:
            pass

        # Remember score
        print('RMSE:', lit_module.rmse_score)
        rmse_scores.append(lit_module.rmse_score)
        rmse_avg_scores.append(np.mean(lit_module.rmse_list))
        rmse_top3_scores.append(lit_module.top3_avg)

    print('RMSE:', rmse_scores)
    print('RMSE:', np.mean(rmse_scores), '+-', np.std(rmse_scores))

    wandb.log({
        'CV RMSE': np.mean(rmse_scores),
        'CV RMSE std': np.std(rmse_scores),
        'CV RMSE+std': np.mean(rmse_scores) + np.std(rmse_scores),
        'CV RMSE average': np.mean(rmse_avg_scores),
        'CV RMSE Top3': np.mean(rmse_top3_scores),
    })


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
    parser.add_argument(
        '--wandb_project', type=str, default=None,
        help='project name for Weights&Biases')
    parser.add_argument(
        '--load_from', type=str, default=None,
        help='saved checkpoint')
    parser.add_argument(
        '--save', type=int, default=0,
        help='If 1, save trained model to ./saved_models/{name}/')
    parser.add_argument(
        '--num_folds', type=int, default=None
    )
    parser.add_argument(
        '--max_folds', type=int, default=None
    )

    parser = pf2.pl_module.LitPet.add_argparse_args(parser)
    parser = pf2.datasets.PetDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser = pf2.utils.initialization.set_argparse_defaults_from_yaml(parser)

    args = parser.parse_args()

    # Run program
    main(args)
