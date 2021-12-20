'''
Prediction script
'''

import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import matplotlib
matplotlib.use("agg")
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm
from joblib import dump, load

import pf2


def main(config):
    # Initialize config
    logger = pf2.initialize(config)

    # Load table
    data_train = pd.read_csv(os.path.join(config.data_path, 'train.csv'))

    # Load predictions
    pred_path = os.path.join('saved_model', config.name, 'oof_prediction.csv')
    predictions = pd.read_csv(pred_path)

    # Measure RMSE
    cols = predictions.drop('Id', axis=1).columns
    print('RMSE:')
    for col in cols:
        residual = predictions[col] - data_train['Pawpularity']
        rmse_score = np.sqrt(np.mean(residual**2, 0))
        print(f'{col}:{" " * (15 - len(col))}{rmse_score}')

    # As dataset
    X = predictions.drop(['Id'], axis=1)
    y = data_train['Pawpularity']
    print()

    # Average manually
    a, b, c = 1, 0.5, 1
    s = a + b + c
    a, b, c = a / s, b / s, c / s
    flip_w = 0.4
    pred_a = (1 - flip_w) * predictions['image'] + flip_w * predictions['image_flip']
    pred_b = (1 - flip_w) * predictions['pet'] + flip_w * predictions['pet_flip']
    pred_c = (1 - flip_w) * predictions['glob'] + flip_w * predictions['glob_flip']
    pred = a * pred_a + b * pred_b + c * pred_c
    rmse = np.sqrt(((pred - y)**2).mean())
    print('Manual RMSE:', rmse)

    # Evaluate blending model
    top_model = Pipeline([
        # ('scaler', StandardScaler()),
        ('regr', Ridge(alpha=1, fit_intercept=False, positive=True))
        # ('regr', SVR(C=1.0, kernel='linear'))
    ])
    cv_mse = cross_val_score(
        top_model,
        X,
        y,
        scoring='neg_mean_squared_error',
    )
    cv_rmse = np.sqrt(-cv_mse.mean())
    print('Blending CV RMSE:', cv_rmse)

    # Train blending model on whole dataset
    top_model.fit(X, y)
    pred = top_model.predict(X)
    rmse = np.sqrt(((pred - y)**2).mean())
    print('Blending RMSE:', rmse)

    print()
    print(top_model['regr'].coef_)
    print(top_model['regr'].coef_.sum())

    # Save blending model
    save_path = os.path.join('saved_model', config.name, 'top_model')
    dump(top_model, save_path)


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
