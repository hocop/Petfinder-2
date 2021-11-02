'''
Very important file.
initialize() must be called from train.py

Functions here are responsible for loading data config from .yaml file
'''

import os
from argparse import Namespace
import yaml

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import wandb


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


def initialize(config: Namespace):
    '''
    1. Set random seeds
    2. Initialize wandb if `config.wandb_project` is given
    3. Renames wandb run if `config.name` is given
    4. Loads parameters from config file if given

    Parameters
    ----------
    config: Namespace

    Returns
    -------
    data_config: dict
    '''
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    logger = []

    # Initialize wandb run
    if config.__dict__.get('wandb_project', None) is not None:
        logger.append(WandbLogger(
            config.name,
            config=config.__dict__,
            project=config.wandb_project,
            sync_tensorboard=True,
            dir='logs/',
            save_dir='logs/',
        ))

    logger.append(TensorBoardLogger('logs/tensorboard/'))

    # Set the random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Set default accelerator to ddp
    if config.__dict__.get('gpus', None) is not None and config.gpus > 1 and config.accelerator is None:
        config.accelerator = 'ddp'

    return logger


def set_argparse_defaults_from_yaml(parser):
    config = parser.parse_args()

    # Read config from config file
    if config.config is not None:
        with open(config.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    return parser