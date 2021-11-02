'''
Classes for loading optical flow datasets
'''

import os
import cv2
import numpy as np
from argparse import Namespace
from typing import Optional
import torch
from torch.utils.data import DataLoader
import albumentations as A
import pytorch_lightning as pl
import nnio


class PetDataModule(pl.LightningDataModule):
    def __init__(self, config: Namespace, train_table, valid_table):
        super().__init__()

        self.config = config
        self.train_table = train_table
        self.valid_table = valid_table

        # Augmentations for training
        self.augmentations = A.Compose([
            # Flips
            A.HorizontalFlip(p=0.5),
            # Color transforms
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        ])

        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None
        self.freeze_augmentations = self.config.freeze_augmentations

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataloader parameters")
        parser.add_argument('--data_path', type=str, default=None)
        parser.add_argument('--num_folds', type=int, default=None)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--freeze_augmentations', type=int, default=None)
        return parent_parser

    def setup(self, stage: Optional[str] = None):
        # Create training dataset
        if self.train_dataset is None:
            self.train_dataset = PetDataset(
                self.config.data_path,
                self.train_table,
                image_size=self.config.image_size,
                augmentations=self.augmentations,
            )

        # Create validation datasets
        if self.val_datasets is None:
            self.val_datasets = PetDataset(
                self.config.data_path,
                self.valid_table,
                image_size=self.config.image_size,
                augmentations=None,
            )


    def train_dataloader(self):
        # Freeze augmentations for some first epochs
        self.freeze_augmentations -= 1
        if self.freeze_augmentations >= 0:
            self.train_dataset.augmentations = None
        else:
            self.train_dataset.augmentations = self.augmentations

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_datasets,
            batch_size=self.config.batch_size,
            num_workers=2
        )

    def test_dataloader(self):
        return None


class PetDataset(torch.utils.data.Dataset):
    '''
    '''
    dense_features = [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ]

    def  __init__(
        self,
        data_path,
        table,
        image_size=None,
        augmentations=None,
        channels_first=True,
    ):
        self.data_path = data_path
        self.resize = (image_size, image_size) if image_size is not None else None
        self.augmentations = augmentations
        self.channels_first = channels_first

        self.preproc = nnio.Preprocessing(
            resize=self.resize,
            channels_first=channels_first,
            dtype='float32',
            divide_by_255=True,
            # means=[0.485, 0.456, 0.406],
            # stds=[0.229, 0.224, 0.225],
        )

        self.table = table

        self.boxes = open(os.path.join(data_path, 'boxes_train.txt')).readlines()

    def __len__(self):
        return len(self.table)

    @staticmethod
    def make_box_more_like_square(x_min, x_max, y_min, y_max, image):
        # Make area be more like square
        if x_max - x_min > y_max - y_min:
            y_c = (y_max + y_min) // 2
            r = (x_max - x_min) // 2
            y_max = min(y_c + r, image.shape[0])
            y_min = max(y_c - r, 0)
        else:
            x_c = (x_max + x_min) // 2
            r = (y_max - y_min) // 2
            x_max = min(x_c + r, image.shape[1])
            x_min = max(x_c - r, 0)
        return x_min, x_max, y_min, y_max

    def __getitem__(self, item):
        row = self.table.iloc[item]

        # Get features
        features = row[self.dense_features].values.astype('float32')
        # Get target
        target = row.Pawpularity.astype('float32')

        # Load image
        img_path = os.path.join(self.data_path, 'train', row['Id'] + '.jpg')
        image = cv2.imread(img_path)[:, :, ::-1]

        # Add image size to features
        features = np.concatenate([
            features,
            np.array(image.shape[:2], dtype='float32') / 1000,
        ])

        # Make background mask and find region of interest
        boxes = self.boxes[item].strip().split(';')[1:]
        mask = np.zeros(image.shape[:2], dtype='uint8')
        x_min_g, x_max_g, y_min_g, y_max_g = mask.shape[1], 0, mask.shape[0], 0
        x_min_dog, x_max_dog, y_min_dog, y_max_dog = None, None, None, None
        clip = lambda x: max(min(x, 1), 0)
        area = lambda x_min, x_max, y_min, y_max: (x_max - x_min) * (y_max - y_min)
        for box in boxes:
            if len(box) == 0:
                continue
            box = eval(box)
            if box.label in ['dog', 'cat', 'human']:
                x_min = int(clip(box.x_min) * mask.shape[1])
                x_max = int(clip(box.x_max) * mask.shape[1])
                y_min = int(clip(box.y_min) * mask.shape[0])
                y_max = int(clip(box.y_max) * mask.shape[0])
                mask[y_min: y_max, x_min: x_max] = 1
                # Update border values
                x_min_g = min(x_min_g, x_min)
                x_max_g = max(x_max_g, x_max)
                y_min_g = min(y_min_g, y_min)
                y_max_g = max(y_max_g, y_max)
                # Find dog with max area
                if box.label == 'dog':
                    if (
                        x_min_dog is None
                        or
                        area(x_min_dog, x_max_dog, y_min_dog, y_max_dog) < area(x_min, x_max, y_min, y_max)
                    ):
                        x_min_dog, x_max_dog, y_min_dog, y_max_dog = x_min, x_max, y_min, y_max

        # Crop the biggest dog
        if x_min_dog is not None and area(x_min_dog, x_max_dog, y_min_dog, y_max_dog) > 1000:
            dog_detected = 1
            x_min_dog, x_max_dog, y_min_dog, y_max_dog = self.make_box_more_like_square(
                x_min_dog, x_max_dog, y_min_dog, y_max_dog,
                image
            )
            image_dog = image[y_min_dog: y_max_dog, x_min_dog: x_max_dog]
        else:
            dog_detected = 0
            image_dog = np.zeros([10, 10, 3], dtype='uint8')

        # Replace background with white noise
        if (self.augmentations is not None) and (np.random.random() < 0.01):
            noise = np.random.randint(0, 255, size=image.shape, dtype='uint8')
            image = image * mask[:, :, None] + noise * (1 - mask)[:, :, None]

        # If something useful is detected
        if x_min_g < x_max_g - 1 and y_min_g < x_max_g - 1:
            # Make area be more like square
            x_min_g, x_max_g, y_min_g, y_max_g = self.make_box_more_like_square(
                x_min_g, x_max_g, y_min_g, y_max_g,
                image
            )

            # Crop image
            if (self.augmentations is not None) and (np.random.random() < 0.1):
                image = image[y_min_g: y_max_g, x_min_g: x_max_g]

        # Augment image
        if self.augmentations is not None:
            image = self.augmentations(
                image=image,
            )['image']
            image_dog = self.augmentations(
                image=image_dog,
            )['image']

        # Resize and preprocess images
        image = self.preproc(image)
        image_dog = self.preproc(image_dog)

        return {
            'image': image,
            'features': features,
            'target': target,
            'dog_detected': dog_detected,
            'image_dog': image_dog,
        }


if __name__ == '__main__':
    DATA_PATH = '/home/ruslan/data/datasets_ssd/kaggle/petfinder-pawpularity-score/'
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib

    data_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

    dataset = PetDataset(DATA_PATH, data_train, channels_first=False)

    for i in range(len(dataset)):
        image = dataset[i]['image']
        image_dog = dataset[i]['image_dog']
        print(dataset[i]['dog_detected'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        axes[0].imshow(image)
        axes[1].imshow(image_dog)
        plt.show()
