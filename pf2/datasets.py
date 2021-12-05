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

        # Clear up training data
        #mask = (self.train_table.Pawpularity < 10) | (self.train_table.Pawpularity == 100)
        # mask = self.train_table.Pawpularity == 100
        # self.train_table = self.train_table[~mask]

        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None
        self.epoch_count = 0

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataloader parameters")
        parser.add_argument('--data_path', type=str, default=None)
        parser.add_argument('--choose_categories', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--freeze_augmentations', type=float, default=None)
        parser.add_argument('--cpus', type=int, default=None)
        parser.add_argument('--replace_bg_prob', type=float, default=None)
        parser.add_argument('--glob_crop_prob', type=float, default=None)
        parser.add_argument('--pet_crop_prob', type=float, default=None)
        # Augmentation parameters
        parser.add_argument('--aug_brightness_lim', type=float, default=None)
        parser.add_argument('--aug_contrast_lim', type=float, default=None)
        parser.add_argument('--aug_hue_lim', type=float, default=None)
        parser.add_argument('--aug_sat_lim', type=float, default=None)
        parser.add_argument('--aug_val_lim', type=float, default=None)
        parser.add_argument('--aug_brcon_prob', type=float, default=None)
        parser.add_argument('--aug_hsv_prob', type=float, default=None)
        parser.add_argument('--aug_gray_prob', type=float, default=None)
        parser.add_argument('--aug_gblur_prob', type=float, default=None)
        parser.add_argument('--aug_affine_prob', type=float, default=None)
        # GAN parameters
        parser.add_argument('--old_domain_prob', type=float, default=None)
        return parent_parser

    def setup(self, stage: Optional[str] = None):
        # Create training dataset
        if self.train_dataset is None:
            self.train_dataset = PetDataset(
                self.config.data_path,
                self.train_table,
                self.config.choose_categories,
                image_size=self.config.image_size,
                augmentations=None,
                old_domain_prob=self.config.old_domain_prob,
            )

        # Create validation datasets
        if self.val_datasets is None:
            self.val_datasets = PetDataset(
                self.config.data_path,
                self.valid_table,
                self.config.choose_categories,
                image_size=self.config.image_size,
                augmentations=None,
            )


    def train_dataloader(self):
        # Augmentations for training
        augs_coef = min(self.epoch_count / self.config.freeze_augmentations, 1.0)
        augmentations = A.Compose([
            # Blur
            A.GaussianBlur(p=self.config.aug_gblur_prob * augs_coef),
            # Color transforms
            A.RandomBrightnessContrast(
                brightness_limit=(-self.config.aug_brightness_lim * augs_coef, self.config.aug_brightness_lim * augs_coef),
                contrast_limit=(-self.config.aug_contrast_lim * augs_coef, self.config.aug_contrast_lim * augs_coef),
                p=self.config.aug_brcon_prob
            ),
            A.HueSaturationValue(
                hue_shift_limit=self.config.aug_hue_lim * augs_coef,
                sat_shift_limit=self.config.aug_sat_lim * augs_coef,
                val_shift_limit=self.config.aug_val_lim * augs_coef,
                p=self.config.aug_hsv_prob
            ),
            A.ToGray(p=self.config.aug_gray_prob * augs_coef),
            # Transforms
            A.ShiftScaleRotate(
                shift_limit=0.0625 * augs_coef,
                scale_limit=0.1 * augs_coef,
                rotate_limit=int(45 * augs_coef),
                p=self.config.aug_affine_prob,
            ),
        ])
        self.train_dataset.augmentations_before_crop = augmentations
        self.train_dataset.augmentations_after_crop = A.HorizontalFlip(p=0.5 * (1 - augs_coef))
        self.train_dataset.replace_bg_prob = self.config.replace_bg_prob * augs_coef
        self.train_dataset.glob_crop_prob = self.config.glob_crop_prob
        self.train_dataset.pet_crop_prob = self.config.pet_crop_prob
        self.epoch_count += 1

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.cpus,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_datasets,
            batch_size=self.config.batch_size,
            num_workers=self.config.cpus
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
        choose_categories,
        image_size=None,
        augmentations=None,
        channels_first=True,
        replace_bg_prob=0.0,
        glob_crop_prob=0.0,
        pet_crop_prob=0.0,
        old_domain_prob=0.0,
    ):
        super().__init__()

        self.data_path = data_path
        self.resize = (image_size, image_size) if image_size is not None else None
        self.augmentations_before_crop = augmentations
        self.augmentations_after_crop = None
        self.channels_first = channels_first
        self.glob_crop_prob = glob_crop_prob
        self.pet_crop_prob = pet_crop_prob
        self.replace_bg_prob = replace_bg_prob
        self.old_domain_prob = old_domain_prob

        self.preproc = nnio.Preprocessing(
            resize=self.resize,
            channels_first=channels_first,
            dtype='float32',
            divide_by_255=True,
            # means=[0.485, 0.456, 0.406],
            # stds=[0.229, 0.224, 0.225],
        )

        self.table = table

        cat_mask = np.array(['cat' in b for b in self.table['boxes']])
        mask = None
        if choose_categories == 'cat':
            mask = cat_mask
        elif choose_categories == 'dog':
            mask = ~cat_mask
        
        if mask is not None:
            self.table = self.table[mask]

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
        target = np.array(row.Pawpularity.astype('float32'))

        # Load image
        img_path = os.path.join(self.data_path, 'train', row['Id'] + '.jpg')
        old_domain = np.random.random() < self.old_domain_prob
        if old_domain:
            img_path = os.path.join(self.data_path, 'old_petfinder', 'images')
            img_path = os.path.join(img_path, np.random.choice(os.listdir(img_path)))
            target = np.array(-100.0).astype('float32')
        image = cv2.imread(img_path)[:, :, ::-1]

        # Augment image
        if self.augmentations_before_crop is not None:
            image = self.augmentations_before_crop(
                image=image,
            )['image']

        # Add image size to features
        features = np.concatenate([
            features,
            np.array(image.shape[:2], dtype='float32') / 1000,
        ])

        # Make background mask and find region of interest
        boxes = row['boxes'].strip().split(';')[1:]
        mask = np.zeros(image.shape[:2], dtype='uint8')
        x_min_g, x_max_g, y_min_g, y_max_g = mask.shape[1], 0, mask.shape[0], 0
        x_min_pet, x_max_pet, y_min_pet, y_max_pet = None, None, None, None
        dog_detected, cat_detected = 0, 0
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
                # Find pet with max area
                if box.label in ['dog', 'cat']:
                    if (
                        x_min_pet is None
                        or
                        area(x_min_pet, x_max_pet, y_min_pet, y_max_pet) < area(x_min, x_max, y_min, y_max)
                    ):
                        x_min_pet, x_max_pet, y_min_pet, y_max_pet = x_min, x_max, y_min, y_max
                if box.label == 'dog':
                    dog_detected = 1
                if box.label == 'cat':
                    cat_detected = 1

        # Crop the biggest pet
        if x_min_pet is not None and area(x_min_pet, x_max_pet, y_min_pet, y_max_pet) > 1000:
            x_min_pet, x_max_pet, y_min_pet, y_max_pet = self.make_box_more_like_square(
                x_min_pet, x_max_pet, y_min_pet, y_max_pet,
                image
            )
            image_pet = image[y_min_pet: y_max_pet, x_min_pet: x_max_pet]
        else:
            image_pet = image

        # Replace background with white noise
        if np.random.random() < self.replace_bg_prob:
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
            image_glob = image[y_min_g: y_max_g, x_min_g: x_max_g]
        else:
            image_glob = image

        # Replace image with only pet cropped
        replace_with_cropped = np.random.random() < self.pet_crop_prob and not old_domain

        # Replace image with all pets cropped
        replace_with_glob = np.random.random() < self.glob_crop_prob and not old_domain

        # Augment images
        if self.augmentations_after_crop is not None:
            image_pet = self.augmentations_after_crop(
                image=image_pet,
            )['image']
            image_glob = self.augmentations_after_crop(
                image=image_glob,
            )['image']
            image = self.augmentations_after_crop(
                image=image,
            )['image']

        # Resize and preprocess images
        if not old_domain:
            image_pet = self.preproc(image_pet)
            image_glob = self.preproc(image_glob)
        if not replace_with_cropped:
            image = self.preproc(image)
        else:
            image = image_pet
        if replace_with_glob:
            image = image_glob
        if old_domain:
            image_pet = image
            image_glob = image

        return {
            'image': image,
            'features': features,
            'target': target,
            'dog_detected': dog_detected,
            'cat_detected': cat_detected,
            'image_pet': image_pet,
            'image_glob': image_glob,
        }


if __name__ == '__main__':
    DATA_PATH = '/home/ruslan/data/datasets_ssd/kaggle/petfinder-pawpularity-score/'
    from sklearn.model_selection import KFold
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib

    data_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    # Load boxes
    with open(os.path.join(DATA_PATH, 'boxes_train.txt')) as boxes:
        data_train['boxes'] = [l.strip() for l in boxes]

    # KFOLD
    kf = KFold(10, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(data_train)):
        print('Fold', fold + 1)
        dataset = PetDataset(
            DATA_PATH,
            data_train.iloc[val_index],
            'all',
            channels_first=False
        )

        for i in range(len(dataset)):
            image = dataset[i]['image']
            image_dog = dataset[i]['image_pet']
            print('dog', dataset[i]['dog_detected'], 'cat', dataset[i]['cat_detected'])

            fig, axes = plt.subplots(1, 2, figsize=(14, 8))
            axes[0].imshow(image)
            axes[1].imshow(image_dog)
            plt.show()
