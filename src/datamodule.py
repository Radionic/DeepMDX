import pytorch_lightning as pl
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from loguru import logger

try:
    from src import dataset
except ModuleNotFoundError:
    import dataset


class DeepMDXDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=4,
                 num_workers=12,
                 sr=44100,
                 hop_length=1024,
                 n_fft=2048,
                 reduction_rate=0.0,
                 reduction_level=0.2,
                 crop_size=256,
                 patches=16,
                 mixup_rate=0.0,
                 mixup_alpha=1.0,
                 dataset_dir='data',
                 split_mode='random',
                 val_rate=0.2,
                 val_filelist=None,
                 val_crop_size=256,
                 **kwself):
        super().__init__()
        
        split_choice = ['random', 'subdirs']
        if split_mode not in split_choice:
            raise ValueError(f"split_mode must be one of {split_choice}")
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hop_length = hop_length
        self.sr = sr
        self.n_fft = n_fft
        self.reduction_rate = reduction_rate
        self.reduction_level = reduction_level
        self.crop_size = crop_size
        self.patches = patches
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.dataset_dir = dataset_dir
        self.split_mode = split_mode
        self.val_rate = val_rate
        self.val_filelist = val_filelist
        self.val_crop_size = val_crop_size
        if self.val_filelist is not None:
            with open(self.val_filelist, 'r', encoding='utf-8') as f:
                self.val_filelist = json.load(f)
        else:
            self.val_filelist = []
        
        self.train_filelist, self.val_filelist = dataset.train_val_split(
            dataset_dir=dataset_dir,
            split_mode=split_mode,
            val_rate= val_rate,
            val_filelist=self.val_filelist
        )
    
    def setup(self, stage: str):
        bins = self.n_fft // 2 + 1
        freq_to_bin = 2 * bins / self.sr
        unstable_bins = int(200 * freq_to_bin)
        stable_bins = int(22050 * freq_to_bin)
        reduction_weight = np.concatenate([
            np.linspace(0, 1, unstable_bins, dtype=np.float32)[:, None],
            np.linspace(1, 0, stable_bins - unstable_bins, dtype=np.float32)[:, None],
            np.zeros((bins - stable_bins, 1), dtype=np.float32),
        ], axis=0) * self.reduction_level
        if stage == 'fit':
            logger.info("Loading training dataset:")
            training_set = dataset.make_training_set(
                filelist=self.train_filelist,
                sr=self.sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            self.train_ds = dataset.VocalRemoverTrainingSet(
                training_set * self.patches,
                cropsize=self.crop_size,
                reduction_rate=self.reduction_rate,
                reduction_weight=reduction_weight,
                mixup_rate=self.mixup_rate,
                mixup_alpha=self.mixup_alpha
            )
            logger.info("Loading validation dataset:")
            patch_list = dataset.make_validation_set(
                filelist=self.val_filelist,
                cropsize=self.val_crop_size,
                sr=self.sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                offset=64
            )
            self.val_ds = dataset.VocalRemoverValidationSet(
                patch_list=patch_list
            )
            
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )