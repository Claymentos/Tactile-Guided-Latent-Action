import torch
import lightning as L

from lam.datasets.rlds_datasets import RLDSBatchTransform, RLDSDataset, EpisodicRLDSDataset


class RLDSModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir,
            batch_size: int = 32,
            validation: bool = False,
            image_size: int = 224,
            shuffle_buffer_size: int = 100_000,
            episodic: bool = False,
            image_aug: bool = False,
            window_size: int = 1,
            data_mix: str = "droid",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.default_image_resolution = (3, image_size, image_size)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.episodic = episodic
        self.image_aug = image_aug
        self.data_mix = data_mix
        self.batch_size = batch_size
        self.validation = validation
        self.window_size = window_size

    def build_dataset(self, train=True):
        batch_transform = RLDSBatchTransform()

        cls = RLDSDataset if not self.episodic else EpisodicRLDSDataset

        dataset = cls(
            self.data_dir,
            self.data_mix,
            batch_transform,
            resize_resolution=self.default_image_resolution[1:],
            shuffle_buffer_size=self.shuffle_buffer_size,
            train=train,
            image_aug=self.image_aug,
            window_size=self.window_size,
        )

        return dataset

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.build_dataset(train=True),
            batch_size=self.batch_size,
            num_workers=0
        )

    def val_dataloader(self):
        if not self.validation:
            return []
        
        return torch.utils.data.DataLoader(
            self.build_dataset(train=False),
            batch_size=self.batch_size,
            num_workers=0
        )