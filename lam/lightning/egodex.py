import torch
import lightning as L
from torch.utils.data import DataLoader, random_split

from lam.datasets.egodex.dataset import EgoDexDataset

class EgoDexDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 window_size: int = 16,
                 validation_split: float = 0.2,
                 batch_size: int = 32,
                 num_workers: int = 6):
        super().__init__()
        self.data_path = data_path
        self.window_size = window_size
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        dataset = EgoDexDataset(root_folder=self.data_path, 
                                max_context_len=self.window_size, 
                                num_context=4, 
                                patch_size=14, 
                                img_size=224, 
                                backbone_name='dinov2', 
                                train=True, 
                                evaluate=False, 
                                keys='all', 
                                var_time=False)

        
        if self.validation_split > 0:
            validation_set_size = int(len(dataset) * self.validation_split)
            training_set_size = len(dataset) - validation_set_size
            self.train_dataset, self.val_dataset = random_split(dataset, [training_set_size, validation_set_size])

        else:
            self.train_dataset, self.val_dataset = dataset, None


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          prefetch_factor=2, 
                          shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          persistent_workers=False,
                          shuffle=True)