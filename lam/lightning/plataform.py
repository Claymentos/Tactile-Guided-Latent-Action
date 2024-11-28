import torch
import lightning as L
import h5py

from sklearn.model_selection import train_test_split

from lam.datasets.plataform_datasets import AtariDataset, CoinRunDataset


class AtariDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 validation_split: float = 0.2,
                 batch_size: int = 32,
                 num_workers: int = 6):
        super().__init__()
        self.data_path = data_path
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        train_indices, val_indices = self.get_split_indices()

        self.train_dataset = AtariDataset(self.data_path, train_indices)
        self.val_dataset = AtariDataset(self.data_path, val_indices)


    def get_split_indices(self):
        assert self.validation_split < 1.0 and self.validation_split >= 0.0, "Validation split must be between 0.0 and 1.0"

        with h5py.File(self.data_path, 'r') as f:
            dataset_idx = list(f.keys())

        train_indices, val_indices = train_test_split(dataset_idx, test_size=self.validation_split, shuffle=True)

        return train_indices, val_indices


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           prefetch_factor=2, 
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers, 
                                           shuffle=True)
    

class CoinRunModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 validation_path: str,
                 window_size: int = 16,
                 batch_size: int = 32,
                 num_workers: int = 6):
        super().__init__()
        self.data_path = data_path
        self.validation_path = validation_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.train_dataset = CoinRunDataset(self.data_path, self.window_size)
        self.val_dataset = CoinRunDataset(self.validation_path, self.window_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           prefetch_factor=2, 
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers, 
                                           shuffle=True)