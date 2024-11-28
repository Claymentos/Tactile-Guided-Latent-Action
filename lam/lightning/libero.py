import torch
import lightning as L
import random

from lam.datasets.libero_datasets import LiberoBatchTransform
from lam.datasets.libero.libero_utils import build_dataset


class LiberoDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 shape_meta: dict,
                 benchmark_name: str = "LIBERO_90",
                 window_size: int = 16,
                 n_demos: int = 50,
                 validation_split: float = 0.2,
                 batch_size: int = 32,
                 num_workers: int = 6):
        super().__init__()
        self.data_path = data_path
        self.benchmark_name = benchmark_name
        self.window_size = window_size
        self.n_demos = n_demos
        self.shape_meta = shape_meta
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        batch_transform = LiberoBatchTransform()

        if self.validation_split > 0:
            demo_ids = random.sample(list(range(self.n_demos)), self.n_demos)
            train_ids = demo_ids[round(self.n_demos*self.validation_split):]
            val_ids = demo_ids[:round(self.n_demos*self.validation_split)]
            self.n_demos = len(train_ids)

            self.val_dataset = build_dataset(data_prefix=self.data_path,
                                suite_name="libero",
                                benchmark_name=self.benchmark_name,
                                mode="fewshot",
                                seq_len=1,
                                frame_stack=self.window_size,
                                shape_meta=self.shape_meta,
                                n_demos=len(val_ids),
                                demo_ids=val_ids,
                                obs_seq_len=1,
                                batch_transform=batch_transform,
                                dataset_keys=[])
            
        else:
            self.val_dataset = None
            train_ids = None

        self.train_dataset = build_dataset(data_prefix=self.data_path,
                                suite_name="libero",
                                benchmark_name=self.benchmark_name,
                                mode="all",
                                seq_len=1,
                                frame_stack=self.window_size,
                                shape_meta=self.shape_meta,
                                n_demos=self.n_demos,
                                demo_ids=train_ids,
                                obs_seq_len=1,
                                batch_transform=batch_transform,
                                dataset_keys=[])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           prefetch_factor=2, 
                                           shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers, 
                                           shuffle=True)