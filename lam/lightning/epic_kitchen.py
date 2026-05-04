import torch
import lightning as L

from torch.utils.data import DataLoader, random_split


class EpicKitchenDataModule(L.LightningDataModule):
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
        dataset = build_dataset(data_prefix=self.data_path,
                                suite_name="libero",
                                benchmark_name=self.benchmark_name,
                                mode="all",
                                seq_len=1,
                                frame_stack=self.window_size,
                                shape_meta=self.shape_meta,
                                n_demos=self.n_demos,
                                obs_seq_len=1,
                                dataset_keys=[])
        
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