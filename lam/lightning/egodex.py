import torch
import lightning as L
from torch.utils.data import DataLoader, random_split

from lam.datasets.egodex.dataset import EgoDexDataset, EgoDexActionsOnlyDataset

class EgoDexActionOnlyDataModule(L.LightningDataModule):
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

        dataset = EgoDexActionsOnlyDataset(root_folder=self.data_path,
                                          max_context_len=self.window_size,
                                          num_context=4,
                                          train=True,
                                          evaluate=False,
                                          keys='all')
        
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

class EgoDexDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 window_size: int = 16,
                 validation_split: float = 0.2,
                 batch_size: int = 32,
                 num_workers: int = 6,
                 num_context: int = 4,
                 patch_size: int = 14,
                 img_size: int = 224,
                 backbone_name: str = 'dinov2',
                 return_heatmaps: bool = True,
                 seed: int = 42,
                 return_motion_chunks: bool = False,
                 chunk_len: int = 16,
                 load_frames: bool = True):
        """num_context transitions par clip (num_context + 1 frames espacées de window_size / num_context) ;
        patch_size 14 -> images 224 x 392 (DINOv2), 16 -> 224 x 384 (DINOv3).
        return_motion_chunks, chunk_len, load_frames : voir EgoDexDataset (chunks de mouvement pour la loss MCP)."""
        super().__init__()
        self.data_path = data_path
        self.window_size = window_size
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_context = num_context
        self.patch_size = patch_size
        self.img_size = img_size
        self.backbone_name = backbone_name
        self.return_heatmaps = return_heatmaps
        self.seed = seed
        self.return_motion_chunks, self.chunk_len, self.load_frames = return_motion_chunks, chunk_len, load_frames

    def setup(self, stage):
        dataset = EgoDexDataset(root_folder=self.data_path, 
                                max_context_len=self.window_size, 
                                num_context=self.num_context, 
                                patch_size=self.patch_size, 
                                img_size=self.img_size, 
                                backbone_name=self.backbone_name, 
                                train=True, 
                                evaluate=False, 
                                keys='all', 
                                var_time=False,
                                return_heatmaps=self.return_heatmaps,
                                return_motion_chunks=self.return_motion_chunks,
                                chunk_len=self.chunk_len,
                                load_frames=self.load_frames)
        self.num_tasks = len(dataset.all_tasks)

        
        if self.validation_split > 0:
            validation_set_size = int(len(dataset) * self.validation_split)
            training_set_size = len(dataset) - validation_set_size
            self.train_dataset, self.val_dataset = random_split(dataset, [training_set_size, validation_set_size], generator=torch.Generator().manual_seed(self.seed))

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