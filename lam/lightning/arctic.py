import lightning as L
from torch.utils.data import DataLoader

from lam.datasets.arctic.dataset import ArcticDataset


class ArcticDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 contact_path: str,
                 mano_uv_path: str = None,
                 window_size: int = 16,
                 num_context: int = 4,
                 patch_size: int = 16,
                 img_size: int = 224,
                 aug: bool = False,
                 backbone_name: str = 'dinov3',
                 normalize: bool = True,
                 contact_threshold: float = 0.01,
                 contact_repr: str = 'uv',
                 protocol: str = 'p1',
                 val_split: str = 'val',
                 batch_size: int = 32,
                 num_workers: int = 6):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.aug = aug
        self.dataset_args = dict(root_folder=data_path, contact_root=contact_path, mano_uv_path=mano_uv_path,
                                 max_context_len=window_size, num_context=num_context, patch_size=patch_size,
                                 img_size=img_size, backbone_name=backbone_name, normalize=normalize,
                                 contact_threshold=contact_threshold, contact_repr=contact_repr, protocol=protocol)

    def setup(self, stage):
        self.train_dataset = ArcticDataset(train=True, aug=self.aug, **self.dataset_args)
        self.val_dataset = ArcticDataset(train=False, split=self.val_split, **self.dataset_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.num_workers > 0,
                          drop_last=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
