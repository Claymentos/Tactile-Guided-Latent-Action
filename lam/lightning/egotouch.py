import lightning as L
from torch.utils.data import DataLoader

from lam.datasets.egotouch.dataset import EgoTouchDataset


class EgoTouchDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 camera: str = 'chest',
                 window_size: int = 16,
                 num_context: int = 4,
                 patch_size: int = 16,
                 img_size: int = 224,
                 aug: bool = False,
                 backbone_name: str = 'dinov3',
                 normalize: bool = True,
                 tactile_repr: str = 'grid',
                 mano_uv_path: str = None,
                 val_split: str = 'val',
                 batch_size: int = 32,
                 num_workers: int = 6):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.aug = aug
        self.dataset_args = dict(root_folder=data_path, max_context_len=window_size, num_context=num_context,
                                 patch_size=patch_size, img_size=img_size, backbone_name=backbone_name, cameras=(camera,),
                                 normalize=normalize, tactile_repr=tactile_repr, mano_uv_path=mano_uv_path)

    def setup(self, stage):
        self.train_dataset = EgoTouchDataset(train=True, aug=self.aug, **self.dataset_args)
        self.val_dataset = EgoTouchDataset(train=False, split=self.val_split, **self.dataset_args)

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
