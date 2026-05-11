from lam.datasets.libero_datasets import SequenceDataset
from torch.utils.data import DataLoader


dataset_path = '/home/cl-ment-prigent/LIBERO/libero/datasets'
dataset = SequenceDataset(hdf5_path=dataset_path, obs_keys=['rgb'], dataset_keys=['actions'])
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

try: 
    batch = next(iter(dataloader))
    video = batch['video']
    actions = batch['actions']

    print(f"Video batch shape:{video.shape}")
    print(f"Actions batch shape: {actions.shape}")

except Exception as e:
    print(f"Error during data loading: {e}")
