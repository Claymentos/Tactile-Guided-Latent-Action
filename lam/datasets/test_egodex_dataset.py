from lam.datasets.egodex.dataset import EgoDexDataset
from torch.utils.data import DataLoader

data_path = '/home/cl-ment-prigent/egodex'
dataset = EgoDexDataset(data_dir=data_path, len_seq=5)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

try: 
    batch = next(iter(dataloader))
    video = batch['video']
    actions = batch['actions']

    print(f"Video batch shape:{video.shape}")
    print(f"Actions batch shape: {actions.shape}")

except Exception as e:
    print(f"Error during data loading: {e}")