from dataset import EgoDexDataset
import torch


dataset = EgoDexDataset(
    root_folder="/home/cl-ment-prigent/egodex",   # <-- à modifier
    max_context_len=90,
    num_context=4,
    patch_size=14,
    img_size=224,
    backbone_name='dinov2',
    train=False,
    evaluate=False,
    keys='all',
    var_time=False,
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,   # mettre 0 pour débugger plus facilement
)

batch = next(iter(loader))
print(f"len batch: {len(batch)}")