from dataset import EgoTouchDataset
import torch


dataset = EgoTouchDataset(
    root_folder="/home/cl-ment-prigent/EgoTouch",   # <-- à modifier
    max_context_len=90,
    num_context=4,
    patch_size=14,
    img_size=224,
    backbone_name='dinov2',
    train=False,
    evaluate=False,
    cameras='chest',
    val_ratio=0.1,
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,   # mettre 0 pour débugger plus facilement
)

batch = next(iter(loader))
print(f"len batch: {len(batch)}")