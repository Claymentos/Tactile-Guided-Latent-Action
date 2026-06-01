import torch
from datasets.egodex.dataset import EgoDexDataset
import matplotlib.pyplot as plt
from einops import rearrange

REPO_DIR = "/home/cl-ment-prigent/LAM/dinov3"

dino = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights='/home/cl-ment-prigent/LAM/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')

DATASET_DIR = "/home/cl-ment-prigent/egodex/"

dataset = EgoDexDataset(root_folder=DATASET_DIR, 
                                max_context_len=16, 
                                num_context=4, 
                                patch_size=14, 
                                img_size=224, 
                                backbone_name='dinov2', 
                                train=False, 
                                evaluate=True, 
                                keys='all', 
                                var_time=False)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,   # mettre 0 pour débugger plus facilement
)

curr_frames, actions, rel_t, heatmaps, all_valid_kp, metadata = next(iter(loader))

# sample = curr_frames[0]
# fig, axes = plt.subplots(1, sample.shape[0], figsize=(20, 4))

# for t in range(sample.shape[0]):
#     img = sample[t].permute(1, 2, 0).cpu().numpy()  # [H,W,C]

#     axes[t].imshow(img)
#     axes[t].set_title(f"t={t}")
#     axes[t].axis("off")

# plt.tight_layout()
# plt.show()

B,T,C,H,W = curr_frames.shape
curr_frames = rearrange(curr_frames, 'b t c h w -> (b t) c h w')
print(f"curr_frames shape: {curr_frames.shape}")


with torch.no_grad():

    out = dino.forward_features(curr_frames)

print(f"out patch tokens norm: {out['x_norm_patchtokens'].shape}")

