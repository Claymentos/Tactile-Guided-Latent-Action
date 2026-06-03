import torch
from lam.datasets.egodex.dataset import EgoDexDataset
import matplotlib.pyplot as plt
from einops import rearrange
from sklearn.decomposition import PCA


def extract_pca(feats, grid_w, grid_h, n_components=3):
    B,N,D = feats.shape

    features_flat = feats.reshape(-1, D).cpu().numpy()    
    pca = PCA(n_components=n_components)
    pca.fit(features_flat)
    pca_features = pca.transform(features_flat)

    pca_min = pca_features.min(axis=0)
    pca_max = pca_features.max(axis=0)
    pca_features = (pca_features - pca_min) / (pca_max - pca_min + 1e-5)

    pca_features = torch.from_numpy(pca_features)
    pca_features = pca_features.view(B, grid_h, grid_w, n_components).permute(0, 3, 1, 2)
    return pca_features


def main():

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

    print(actions.shape)


    B,T,C,H,W = curr_frames.shape
    curr_frames = rearrange(curr_frames, 'b t c h w -> (b t) c h w')
    print(f"curr_frames shape: {curr_frames.shape}")


    with torch.no_grad():

        out = dino.forward_features(curr_frames)

    print(f"out patch tokens norm: {out['x_norm_patchtokens'].shape}")

    n_layers = 12

    with torch.inference_mode():
        patch_features = dino.forward_features(curr_frames)["x_norm_patchtokens"]  # [B*T, P, D]
        print(f"patch features shape: {patch_features.shape}")



    grid_h = 14
    grid_w = 24

    # PCA sur tous les patchs du batch
    pca_features = extract_pca(
        patch_features.cpu(),
        grid_w=grid_w,
        grid_h=grid_h,
    )

    # Premier élément du batch aplati
    img = curr_frames[0].permute(1, 2, 0).cpu().numpy()

    # [3,14,24] -> [14,24,3]
    pca_rgb = pca_features[0].permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(pca_rgb)
    axes[1].set_title("DINOv3 PCA features")
    axes[1].axis("off")

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()