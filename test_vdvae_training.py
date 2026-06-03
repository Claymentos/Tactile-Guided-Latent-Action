import torch
import torch.nn as nn
from lam.lam import VDVAE, ActVAE
from lam.lightning.lam import VDVAETraining

# Mocking ActVAE and VDVAE dependencies if needed, or using them if they are simple enough
# ActVAE needs action_dim, model_dim, etc.
act_vae = ActVAE(action_dim=264, model_dim=128, latent_dim=128, num_latents=16, enc_layers=2, dec_layers=2)

# VDVAE needs dino_repo, dino_weights, etc.
# Since I cannot load real weights easily here, I'll mock the feature extractor if it fails
# But let's try to see if we can just mock the whole VDVAE forward if needed.
# Actually, I want to test MY changes in VDVAE too.

class MockFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward_features(self, x):
        return {"x_norm_patchtokens": torch.randn(x.shape[0], 448, 768)}
    def eval(self): return self

# I'll monkeypatch torch.hub.load to return my mock
import torch.hub
torch.hub.load = lambda *args, **kwargs: MockFeatureExtractor()

model = VDVAE(
    dino_repo="mock",
    dino_weights="mock",
    act_vae=act_vae,
    image_size=224,
    num_latents=16,
    model_dim=128,
    latent_dim=128,
    act_latent_dim=128,
    env_codebook_size=16,
    env_latent_dim=128,
    nhead=4,
    enc_layers=2,
    dec_layers=2,
    lambda_rec=1.0,
    lambda_vq=1.0,
    lambda_con=1.0,
    lambda_reg=1.0,
    commitment_weight=0.25
)

training_module = VDVAETraining(model=model)

# Create dummy batch matching EgoDexDataset output
# frames: (B, T+1, C, H, W)
# actions: (B, T, 88, 3)
B, T_plus_1 = 2, 5
frames = torch.randn(B, T_plus_1, 3, 224, 392)
actions = torch.randn(B, T_plus_1 - 1, 88, 3)
rel_t = torch.ones(B, T_plus_1 - 1)
batch = (frames, actions, rel_t, None, None, {})

print("Running training_step...")
loss = training_module.training_step(batch, 0)
print(f"Loss: {loss.item()}")

print("Running validation_step...")
val_loss = training_module.validation_step(batch, 0)
print(f"Val Loss: {val_loss.item()}")

print("Testing configure_optimizers...")
optimizers, schedulers = training_module.configure_optimizers()
print(f"Optimizers: {len(optimizers)}")
print(f"Schedulers: {len(schedulers)}")

print("Verification complete!")
