import torch

from torch import nn
from einops import rearrange

from lam.feature_extractor.base import Extractor

class DinoV2(Extractor):
    """
    Returns features extracted with DinoV2 for each frame in the video.
    """
    def __init__(self,
                 model_id,
                 image_size,
                 patch_size,
                 model_dim,
                 channels=3):
        super().__init__()
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', model_id)

        # Freeze the image encoder parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False


    def forward(self, x):
        """
        arguments: x: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        returns: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        """
        self.feature_extractor.eval()

        with torch.no_grad():
            b, t, c, h, w = x.shape
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            features = self.feature_extractor(x, is_training=True)["x_prenorm"][:, self.feature_extractor.num_register_tokens + 1 :]
            features = rearrange(features, '(b t) n d -> b t n d', b=b, t=t)

        return features
    
    def _post_process(self, x):
        """
        Post-process the output of the decoder before computing the loss.
        arguments: x: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        returns: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        """
        return x
    
    def compute_loss(self, logits, images, features):
        """
        Compute the reconstruction loss between the decoder output and input.
        MSE loss between the reconstructed features and the original features.

        arguments:
        logits: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        images: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        features: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        returns: torch.Tensor, scalar loss value
                 torch.Tensor | None, shape (batch_size, seq_len, channels, height, width) - reconstructed images
        """
        recon_loss = nn.functional.mse_loss(logits, features)
        return recon_loss, None