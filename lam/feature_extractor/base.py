import torch

from torch import nn
from einops.layers.torch import Rearrange
from abc import ABC, abstractmethod


class Extractor(ABC, torch.nn.Module):
    '''
    Lightweight interface for feature extractor models
    '''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def _post_process(self, indices):
        pass

    @abstractmethod
    def compute_loss(self):
        pass


class Identity(Extractor):
    """
    Returns patchified video projected to model_dim, normalized with layer normalization.
    """
    def __init__(self,
                 image_size,
                 patch_size,
                 model_dim,
                 channels=3,
                 loss_fn=nn.MSELoss()):
        super().__init__()
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.to_image = nn.Sequential(
            nn.Linear(model_dim, channels * patch_height * patch_width),
            Rearrange('b t (nh nw) (p1 p2 c) -> b t c (nh p1) (nw p2)',
                      p1=patch_size, p2=patch_width, c=channels, nh=image_height//patch_height, nw=image_width//patch_width),
        )

        self.loss_fn = loss_fn

    def forward(self, x):
        """
        arguments: x: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        returns: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        """
        return self.to_patch_embedding(x)
    
    def _post_process(self, x):
        """
        Post-process the output of the decoder before computing the loss.
        In the case of Identity, it simply reconstructs the image from the patches and add a sigmoid activation, computing the loss in image space.
        arguments: x: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        returns: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        """
        image = self.to_image(x)
        return nn.functional.sigmoid(image)
    
    def compute_loss(self, logits, images, features):
        """
        Compute the reconstruction loss between the decoder output and input.
        For this identity encoder, it is simply the MSE loss between the reconstruction and the input images.
        When working in feature space, it will be the MSE loss between the reconstruction and the input features.

        arguments:
        logits: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        images: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        features: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        returns: torch.Tensor, scalar loss value
                 torch.Tensor, shape (batch_size, seq_len, channels, height, width) - reconstructed images
        """
        reconstruction = self._post_process(logits)
        recon_loss = self.loss_fn(reconstruction, images)
        return recon_loss, reconstruction