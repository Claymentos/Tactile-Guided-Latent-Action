import torch

from torch import nn
from einops import rearrange
from omegaconf import OmegaConf
from hydra.utils import call

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
                 channels=3,
                 decoder_path=None,
                 loss_fn=nn.MSELoss()):
        super().__init__()
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', model_id)
        self.loss_fn = loss_fn

        # Freeze the image encoder parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.load_decoder(decoder_path)

    def load_decoder(self, decoder_path):
        """
        Load DinoV2 decoder for image visualization and metric computation
        """
        if decoder_path is not None:
            model_config_path = "/".join(decoder_path.split("/")[0:-1]) + f"/model_config.yaml"
            model_config = OmegaConf.load(model_config_path)

            model_config.lightning_module._target_ = model_config.lightning_module._target_ + ".load_from_checkpoint"
            model_config.lightning_module.checkpoint_path = decoder_path
            lightning_module = call(model_config.lightning_module)
            
            self.decoder = lightning_module.model.image_decoder
            self.to_image = lightning_module.model.to_image

            self.decoder.eval()
            self.to_image.eval()

            for param in self.decoder.parameters():
                param.requires_grad = False

            for param in self.to_image.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        arguments: x: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        returns: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        """
        self.feature_extractor.eval()

        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        features = self.feature_extractor(x, is_training=True)["x_prenorm"][:, self.feature_extractor.num_register_tokens + 1 :]
        features = rearrange(features, '(b t) n d -> b t n d', b=b, t=t)

        return features
    
    def _post_process(self, logits):
        """
        Post-process the output of the decoder, turning it into an image.
        arguments: logits: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        returns: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        """
        if not hasattr(self, 'decoder'):
            return None
        
        self.decoder.eval()
        self.to_image.eval()

        b, t, n, f = logits.shape
        logits = rearrange(logits, 'b t n f -> (b t) n f')
        for block in self.decoder.blocks:
            logits = block(logits)
        logits = rearrange(logits, '(b t) n f -> b t n f', b=b, t=t)
        recon_image = self.to_image(logits)

        return recon_image
    
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
        recon_loss = self.loss_fn(logits, features)
        recon_image = self._post_process(logits)

        return recon_loss, recon_image