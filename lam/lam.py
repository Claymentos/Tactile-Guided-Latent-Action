import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
from torch import Tensor
from typing import Dict


from lam.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer
from torch import Tensor


class LatentActionModel(nn.Module):
    """
    Latent action VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout
        )

        self.mu_record = None

    def encode(self, videos: Tensor) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = self.action_prompt.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        # Encode
        z = self.encoder(padded_patches)  # (B, T, 1+N, E)
        # Get latent action for all future frames
        z = z[:, 1:, 0]  # (B, T-1, 1, E)

        # VAE
        z = z.reshape(B * (T - 1), self.model_dim)
        moments = self.fc(z)
        z_mu, z_var = torch.chunk(moments, 2, dim=1)
        # Reparameterization
        if not self.training:
            z_rep = z_mu
        else:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
        z_rep = z_rep.reshape(B, T - 1, 1, self.latent_dim)

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)

        return {
            "patches": patches,
            "z_rep": z_rep,
            "z_mu": z_mu,
            "z_var": z_var
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VAE
        H, W = batch["videos"].shape[2:4]
        outputs = self.encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_rep"])
        video_action_patches = video_patches + action_patches

        del outputs["patches"]

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = F.sigmoid(video_recon)
        outputs.update(
            {
                "recon": unpatchify(video_recon, self.patch_size, H, W)
            }
        )
        return outputs


class LAM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 vq,
                 feature_extractor,
                 perceptual_regularizer=None,
                 num_register_tokens=0,
                 register_dropout_mean=0.0,
                 register_dropout_std=0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
        self.feature_extractor = feature_extractor
        self.perceptual_regularizer = perceptual_regularizer

        self.num_register_tokens = num_register_tokens
        self.register_dropout_mean = register_dropout_mean
        self.register_dropout_std = register_dropout_std

        if num_register_tokens:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, encoder.spatial_pos_embedding.weight.shape[1]))
            nn.init.normal_(self.register_tokens, std=1e-6)


    def forward(self, x):
        '''
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        '''
        with torch.no_grad():
            features = self.feature_extractor(x)                                            # Extract image features or simply patchfy and project if working in image space

        B, T, P, F = features.shape
        attention_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)         # Attention mask for the patches (False means keep, True means mask)

        if hasattr(self, 'register_tokens'):
            p = torch.clamp(torch.normal(self.register_dropout_mean, self.register_dropout_std, size=(B,), device=x.device), 0.0, 1.0).unsqueeze(1)
            register_attention_mask = (torch.rand((B, self.num_register_tokens), device=x.device) < p)

            attention_mask = torch.cat([register_attention_mask, attention_mask], dim=-1)
            register_tokens = repeat(self.register_tokens, '1 n d -> b t n d', b=B, t=T)
        else:
            register_tokens, register_readouts = None, None

        attention_mask = repeat(attention_mask, 'b p -> b t p', b=B, t=T)

        logits, readout = self.encode(features, registers=register_tokens, attention_mask=attention_mask)
        readout = readout[:, 1:, :]                                                                                                             # Ignore the first action token, it doesn't represent any transition
        action_tokens, indices, commit_loss = self.vq.quantize(readout)

        if register_tokens is not None:
            register_readouts = logits[:, 1:, :self.num_register_tokens, :]                                                                     # Ignore the first register token, it doesn't represent any transition

        logits = self.decode(features[:, :-1], actions=action_tokens, registers=register_readouts, attention_mask=attention_mask[:, :-1])       # Ignore the last frame, since there is no target for it
        logits = logits[:, :, self.num_register_tokens:, :]                                                                                     # Remove the register tokens from the decoder output
        recon_loss, recon_video = self.feature_extractor.compute_loss(logits, x[:, 1:], features[:, 1:])                                        # The first frame of x is not used as a target
        
        if self.perceptual_regularizer is not None:
            with torch.no_grad():
                ground_truth_features = self.perceptual_regularizer(x[:, 1:])
            reconstruction_features = self.perceptual_regularizer(recon_video)
            perceptual_loss, _ = self.perceptual_regularizer.compute_loss(ground_truth_features, None, reconstruction_features)
        else:
            perceptual_loss = 0.0

        return dict(recon_loss=recon_loss, perceptual_loss=perceptual_loss, commit_loss=commit_loss, indices=indices, action_tokens=action_tokens, features=features, recon_features=logits, recon_video=recon_video)

    def encode(self, x, registers=None, attention_mask=None):
        return self.encoder(x, registers=registers, attention_mask=attention_mask)


    def decode(self, x, actions, registers=None, attention_mask=None):
        return self.decoder(x, actions, registers=registers, attention_mask=attention_mask)
    
    def get_latent_actions(self, x):
        '''
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        '''
        features = self.feature_extractor(x)                                            # Extract image features or simply patchfy and project if working in image space

        logits, readout = self.encode(features)
        readout = readout[:, 1:, :]                                                 # Ignore the first action token, it doesn't represent any transition
        action_tokens, indices, commit_loss = self.vq.quantize(readout)

        return action_tokens, indices


class ChunkLAM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 vq,
                 model_dim,
                 patch_size,
                 image_size,
                 channels=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq

        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.to_image = nn.Sequential(
            nn.Linear(model_dim, channels * patch_height * patch_width),
            Rearrange('b t (nh nw) (p1 p2 c) -> b t c (nh p1) (nw p2)',
                      p1=patch_size, p2=patch_width, c=channels, nh=image_height//patch_height, nw=image_width//patch_width),
        )

    def forward(self, x):
        '''
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        '''
        logits, readout = self.encode(x)
        readout = readout[:, -1:, :]                                                # Take only the last action token, which will represent the whole chunk
        action_tokens, indices, commit_loss = self.vq.quantize(readout)

        decoder_input = x[:, :1, :, :, :].repeat(1, x.shape[1]-1, 1, 1, 1)            # The decoder only receives the first frame, repeated for the whole chunk
        action_tokens = action_tokens.repeat(1, x.shape[1]-1, 1)                      # The action token is also repeated for the whole chunk

        recon_video = self.decode(decoder_input, action_tokens)                 
        recon_loss = nn.functional.mse_loss(recon_video, x[:, 1:, :, :, :])         # The first frame of x is not used as a target
        
        return recon_loss, commit_loss, recon_video, indices

        
    def encode(self, x):
        return self.encoder(x)
    

    def decode(self, x, actions):
        logits = self.decoder(x, actions)

        reconstruction = self.to_image(logits)
        reconstruction = nn.functional.sigmoid(reconstruction)
     
        return reconstruction


class MultiviewLAM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 vq,
                 model_dim,
                 patch_size,
                 image_size,
                 channels=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq

        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.to_image = nn.Sequential(
            nn.Linear(model_dim, channels * patch_height * patch_width),
            Rearrange('b t (nh nw) (p1 p2 c) -> b t c (nh p1) (nw p2)',
                      p1=patch_size, p2=patch_width, c=channels, nh=image_height//patch_height, nw=image_width//patch_width),
        )

    def forward(self, source_view, target_view):
        '''
        arguments:
        source_view: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        target_view: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        '''
        logits, readout = self.encode(source_view)
        readout = readout[:, 1:, :]                                                      # Ignore the first action token, it doesn't represent any transition
        action_tokens, indices, commit_loss = self.vq.quantize(readout)

        recon_video = self.decode(target_view[:, :-1, :, :, :], action_tokens)           # Ignore the last frame, since there is no target for it
        recon_loss = nn.functional.mse_loss(recon_video, target_view[:, 1:, :, :, :])    # The first frame of x is not used as a target
        
        return recon_loss, commit_loss, recon_video, indices

        
    def encode(self, x):
        return self.encoder(x)
    

    def decode(self, x, actions):
        logits = self.decoder(x, actions)

        reconstruction = self.to_image(logits)
        reconstruction = nn.functional.sigmoid(reconstruction)
     
        return reconstruction