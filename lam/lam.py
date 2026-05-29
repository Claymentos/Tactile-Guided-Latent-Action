import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
from torch import Tensor
from typing import Dict, List

import torch.nn.functional as F
from lam.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer
from torch import Tensor
from torchvision import transforms

from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer

from lam.genie.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer
from vector_quantize_pytorch import VectorQuantize


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ActVAEVisionVAE(nn.Module):
    def __init__(self, 
                 vq,
                 vision_encoder,
                 vision_decoder,
                 action_dim=264,       
                 model_dim=512,        
                 latent_dim=512, 
                 num_latents=16,       
                 enc_layers=15,        
                 dec_layers=15,       
                 nhead=8,
                 dropout=0.1):
        super(ActVAEVisionVAE, self).__init__()
        
        self.action_dim = action_dim
        
        self.input_proj = nn.Linear(action_dim, model_dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, model_dim))
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=nhead, 
            dim_feedforward=model_dim * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        
        self.to_latent = nn.Linear(model_dim, latent_dim)
        
        self.vq = vq

        self.vision_encoder = vision_encoder
        self.vision_decoder = vision_decoder

        self.latent_proj = nn.Linear(latent_dim, model_dim)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=nhead, 
            dim_feedforward=model_dim * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        
        self.output_proj = nn.Linear(model_dim, action_dim)

    def forward(self, actions):

        B, T = actions.shape[:2]
        x = rearrange(actions, 'b t p c -> b t (p c)') 
        z_e = self.encode(x)
        z_e = self.to_latent(z_e)
        
        z_q, z, emb, indices = self.vq(z_e)
        
        actions_recon = self.decode(z_q)

        
        recon_loss = nn.functional.mse_loss(actions_recon, actions)

        codebook_commitment_loss = F.mse_loss(z, z_e.detach())

        codebook_loss =  F.mse_loss(z_e, z.detach())     

        return {
            "recon_actions": actions_recon,
            "recon_loss": recon_loss,
            "commit_loss": codebook_commitment_loss,
            "codebook_loss": codebook_loss,
            "z_q": z_q,
            "emb": emb,
            "indices": indices
        }
    
    def encode(self, x):
        return self.encoder(self.input_proj(x) + self.pos_embedding[:, :x.shape[1], :])
    
    def decode(self, z_q):
        dec_in = self.latent_proj(z_q) + self.pos_embedding[:, :z_q.shape[1], :]
        out = self.decoder(dec_in)
        actions_recon_flat = self.output_proj(out)
        actions_recon = rearrange(actions_recon_flat, 'b t (p c) -> b t p c', p=88, c=3)
        return actions_recon


class ActVAE(nn.Module):
    def __init__(self, 
                 action_dim=264,       
                 model_dim=512,        
                 latent_dim=512, 
                 num_latents=256,       
                 enc_layers=15,        
                 dec_layers=15,       
                 nhead=8,
                 dropout=0.1):
        super(ActVAE, self).__init__()
        
        self.action_dim = action_dim
        
        self.input_proj = nn.Linear(action_dim, model_dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, model_dim))
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=nhead, 
            dim_feedforward=model_dim * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        
        self.to_latent = nn.Linear(model_dim, latent_dim)
        
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True
        )
        
        self.latent_proj = nn.Linear(latent_dim, model_dim)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=nhead, 
            dim_feedforward=model_dim * 4, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        
        self.output_proj = nn.Linear(model_dim, action_dim)

    def forward(self, actions):

        B, T = actions.shape[:2]
        x = rearrange(actions, 'b t p c -> b t (p c)') 
        z_e = self.encode(x)
        z_e = self.to_latent(z_e)
        
        z_q, z, emb, indices = self.vq(z_e)
        
        actions_recon = self.decode(z_q)

        
        recon_loss = nn.functional.mse_loss(actions_recon, actions)

        codebook_commitment_loss = F.mse_loss(z, z_e.detach())

        codebook_loss =  F.mse_loss(z_e, z.detach())     

        return {
            "recon_actions": actions_recon,
            "recon_loss": recon_loss,
            "commit_loss": codebook_commitment_loss,
            "codebook_loss": codebook_loss,
            "z_q": z_q,
            "emb": emb,
            "indices": indices
        }
    
    def encode(self, x):
        return self.encoder(self.input_proj(x) + self.pos_embedding[:, :x.shape[1], :])
    
    def decode(self, z_q):
        dec_in = self.latent_proj(z_q) + self.pos_embedding[:, :z_q.shape[1], :]
        out = self.decoder(dec_in)
        actions_recon_flat = self.output_proj(out)
        actions_recon = rearrange(actions_recon_flat, 'b t (p c) -> b t p c', p=88, c=3)
        return actions_recon


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