import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Dict, List, Optional

import torch.nn.functional as F
from lam.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, SpatioBlock, SpatioTemporalBlock
from torch import Tensor
from torchvision import transforms
from lam.st import Transformer
from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer
from lam.genie.modules.blocks import VectorQuantizer                                                    
from vector_quantize_pytorch import VectorQuantize
from lam.lightning.lam import ActVAETraining


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class SigLIPContrastiveLoss(nn.Module): 
    def __init__(self, init_temperature: float = 0.07, init_bias: float = -10.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())
        self.bias = nn.Parameter(torch.tensor(init_bias))
 
    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-4)
 
    def forward(
        self,
        z_visual: torch.Tensor,  # (N, D)
        z_action: torch.Tensor,  # (N, D)
    ) -> torch.Tensor:
        N = z_visual.size(0)
        z_v = F.normalize(z_visual, dim=-1)
        z_a = F.normalize(z_action, dim=-1)
 
        # Matrice de similarité (N, N)
        sim = torch.matmul(z_v, z_a.t()) / self.temperature + self.bias
 
        # Labels : 1 sur la diagonale (paires positives), 0 ailleurs
        labels = torch.eye(N, device=z_visual.device)
 
        # BCE indépendante sur chaque paire (formulation SigLIP)
        return F.binary_cross_entropy_with_logits(sim, labels)
    

class InverseDynamicEncoder(nn.Module):
    def __init__(self,
                 model_dim: int,
                 n_layers: int,
                 n_heads: int,
                 act_latent_dim: int,
                 env_latent_dim: int,
                 dropout: float):
        
        super().__init__()
        self.readout_token = nn.Parameter(torch.randn(1, 1, 1, model_dim))
        nn.init.normal(self.readout_token, std=1e-6)

        self.layers = nn.ModuleList([
            SpatioTemporalBlock(model_dim, n_heads, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(model_dim)
        self.proj_action = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, act_latent_dim)
        )
        self.proj_env = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, env_latent_dim)
        ) 

    def forward(self, f_t: torch.Tensor, f_t_H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B,P,D = f_t.shape
        x = torch.stack([f_t, f_t_H], dim=1)
        readout = rearrange(self.readout_token, '1 1 1 d -> b t 1 d', b=B, t=2)
        x = torch.cat([readout, x], dim=2)
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        readout_out = x[:, -1, 0, :]
        zv_a = self.proj_action(readout_out)
        zv_e = self.proj_env(readout_out)
        return zv_a, zv_e


class ForwardDynamicsDecoder(nn.Module):
    def __init__(self,
                 model_dim: int,
                 n_layers: int,
                 n_heads: int,
                 act_latent_dim: int,
                 env_latent_dim: int,
                 dropout: float):
        
        super().__init__()
        self.act_proj = nn.Linear(act_latent_dim, model_dim)
        self.env_proj = nn.Linear(env_latent_dim, model_dim)

        self.layers = nn.ModuleList([
            SpatioBlock(model_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, f_t: torch.Tensor, z_qa: torch.Tensor, z_qi: torch.Tensor):
        B,P,D = f_t.shape

        act_cond = self.act_proj(z_qa).unsqueeze(1).unsqueeze(1).expand(B, P,D)
        env_cond = self.env_proj(z_qi).unsqueeze(1).unsqueeze(1).expand(B, P,D)

        x = (f_t + act_cond + env_cond).unsqueeze(1)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x.squeeze(1)



class VDVAE(nn.Module):
    def __init__(self,
                 dino_repo,
                 dino_weights,
                 act_vae_checkpoint,
                 act_vae_cfg,
                 num_latents,
                 model_dim,
                 latent_dim,
                 act_latent_dim,
                 env_codebook_size,
                 env_latent_dim,
                 nhead,
                 enc_layers,
                 dec_layers,
                 lambda_rec,
                 lambda_vq,
                 lambda_con,
                 lambda_reg,
                 commitment_weight):
        
        super().__init__()
        act_vae_model = ActVAE(**act_vae_cfg)
        ckpt = ActVAETraining.load_from_checkpoint(act_vae_checkpoint, model=act_vae_model)
        self.act_vae = ckpt.model
        self.num_latents = num_latents
        self.model_dim = model_dim  
        self.latent_dim = latent_dim
        self.nhead = nhead
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.act_latent_dim = act_latent_dim

        self.inv_encoder = InverseDynamicEncoder(
            model_dim=model_dim,
            n_layers=enc_layers,
            n_heads=nhead,
            act_latent_dim=act_latent_dim,
            env_latent_dim=env_latent_dim,
            dropout=0.1
        )

        self.fwd_decoder = ForwardDynamicsDecoder(
            model_dim=model_dim,
            n_layers=dec_layers,
            n_heads=nhead,
            act_latent_dim=act_latent_dim,
            env_latent_dim=env_latent_dim,
            dropout=0.1
        )

        self.feature_extractor = torch.hub.load(dino_repo, 'dinov3_vitb16', source='local', weights=dino_weights)
        self.feature_extractor.eval().to('cuda')

        for param in self.feature_extractor.parameters():
            param.requires_grad = False


        for param in self.act_vae.parameters():
            param.requires_grad = False
        self.act_vae.eval()

        self.act_codebook : nn.Embedding = self.act_vae.vq.codebook

        assert self.act_latent_dim == self.act_codebook.embedding_dim, "The latent dimension of the action VAE must match the embedding dimension of its codebook."

        self.env_vq = VectorQuantize(
            dim=env_latent_dim,
            codebook_size=env_codebook_size,
            commitment_weight=commitment_weight,
            use_cosine_sim=False

        )

        ## Get dino dim with dummy input
        dummy_input = torch.randn(1,3,224,392).to('cuda')
        with torch.no_grad():
            dummy_features = self.feature_extractor.forward_features(dummy_input)["x_norm_patchtokens"]

        self.feature_proj = nn.Linear(dummy_features.shape[-1], model_dim)


        ## Disentanglement head
        self.proj_action = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, act_latent_dim)
        )

        self.proj_env = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, env_latent_dim)
        )

        self.contrastive_loss = SigLIPContrastiveLoss()


        self.lambda_rec = lambda_rec
        self.lambda_vq = lambda_vq
        self.lambda_con = lambda_con
        self.lambda_reg = lambda_reg
        self.commitment_weight = commitment_weight


    def extract_features(self, x):
        if x.dim()==5:
            B,T,C,H,W = x.shape
            flat = rearrange(x, 'b t c h w -> (b t) c h w')
            tokens = self.feature_extractor.forward_features(flat)["x_norm_patchtokens"]
            return rearrange(tokens, '(b t) p f -> b t p f', b=B, t=T)
        elif x.dim() == 4:
            return self.feature_extractor.forward_features(x)["x_norm_patchtokens"]
    
    # def encode(self, x, registers=None, attention_mask=None):
    #     return self.encoder(x, registers=registers, attention_mask=attention_mask)
    
    def encode_actions(self, actions: torch.Tensor):
        if actions.dim() == 3:
            actions = actions.unsqueeze(1)
        
        B, T, P, C = actions.shape
        actions_flat = rearrange(actions, 'b t p c -> b t (p c)')

        with torch.no_grad():
            z_enc = self.act_vae.encode(actions_flat)
            z_lat = self.act_vae.to_latent(z_enc)
            z_a = z_lat.mean(dim=1)
        return z_a
    
    
    def _vq_losses(
        self,
        z_pre: torch.Tensor,
        z_e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss VQ = commitment loss + codebook loss.
        Le codebook ActVAE est gelé → on calcule seulement la commitment loss
        pour z_{v,a} (pousser l'encodeur vers le codebook).
        Pour le codebook d'env (learnable), on calcule les deux.
        """
        commit = F.mse_loss(z_pre, z_e.detach())
        codebook = F.mse_loss(z_e, z_pre.detach())
        return commit + self.commitment_weight * codebook
 
    
    # def decode(self, x, actions=None, registers=None, attention_mask=None):
    #     return self.decoder(x, actions=actions, registers=registers, attention_mask=attention_mask)    
    
    def quantize_on_frozen_codebook(self, z_va, codebook):
        dist = torch.cdist(z_va, codebook.weight)
        indices = dist.argmin(dim=-1)
        z_e = codebook[indices]
        z_q = z_va + (z_e - z_va).detach()
        return z_q, z_e, indices
    
    def forward(self, frame_t: torch.Tensor, frame_t_H: torch.Tensor, actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        feats_t = self.extract_features(frame_t)
        feats_t_H = self.extract_features(frame_t_H)

        feats_t_proj = self.feature_proj(feats_t)
        feats_t_H_proj = self.feature_proj(feats_t_H)
        N, P, D = feats_t_proj.shape
        
        z_va, z_vi = self.inv_encoder(feats_t_proj, feats_t_H_proj)

        z_qa, z_ea, act_indices = self.quantize_on_frozen_codebook(z_va, self.act_codebook)

        z_qi, env_indices, env_commit_loss = self.env_vq(z_vi)

        ft_H_hat = self.fwd_decoder(feats_t_proj, z_qa, z_qi)


        if actions is not None:
            z_a = self.encode_actions(actions)
        else:
            z_a = z_qa.detach()

        loss_contrastive = self.contrastive_loss(z_va, z_a)
        loss_recon = F.mse_loss(ft_H_hat, feats_t_H_proj)
        loss_vq_act = F.mse_loss(z_va, z_ea.detach())
        loss_vq_env = env_commit_loss
        loss_reg = z_vi.abs().mean()
        L_total = self.lambda_rec * loss_recon + self.lambda_vq * (loss_vq_act + loss_vq_env) + self.lambda_con * loss_contrastive + self.lambda_reg * loss_reg
        
        return {
            "loss": L_total,
            "loss_recon": loss_recon,
            "loss_vq_act": loss_vq_act,
            "loss_vq_env": loss_vq_env,
            "loss_contrastive": loss_contrastive,
            "loss_reg": loss_reg,
            "z_va": z_va,
            "z_qa": z_qa,
            "z_ea": z_ea,
            "z_vi": z_vi,
            "z_qi": z_qi,
        }
    
    @torch.no_grad()
    def get_action_token(self,
                         frame_t : torch.Tensor,
                         frame_t_H: torch.Tensor,
                         ):
        f_t = self.feature_proj(self.extract_features(frame_t))
        f_t_H = self.feature_proj(self.extract_features(frame_t_H))
        z_va, _ = self.inv_encoder(f_t, f_t_H)
        z_qa, _, act_indices = self.quantize_on_frozen_codebook(z_va, self.act_codebook)
        return z_qa, act_indices
        

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
            batch_first=True,
            norm_first=True
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