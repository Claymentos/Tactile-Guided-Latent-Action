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
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from lam.lightning.lam import ActVAETraining
from lam.tactile import (ALL_BEND_CELLS, ALL_PRESSURE_CELLS, CONTACT_THRESHOLD, TactileDecoder, TactileEncoder,
                         canonical_cells, cells_to_maps, f1_score, zero_init)


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
        readout = repeat(self.readout_token, '1 1 1 d -> b t 1 d', b=B, t=2)
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

        act_cond = self.act_proj(z_qa).unsqueeze(1).expand(B, P,D)
        env_cond = self.env_proj(z_qi).unsqueeze(1).expand(B, P,D)

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
        z_e = codebook(indices)
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


class TactileForwardDecoder(nn.Module):
    """Predicts the gloves at t+H from the frame t, the gloves at t and the action latent only."""

    def __init__(self,
                 model_dim: int,
                 n_layers: int,
                 n_heads: int,
                 act_latent_dim: int,
                 dropout: float):
        super().__init__()
        self.act_proj = nn.Linear(act_latent_dim, model_dim)
        self.layers = nn.ModuleList([
            SpatioBlock(model_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, f_t: torch.Tensor, tactile_tokens: torch.Tensor, z_qa: torch.Tensor) -> torch.Tensor:
        num_patches = f_t.shape[1]
        x = torch.cat([f_t, tactile_tokens], dim=1) + self.act_proj(z_qa).unsqueeze(1)
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x.squeeze(1))[:, num_patches:]


class TactileTargetMixin:
    """The tactile target shared by TactileVDVAE and TactileLAM: a forward model predicts the gloves at t+H (pressure,
    contacts, finger flexion) from the frame t, the gloves at t and the action latent only. The gloves change with the
    fingers and the contacts, not with the camera or the arm, so the action latent has to encode what the fingers did.

    tactile_target=False is the baseline: the tactile forward model is still trained, but on a detached action latent,
    so that its errors measure how much the latent knows about the fingers without being pushed to."""

    def init_tactile_target(self,
                            model_dim: int,
                            act_latent_dim: int,
                            nhead: int,
                            tactile_dec_layers: int,
                            pressure_weight: float,
                            contact_weight: float,
                            bend_weight: float,
                            contact_cells_weight: float,
                            contact_pos_weight: float,
                            tactile_target: bool):
        self.tactile_target = tactile_target
        self.pressure_weight = pressure_weight
        self.contact_weight = contact_weight
        self.bend_weight = bend_weight
        self.contact_cells_weight = contact_cells_weight
        self.register_buffer('contact_pos_weight', torch.tensor(contact_pos_weight), persistent=False)

        self.tactile_encoder = TactileEncoder(model_dim)
        self.tactile_fwd_decoder = TactileForwardDecoder(model_dim=model_dim, n_layers=tactile_dec_layers, n_heads=nhead,
                                                         act_latent_dim=act_latent_dim, dropout=0.1)
        self.tactile_decoder = TactileDecoder(model_dim)

    def predict_tactile(self, context: torch.Tensor, tactile_t: torch.Tensor, z_a: torch.Tensor):
        """(N, P, D) patches of the frames t, (N, 2, 21, 21) gloves at t, (N, D_a) action latents -> the tactile decoder's
        outputs (pressure change, contact logits, bend change)."""
        if not self.tactile_target:
            context, z_a = context.detach(), z_a.detach()
        tokens = self.tactile_fwd_decoder(context, self.tactile_encoder(tactile_t), z_a)
        return self.tactile_decoder(tokens)

    def tactile_losses(self, tactile_prediction, tactile_t: torch.Tensor, tactile_t_H: torch.Tensor) -> Dict[str, torch.Tensor]:
        delta_pressure, contact_logits, delta_bend = tactile_prediction
        cells_t, cells_t_H = canonical_cells(tactile_t), canonical_cells(tactile_t_H)
        pressure_t, pressure_t_H = cells_t[..., ALL_PRESSURE_CELLS], cells_t_H[..., ALL_PRESSURE_CELLS]
        bend_t, bend_t_H = cells_t[..., ALL_BEND_CELLS], cells_t_H[..., ALL_BEND_CELLS]
        contact_t, contact_t_H = pressure_t > CONTACT_THRESHOLD, pressure_t_H > CONTACT_THRESHOLD
        pressure_hat, bend_hat = pressure_t + delta_pressure, bend_t + delta_bend

        # the maps are sparse: without these weights, predicting no contact at all is already a good score
        cell_weights = 1.0 + self.contact_cells_weight * (contact_t | contact_t_H).to(pressure_t.dtype)
        loss_pressure = (cell_weights * (pressure_hat - pressure_t_H) ** 2).sum() / cell_weights.sum()
        loss_contact = F.binary_cross_entropy_with_logits(contact_logits, contact_t_H.to(contact_logits.dtype),
                                                          pos_weight=self.contact_pos_weight.to(contact_logits.dtype))
        loss_bend = F.mse_loss(bend_hat, bend_t_H)

        return {
            "loss_tactile": self.pressure_weight * loss_pressure + self.contact_weight * loss_contact + self.bend_weight * loss_bend,
            "loss_pressure": loss_pressure.detach(),
            "loss_contact": loss_contact.detach(),
            "loss_bend": loss_bend.detach(),
            # baselines "nothing changes" for each error
            "pressure_mae": (pressure_hat - pressure_t_H).abs().mean().detach(),
            "pressure_mae_copy": (pressure_t - pressure_t_H).abs().mean(),
            "bend_mae": (bend_hat - bend_t_H).abs().mean().detach(),
            "bend_mae_copy": (bend_t - bend_t_H).abs().mean(),
            "contact_f1": f1_score(contact_logits > 0, contact_t_H),
            "contact_f1_copy": f1_score(contact_t, contact_t_H),
        }

    @staticmethod
    def predicted_maps(tactile_prediction, tactile_t: torch.Tensor) -> torch.Tensor:
        """The predicted gloves at t+H as (N, 2, 21, 21) maps, as the dataset draws them."""
        delta_pressure, _, delta_bend = tactile_prediction
        cells_t = canonical_cells(tactile_t)
        pressure = (cells_t[..., ALL_PRESSURE_CELLS] + delta_pressure).clamp(0, 1)
        bend = (cells_t[..., ALL_BEND_CELLS] + delta_bend).clamp(0, 1)
        return cells_to_maps(pressure, bend)


class TactileVDVAE(TactileTargetMixin, nn.Module):
    """VDVAE trained without action labels, whose action latent is pushed towards the fine movements of the hands by
    touch.

    The IDM only sees the video. Two forward models read its latents:
      - the video one predicts the DINOv3 patches of t+H from those of t, z_a and z_env;
      - the tactile one predicts the gloves at t+H (pressure, contacts, finger flexion) from the frame t, the gloves at
        t and z_a only. The gloves change with the fingers and the contacts, not with the camera or the arm, and z_a is
        its only way to know how they changed: z_a has to encode what the fingers did, read from the video.
    Touch is only a training target: the trained IDM runs on videos without gloves.

    tactile_target=False is the baseline: the tactile forward model is still trained, but on a detached z_a, so that
    its errors measure how much z_a knows about the fingers without being pushed to.
    """

    def __init__(self,
                 dino_repo: str,
                 dino_weights: str,
                 model_dim: int,
                 act_latent_dim: int,
                 env_latent_dim: int,
                 act_codebook_size: int,
                 env_codebook_size: int,
                 nhead: int,
                 enc_layers: int,
                 dec_layers: int,
                 tactile_dec_layers: int,
                 lambda_rec: float,
                 lambda_tactile: float,
                 lambda_vq: float,
                 lambda_reg: float,
                 pressure_weight: float,
                 contact_weight: float,
                 bend_weight: float,
                 contact_cells_weight: float,
                 contact_pos_weight: float,
                 commitment_weight: float,
                 vq_args: Optional[dict] = None,
                 tactile_target: bool = True):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_tactile = lambda_tactile
        self.lambda_vq = lambda_vq
        self.lambda_reg = lambda_reg

        self.feature_extractor = torch.hub.load(dino_repo, 'dinov3_vitb16', source='local', weights=dino_weights)
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()
        feature_dim = self.feature_extractor.embed_dim

        self.feature_proj = nn.Linear(feature_dim, model_dim)
        self.inv_encoder = InverseDynamicEncoder(model_dim=model_dim, n_layers=enc_layers, n_heads=nhead,
                                                 act_latent_dim=act_latent_dim, env_latent_dim=env_latent_dim, dropout=0.1)
        self.fwd_decoder = ForwardDynamicsDecoder(model_dim=model_dim, n_layers=dec_layers, n_heads=nhead,
                                                  act_latent_dim=act_latent_dim, env_latent_dim=env_latent_dim, dropout=0.1)
        self.feature_head = nn.Linear(model_dim, feature_dim)
        zero_init(self.feature_head)

        self.init_tactile_target(model_dim=model_dim, act_latent_dim=act_latent_dim, nhead=nhead,
                                 tactile_dec_layers=tactile_dec_layers, pressure_weight=pressure_weight,
                                 contact_weight=contact_weight, bend_weight=bend_weight,
                                 contact_cells_weight=contact_cells_weight, contact_pos_weight=contact_pos_weight,
                                 tactile_target=tactile_target)

        vq_args = dict(vq_args or {})
        self.act_vq = VectorQuantize(dim=act_latent_dim, codebook_size=act_codebook_size,
                                     commitment_weight=commitment_weight, **vq_args)
        self.env_vq = VectorQuantize(dim=env_latent_dim, codebook_size=env_codebook_size,
                                     commitment_weight=commitment_weight, **vq_args)

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.eval()  # stays frozen, Lightning sets the whole module to train mode
        return self

    @torch.no_grad()
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T, P, F)"""
        B = frames.shape[0]
        tokens = self.feature_extractor.forward_features(frames.flatten(0, 1))["x_norm_patchtokens"]
        return rearrange(tokens, '(b t) p f -> b t p f', b=B)

    def encode(self, video_t: torch.Tensor, video_t_H: torch.Tensor):
        """The IDM, from the video only: latents before and after quantization, code indices and commitment losses."""
        z_va, z_vi = self.inv_encoder(video_t, video_t_H)
        z_qa, act_indices, loss_vq_act = self.act_vq(z_va)
        z_qi, env_indices, loss_vq_env = self.env_vq(z_vi)
        return z_va, z_vi, z_qa, z_qi, act_indices, env_indices, loss_vq_act, loss_vq_env

    def decode(self, features_t: torch.Tensor, video_t: torch.Tensor, tactile_t: torch.Tensor, z_qa: torch.Tensor, z_qi: torch.Tensor):
        """Both forward models: the DINOv3 patches of t+H, and the tactile decoder's outputs (pressure change, contact
        logits, bend change)."""
        # the video forward model predicts the change of the DINOv3 patches
        features_hat = features_t + self.feature_head(self.fwd_decoder(video_t, z_qa, z_qi))

        # the tactile forward model only gets z_a, not z_env
        return features_hat, self.predict_tactile(video_t, tactile_t, z_qa)

    @torch.no_grad()
    def visualize(self, frames: torch.Tensor, tactile: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predictions of every transition of the clips, with the inferred action latents ('recon') and with action codes
        drawn at random from the codebook ('imagined', environment latents kept). frames (B, T+1, 3, H, W), tactile
        (B, T+1, 2, 21, 21) -> features_* (B, T, P, F) and tactile_* (B, T, 2, 21, 21) for the frames 1..T."""
        B = frames.shape[0]
        features = self.extract_features(frames)
        features_t, features_t_H = features[:, :-1].flatten(0, 1), features[:, 1:].flatten(0, 1)
        tactile_t = tactile[:, :-1].flatten(0, 1)
        video_t, video_t_H = self.feature_proj(features_t), self.feature_proj(features_t_H)
        _, _, z_qa, z_qi, _, _, _, _ = self.encode(video_t, video_t_H)
        random_indices = torch.randint(0, self.act_vq.codebook_size, (z_qa.shape[0],), device=z_qa.device)
        z_random = self.act_vq.get_output_from_indices(random_indices).to(z_qa.dtype)

        outputs = {"features_target": features[:, 1:], "tactile_target": tactile[:, 1:]}
        for name, z in (("recon", z_qa), ("imagined", z_random)):
            features_hat, tactile_prediction = self.decode(features_t, video_t, tactile_t, z, z_qi)
            outputs[f"features_{name}"] = features_hat.unflatten(0, (B, -1))
            outputs[f"tactile_{name}"] = self.predicted_maps(tactile_prediction, tactile_t).unflatten(0, (B, -1))
        return outputs

    def forward(self, frames: torch.Tensor, tactile: torch.Tensor, rel_t: Optional[torch.Tensor] = None,
                shuffle_latent: bool = False) -> Dict[str, torch.Tensor]:
        """frames (B, T+1, 3, H, W), tactile (B, T+1, 2, 21, 21), rel_t (B, T). Every pair of consecutive frames is a
        transition, except the ones repeating a frame to pad a short episode. shuffle_latent gives both forward models
        the latents of other transitions: how much worse they get measures how much they rely on the latents."""
        features = self.extract_features(frames)
        keep = (rel_t > 0).flatten() if rel_t is not None else torch.ones_like(tactile[:, 1:, 0, 0, 0], dtype=torch.bool).flatten()
        features_t, features_t_H = features[:, :-1].flatten(0, 1)[keep], features[:, 1:].flatten(0, 1)[keep]
        tactile_t, tactile_t_H = tactile[:, :-1].flatten(0, 1)[keep], tactile[:, 1:].flatten(0, 1)[keep]

        video_t, video_t_H = self.feature_proj(features_t), self.feature_proj(features_t_H)
        z_va, z_vi, z_qa, z_qi, act_indices, env_indices, loss_vq_act, loss_vq_env = self.encode(video_t, video_t_H)
        if shuffle_latent:
            permutation = torch.randperm(z_qa.shape[0], device=z_qa.device)
            z_qa, z_qi = z_qa[permutation], z_qi[permutation]

        features_hat, tactile_prediction = self.decode(features_t, video_t, tactile_t, z_qa, z_qi)
        loss_recon = F.mse_loss(features_hat, features_t_H)
        outputs = self.tactile_losses(tactile_prediction, tactile_t, tactile_t_H)

        loss_reg = z_vi.abs().mean()
        outputs.update({
            "loss": self.lambda_rec * loss_recon + self.lambda_tactile * outputs["loss_tactile"]
                    + self.lambda_vq * (loss_vq_act + loss_vq_env) + self.lambda_reg * loss_reg,
            "loss_recon": loss_recon,
            "loss_recon_copy": F.mse_loss(features_t, features_t_H),
            "loss_vq_act": loss_vq_act,
            "loss_vq_env": loss_vq_env,
            "loss_reg": loss_reg,
            "act_codes_used": act_indices.unique().numel() * 1.0,
            "env_codes_used": env_indices.unique().numel() * 1.0,
        })
        return outputs

    @torch.no_grad()
    def get_action_token(self, frame_t: torch.Tensor, frame_t_H: torch.Tensor):
        """(B, 3, H, W) twice -> quantized action latents and their indices, from the video only."""
        features = self.extract_features(torch.stack([frame_t, frame_t_H], dim=1))
        _, _, z_qa, _, act_indices, _, _, _ = self.encode(self.feature_proj(features[:, 0]), self.feature_proj(features[:, 1]))
        return z_qa, act_indices


class ResidualActVAE(nn.Module):
    def __init__(self,
                 action_dim: int,
                 model_dim: int,
                 latent_dim: int,
                 num_quantizers: int,
                 num_latents: int, 
                 enc_layers: int,
                 dec_layers: int,
                 nhead: int,
                 dropout: float,
                 codebook_size: int,
                 codebook_dim: int,
                 decay: float,
                 commitment_weight: float,
                 kmeans_init: bool,
                 ):
        super().__init__()

        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_quantizers = num_quantizers
        self.num_latents = num_latents
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.nhead = nhead
        self.dropout = dropout

        self.rvq = ResidualVQ(
            dim=action_dim,
            num_quantizers=num_quantizers,
            codebook_size=num_latents,
            decay=decay,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            codebook_dim=codebook_dim
        )

        self.input_proj = nn.Linear(action_dim, latent_dim)
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

    def encode(self, x):
        return self.encoder(self.input_proj(x) + self.pos_embedding[:, :x.shape[1], :])

    def decode(self, z_q):
        dec_in = self.latent_proj(z_q) + self.pos_embedding[:, :z_q.shape[1], :]
        out = self.decoder(dec_in)
        actions_recon_flat = self.output_proj(out)
        actions_recon = rearrange(actions_recon_flat, 'b t (p c) -> b t p c', p=88, c=3)
        return actions_recon
    
    def forward(self, actions: torch.Tensor):
        B,T =actions.shape[:2]
        x = rearrange(actions, 'b t p c -> b t (p c)') 
        z_e = self.encode(x)
        z_e = self.to_latent(z_e)

        z_q, indices, vq_loss = self.rvq(z_e)
        action_recons = self.decode(z_q)
        recon_loss = F.mse_loss(action_recons, actions)
        total_loss = recon_loss + vq_loss.mean()

        return {
            "recon_actions": action_recons,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss.mean(),
            "z_q": z_q,
            "indices": indices,
            "z_e": z_e,
            "loss": total_loss
        }


class ActVAE(nn.Module):
    def __init__(self, 
                 action_dim=264,       
                 model_dim=512,        
                 latent_dim=512, 
                 num_latents=256,       
                 enc_layers=12,        
                 dec_layers=12,       
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
        
        self.vq = VectorQuantize(
            dim=model_dim,
            codebook_size=num_latents,
            codebook_dim=latent_dim,
            commitment_weight=1.0
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
        
        z_q, indices, vq_loss = self.vq(z_e)
        
        actions_recon = self.decode(z_q)

        
        recon_loss = nn.functional.mse_loss(actions_recon, actions)
        total_loss = recon_loss + vq_loss

  

        return {
            "recon_actions": actions_recon,
            "recon_loss": recon_loss,
            "z_q": z_q,
            "indices": indices,
            "vq_loss": vq_loss,
            "loss": total_loss
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


class TactileLAM(TactileTargetMixin, LAM):
    """LAM (LAPA) whose action tokens are also pushed towards the fine movements of the hands by touch.

    LAM is unchanged: the spatio-temporal encoder reads the clip and outputs one action token per transition, the decoder
    predicts the frame t+1 from the frame t and its action token. Added: the tactile forward model (TactileTargetMixin)
    predicts the gloves at t+1 from the features of the frame t, the gloves at t and the action token. LAM has a single
    latent, so the action tokens carry everything touch requires. Touch is only a training target: get_latent_actions
    still reads the video only.
    """

    def __init__(self,
                 encoder,
                 decoder,
                 vq,
                 feature_extractor,
                 model_dim: int,
                 nhead: int,
                 tactile_dec_layers: int,
                 pressure_weight: float,
                 contact_weight: float,
                 bend_weight: float,
                 contact_cells_weight: float,
                 contact_pos_weight: float,
                 tactile_target: bool = True,
                 perceptual_regularizer=None,
                 num_register_tokens=0,
                 register_dropout_mean=0.0,
                 register_dropout_std=0.0):
        super().__init__(encoder=encoder, decoder=decoder, vq=vq, feature_extractor=feature_extractor,
                         perceptual_regularizer=perceptual_regularizer, num_register_tokens=num_register_tokens,
                         register_dropout_mean=register_dropout_mean, register_dropout_std=register_dropout_std)
        self.init_tactile_target(model_dim=model_dim, act_latent_dim=model_dim, nhead=nhead,
                                 tactile_dec_layers=tactile_dec_layers, pressure_weight=pressure_weight,
                                 contact_weight=contact_weight, bend_weight=bend_weight,
                                 contact_cells_weight=contact_cells_weight, contact_pos_weight=contact_pos_weight,
                                 tactile_target=tactile_target)

    def forward(self, x, tactile):
        """
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        tactile: torch.Tensor, shape (batch_size, seq_len, 2, 21, 21), the gloves of every frame
        returns: LAM's outputs, the tactile losses and metrics, and tactile_prediction (the tactile decoder's outputs for
        every transition, flattened over batch and time)
        """
        output = super().forward(x)
        tactile_t, tactile_t_H = tactile[:, :-1].flatten(0, 1), tactile[:, 1:].flatten(0, 1)
        context = output["features"][:, :-1].flatten(0, 1)
        tactile_prediction = self.predict_tactile(context, tactile_t, output["action_tokens"].flatten(0, 1).to(context.dtype))
        output.update(self.tactile_losses(tactile_prediction, tactile_t, tactile_t_H))
        output["tactile_prediction"] = tactile_prediction
        return output

    @torch.no_grad()
    def tactile_with_random_actions(self, features, tactile):
        """The tactile decoder's outputs with action tokens drawn at random from the codebook, like LAMTraining's
        reconstruct_with_random_actions for the video. features (B, T, P, D), tactile (B, T, 2, 21, 21)."""
        B, T = tactile.shape[:2]
        indices = torch.randint(0, self.vq.codebook_size, (B, T - 1), device=tactile.device)
        tokens = self.vq.dequantize(indices).flatten(0, 1).to(features.dtype)
        return self.predict_tactile(features[:, :-1].flatten(0, 1), tactile[:, :-1].flatten(0, 1), tokens)


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