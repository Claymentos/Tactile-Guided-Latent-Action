"""Tokenizer de mouvement des mains pour la loss MCP de JALA (motion-token prediction, éq. 2 de arXiv 2602.21736).

Dans le papier, chaque chunk de mouvement (15 frames) est quantifié par GRVQ (Group Residual VQ) en 128 tokens discrets,
64 pour le poignet et 64 pour les doigts, avec un codebook de 4096 entrées par partie ; le VLM prédit ces tokens. Le
tokenizer est appris avant JALA puis figé.

Ici, sur les 42 points 3D EgoDex (pas de paramètres MANO) :
  - partie 'wrist'   : trajectoire du poignet de chaque main, relative à sa position au début du chunk (C, 3) ;
  - partie 'fingers' : trajectoire des 20 points des doigts relatifs au poignet, moins leur valeur au début du chunk
                       (C, 20, 3) : l'articulation, sans la translation de la main ;
  en cm. Chaque (main, partie) est encodée par un MLP en `latents_per_part` vecteurs de dimension code_dim, quantifiés
  par un GroupedResidualVQ (groups x num_quantizers codes chacun) propre à la partie (codebook partagé entre les deux
  mains). Tokens par chunk : 2 mains x 2 parties x latents_per_part x groups x num_quantizers (défaut 4 x 2 x 4 -> 128),
  ordonnés [gauche-poignet, gauche-doigts, droite-poignet, droite-doigts] : la première moitié correspond à la main
  gauche, comme les queries de LAP (tête gauche puis tête droite).
"""

import torch
import torch.nn.functional as F
from torch import nn
from vector_quantize_pytorch import GroupedResidualVQ

from lam.jala import FINGERS, METERS_TO_CM, NUM_HAND_KEYPOINTS, WRIST

PARTS = ('wrist', 'fingers')
PART_POINTS = {'wrist': 1, 'fingers': 20}
HAND_FINGERS = (tuple(i for i in FINGERS if i < 21), tuple(i for i in FINGERS if i >= 21))


def chunk_parts(chunks):
    """Trajectoires (..., C, 42, 3) en mètres -> {'wrist': (..., 2, C, 1, 3), 'fingers': (..., 2, C, 20, 3)} en cm,
    relatives au début du chunk (main gauche puis droite)."""
    chunks = chunks.float() * METERS_TO_CM
    wrist = torch.stack([chunks[..., w:w + 1, :] for w in WRIST], dim=-4)                     # (..., 2, C, 1, 3)
    fingers = torch.stack([chunks[..., list(f), :] for f in HAND_FINGERS], dim=-4)            # (..., 2, C, 20, 3)
    fingers = fingers - wrist
    return dict(wrist=wrist - wrist[..., :1, :, :], fingers=fingers - fingers[..., :1, :, :])


def parts_to_keypoints(parts):
    """Inverse de chunk_parts (au début du chunk près) : déplacement des 42 points (..., C, 42, 3) en cm."""
    wrist, fingers = parts['wrist'], parts['fingers'] + parts['wrist']
    out = []
    for hand in range(2):
        out += [wrist[..., hand, :, :, :], fingers[..., hand, :, :, :]]
    return torch.cat(out, dim=-2)


class MotionTokenizer(nn.Module):
    def __init__(self,
                 chunk_len: int = 16,
                 latents_per_part: int = 4,
                 code_dim: int = 64,
                 groups: int = 2,
                 num_quantizers: int = 4,
                 codebook_size: int = 4096,
                 hidden_dim: int = 512,
                 commitment_weight: float = 0.25,
                 kmeans_init: bool = True,
                 threshold_ema_dead_code: int = 2):
        """chunk_len : points de trajectoire par chunk (lam.datasets.egodex.dataset, return_motion_chunks) ;
        codebook_size : par partie (4096 dans le papier)."""
        super().__init__()
        self.chunk_len, self.latents_per_part, self.code_dim = chunk_len, latents_per_part, code_dim
        self.groups, self.num_quantizers, self.codebook_size = groups, num_quantizers, codebook_size
        self.tokens_per_part = latents_per_part * groups * num_quantizers
        self.num_tokens = 2 * len(PARTS) * self.tokens_per_part
        self.encoders, self.decoders, self.quantizers = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for part in PARTS:
            size = chunk_len * PART_POINTS[part] * 3
            self.encoders[part] = nn.Sequential(nn.Linear(size + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
                                                nn.GELU(), nn.Linear(hidden_dim, latents_per_part * code_dim))
            self.decoders[part] = nn.Sequential(nn.Linear(latents_per_part * code_dim + 2, hidden_dim), nn.GELU(),
                                                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, size))
            self.quantizers[part] = GroupedResidualVQ(dim=code_dim, groups=groups, num_quantizers=num_quantizers,
                                                      codebook_size=codebook_size, kmeans_init=kmeans_init,
                                                      threshold_ema_dead_code=threshold_ema_dead_code,
                                                      commitment_weight=commitment_weight)
        # partie de chaque token (0 : poignet, 1 : doigts), dans l'ordre [g-poignet, g-doigts, d-poignet, d-doigts]
        self.register_buffer('token_part', torch.arange(len(PARTS)).repeat_interleave(self.tokens_per_part).repeat(2),
                             persistent=False)

    def _hand_code(self, shape, device):
        return torch.eye(2, device=device).expand(*shape, 2, 2)                    # one-hot de la main

    def _encode_part(self, part, x):
        """x (N, 2, C, J, 3) -> quantifié (N, 2, L, code_dim), indices (N, 2, L*groups*q), commit (scalaire)."""
        N = x.shape[0]
        hand = self._hand_code((N,), x.device)
        latent = self.encoders[part](torch.cat([x.flatten(2), hand], -1)).unflatten(-1, (self.latents_per_part, self.code_dim))
        quantized, indices, commit = self.quantizers[part](latent.flatten(0, 1))      # indices (groups, 2N, L, q)
        indices = indices.permute(1, 2, 0, 3).flatten(1)                              # (2N, L * groups * q)
        return quantized.unflatten(0, (N, 2)), indices.unflatten(0, (N, 2)), commit.sum()

    def _decode_part(self, part, quantized):
        N = quantized.shape[0]
        hand = self._hand_code((N,), quantized.device)
        out = self.decoders[part](torch.cat([quantized.flatten(2), hand], -1))
        return out.unflatten(-1, (self.chunk_len, PART_POINTS[part], 3))

    def _split(self, tokens):
        """(N, K) -> {part: (N, 2, tokens_per_part)}"""
        t = tokens.unflatten(-1, (2, len(PARTS), self.tokens_per_part))
        return {part: t[:, :, i] for i, part in enumerate(PARTS)}

    def forward(self, chunks):
        """chunks (..., C, 42, 3) en mètres -> pertes et reconstructions (apprentissage du tokenizer)."""
        lead = chunks.shape[:-3]
        parts = {k: v.flatten(0, len(lead) - 1) if lead else v[None] for k, v in chunk_parts(chunks).items()}
        out, commit, indices = {}, 0.0, []
        for part in PARTS:
            quantized, idx, c = self._encode_part(part, parts[part])
            out[part] = self._decode_part(part, quantized)
            commit, indices = commit + c, indices + [idx]
        recon = {part: (out[part] - parts[part]).abs().mean() for part in PARTS}
        keypoints, target = parts_to_keypoints(out), parts_to_keypoints(parts)
        return dict(loss_wrist=recon['wrist'], loss_fingers=recon['fingers'], loss_commit=commit,
                    end_mae=(keypoints[..., -1, :, :] - target[..., -1, :, :]).abs().mean(),        # cm, fin du chunk
                    end_zero=target[..., -1, :, :].abs().mean(),                                    # référence : immobile
                    indices=self._interleave(indices))

    def _interleave(self, indices):
        """[(N, 2, T) poignet, (N, 2, T) doigts] -> (N, K) dans l'ordre des tokens."""
        return torch.stack(indices, dim=2).flatten(1)

    @torch.no_grad()
    def encode(self, chunks):
        """chunks (..., C, 42, 3) en mètres -> tokens (..., K) (indices dans le codebook de leur partie)."""
        lead = chunks.shape[:-3]
        parts = {k: v.reshape(-1, *v.shape[len(lead):]) for k, v in chunk_parts(chunks).items()}
        indices = [self._encode_part(part, parts[part])[1] for part in PARTS]
        return self._interleave(indices).reshape(*lead, self.num_tokens)

    @torch.no_grad()
    def decode(self, tokens):
        """tokens (..., K) -> déplacement des 42 points sur le chunk (..., C, 42, 3) en cm, relatif au début."""
        lead = tokens.shape[:-1]
        split = self._split(tokens.reshape(-1, self.num_tokens))
        out = {}
        for part in PARTS:
            idx = split[part].unflatten(-1, (self.latents_per_part, self.groups, self.num_quantizers))
            idx = idx.flatten(0, 1).permute(2, 0, 1, 3)                                         # (groups, 2N, L, q)
            quantized = self.quantizers[part].get_output_from_indices(idx).unflatten(0, (-1, 2))
            out[part] = self._decode_part(part, quantized)
        return parts_to_keypoints(out).reshape(*lead, self.chunk_len, NUM_HAND_KEYPOINTS, 3)

    @classmethod
    def from_checkpoint(cls, path, map_location='cpu'):
        """Tokenizer figé depuis un checkpoint lightning de lam.lightning.motion_tokenizer.MotionTokenizerTraining."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        hparams = dict(checkpoint.get('hyper_parameters', {}).get('tokenizer', {}))
        tokenizer = cls(**hparams)
        state = {k[len('tokenizer.'):]: v for k, v in checkpoint['state_dict'].items() if k.startswith('tokenizer.')}
        tokenizer.load_state_dict(state)
        return tokenizer.eval().requires_grad_(False)
