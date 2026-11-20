"""JALA-LAM : Joint-Aligned Latent Action (Luo et al., CVPR 2026, arXiv 2602.21736), version compacte pour ce repo.

JALA remplace la reconstruction de frames futures (LAPA, Genie, lam.lam.LAM) par un alignement entre deux espaces :
  - un modèle d'inverse dynamics, le Latent Action Perceiver (LAP), qui lit les frames de début et de fin d'une
    transition (v_t, v_t+δ) sur des features visuelles figées (DINOv3) et produit K actions latentes continues z_t ;
  - les embeddings prédictifs h_t d'un modèle qui ne voit que le passé (dans le papier, la couche 19/28 d'un VLM
    InternVL3-2B qui prédit les tokens de mouvement de la main) ;
avec  L = 1_labeled · L_action + λ · L_align,   L_align = Σ_k |h_t,k - z_t,k|_1   (λ = 0.5).

Le contexte du prédicteur passe par le Latent State Perceiver (LSP) : même architecture que LAP, lancé sur une frame
dupliquée (v_t, v_t). Pour que LAP et LSP partagent un espace latent sans s'effondrer, JALA découple leurs mises à jour :
  - le *backbone* (projection, attention, têtes de sortie) est appris par le gradient côté LSP, et LAP le suit en EMA ;
  - les *queries* sont apprises par le gradient côté LAP, et LSP les suit en EMA :
        θ_b^LAP <- α θ_b^LAP + (1 - α) θ_b^LSP,      θ_q^LSP <- α θ_q^LSP + (1 - α) θ_q^LAP,      α = 0.999.

Adaptations, faute de VLM dans ce repo :
  - le VLM est remplacé par LatentPredictor, un petit transformer causal dans le temps : à la transition t, il voit les
    tokens d'état LSP des frames 0..t (et la tâche EgoDex, à la place de l'instruction), et K queries de mouvement dont
    l'état caché à la couche align_layer est h_t ;
  - la loss sur les tokens de mouvement GRVQ (MCP) est remplacée par une régression L1 du déplacement 3D des 42 points
    des mains EgoDex entre t et t+δ (en cm, repère caméra de la première frame), prédit depuis la sortie du prédicteur ;
  - align_norm (par défaut) normalise h et z (LayerNorm sans paramètres) avant la L1 : sans le VLM pré-entraîné qui
    fixe l'échelle de h, les deux côtés pourraient sinon rétrécir ensemble vers 0. align_norm=False donne la L1 brute.
Une sonde (linéaire + MLP) sur z détaché mesure l'information d'action contenue dans les actions latentes, sans agir
sur elles.

Entrées : images (B, T, 3, H, W) normalisées ImageNet (EgoDexDataset), actions EgoDex (B, T-1, 88, 3).
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

NUM_HAND_KEYPOINTS = 42          # 21 gauche puis 21 droite (lam.datasets.egodex.dataset.ALL_KEYS)
POSES_PER_FRAME = 44             # 42 points + position caméra + rotation caméra
METERS_TO_CM = 100.0


def hand_motion_target(actions):
    """Actions EgoDex (B, T-1, 88, 3) -> déplacement des 42 points de la main entre t et t+δ, en cm (B, T-1, 42, 3)."""
    current = actions[..., :NUM_HAND_KEYPOINTS, :]
    following = actions[..., POSES_PER_FRAME:POSES_PER_FRAME + NUM_HAND_KEYPOINTS, :]
    return (following - current) * METERS_TO_CM


def sincos_2d(grid_size, dim, temperature=10000.0):
    """Embedding de position sinus/cosinus 2D fixe (grid_h * grid_w, dim), d'échelle ~1 (comme MAE)."""
    assert dim % 4 == 0, "dim doit être multiple de 4"
    gh, gw = grid_size
    y, x = torch.meshgrid(torch.arange(gh, dtype=torch.float32), torch.arange(gw, dtype=torch.float32), indexing='ij')
    omega = 1.0 / temperature ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    y, x = y.flatten()[:, None] * omega, x.flatten()[:, None] * omega
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, out_dim=None, dropout=0.0):
        super().__init__(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim or dim))


class PerceiverLayer(nn.Module):
    """Les queries lisent les features visuelles (cross-attention), échangent entre elles (self-attention), puis MLP."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm_q, self.norm_x = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, 4 * dim, dropout=dropout)

    def forward(self, queries, x):
        x = self.norm_x(x)
        q = self.norm_q(queries)
        queries = queries + self.cross_attention(q, x, x, need_weights=False)[0]
        q = self.norm_self(queries)
        queries = queries + self.self_attention(q, q, q, need_weights=False)[0]
        return queries + self.mlp(self.norm_mlp(queries))


class PerceiverBackbone(nn.Module):
    """Tout le Perceiver sauf les queries : projection des features des deux frames, couches Perceiver, et têtes de
    sortie vers l'espace du prédicteur (une par main, comme JALA pour la vidéo à deux mains : la première moitié des
    queries passe par la tête gauche, la seconde par la tête droite).

    pair_fusion (le papier ne dit pas comment les deux frames sont combinées) :
      'patch'    : un token par patch, W_s LN(f_a) + W_d (LN(f_b) - LN(f_a)) + position. La différence est explicite et
                   nulle sur le fond immobile ; pour LSP sur (v_t, v_t), elle est exactement nulle : il ne reste que
                   l'état. Deux fois moins de tokens.
      'sequence' : les 2P tokens des deux frames, distingués par un embedding de frame.
    """

    def __init__(self, feature_dim, dim, out_dim, grid_size, num_layers=2, num_heads=8, num_hands=2, dropout=0.0,
                 pair_fusion='patch'):
        super().__init__()
        assert pair_fusion in ('patch', 'sequence'), pair_fusion
        self.pair_fusion = pair_fusion
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, dim)
        self.difference_projection = nn.Linear(feature_dim, dim) if pair_fusion == 'patch' else None
        # l'inverse dynamics compare des positions entre les deux frames : position sinus/cosinus fixe d'échelle ~1
        # (un embedding appris initialisé petit laisse le Perceiver bloqué longtemps), et frames bien distinguées
        self.frame_embedding = nn.Parameter(torch.randn(2, dim) * 0.5) if pair_fusion == 'sequence' else None
        self.register_buffer('position_embedding', sincos_2d(grid_size, dim), persistent=False)
        self.layers = nn.ModuleList([PerceiverLayer(dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleList([FeedForward(dim, 2 * dim, out_dim) for _ in range(num_hands)])

    def forward(self, queries, first, last):
        """queries (B, K, dim) ; first, last : features (B, P, feature_dim) -> latents (B, K, out_dim)."""
        first, last = self.input_norm(first), self.input_norm(last)
        if self.pair_fusion == 'patch':
            x = self.input_projection(first) + self.difference_projection(last - first) + self.position_embedding
        else:
            x = torch.cat([self.input_projection(first) + self.frame_embedding[0] + self.position_embedding,
                           self.input_projection(last) + self.frame_embedding[1] + self.position_embedding], dim=1)
        for layer in self.layers:
            queries = layer(queries, x)
        groups = self.norm(queries).chunk(len(self.heads), dim=1)
        return torch.cat([head(group) for head, group in zip(self.heads, groups)], dim=1)


class Perceiver(nn.Module):
    """LAP ou LSP : backbone + K queries apprises."""

    def __init__(self, num_queries, dim, **backbone_kwargs):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.backbone = PerceiverBackbone(dim=dim, **backbone_kwargs)

    def forward(self, first, last):
        return self.backbone(self.queries.expand(first.shape[0], -1, -1), first, last)


class LatentPredictor(nn.Module):
    """Remplace le VLM de JALA. Séquence par transition t : [Pv tokens visuels de v_t (patchs DINO poolés, comme les
    tokens visuels du VLM), K tokens d'état LSP de v_t, K queries de mouvement], avec attention causale par blocs (les
    tokens de la transition t voient ceux des transitions <= t, donc jamais v_t+1), précédée d'un token de tâche
    optionnel. Renvoie h (état caché des queries de mouvement à la couche align_layer,
    aligné sur les actions latentes de LAP) et la sortie finale des queries (pour la tête d'action)."""

    def __init__(self, dim, num_queries, n_layer=6, num_heads=8, align_layer=4, max_steps=16, num_tasks=0, dropout=0.0,
                 feature_dim=None, visual_grid=None):
        """feature_dim, visual_grid : dimension des features DINO et grille (h, w) après pooling ; None : pas de tokens
        visuels, le prédicteur ne voit que les tokens LSP."""
        super().__init__()
        assert 1 <= align_layer <= n_layer, f"align_layer={align_layer} doit être dans [1, n_layer={n_layer}]"
        self.num_queries, self.align_layer = num_queries, align_layer
        self.motion_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.type_embedding = nn.Parameter(torch.randn(3, dim) * 0.02)          # 0 : état, 1 : mouvement, 2 : visuel
        self.step_embedding = nn.Embedding(max_steps, dim)
        self.task_embedding = nn.Embedding(num_tasks, dim) if num_tasks > 0 else None
        self.state_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.num_visual = 0
        if visual_grid is not None:
            self.num_visual = visual_grid[0] * visual_grid[1]
            self.visual_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, dim))
            self.register_buffer('visual_position', sincos_2d(visual_grid, dim), persistent=False)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, num_heads, 4 * dim, dropout, activation='gelu',
                                                                batch_first=True, norm_first=True) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(dim)

    def attention_mask(self, steps, device, with_task):
        """Masque booléen (True = interdit) : bloc par transition, causal entre transitions ; le token de tâche (s'il y
        en a un, en tête) est vu par tous et ne voit que lui-même."""
        step_of_token = torch.arange(steps, device=device).repeat_interleave(self.num_visual + 2 * self.num_queries)
        if with_task:
            step_of_token = torch.cat([torch.full((1,), -1, device=device), step_of_token])
        mask = step_of_token[None, :] > step_of_token[:, None]
        if with_task:
            mask[0, 1:] = True
        return mask

    def forward(self, states, task_ids=None, visual=None):
        """states : (B, S, K, dim) tokens LSP ; task_ids : (B,) ou None (pas de token de tâche) ; visual : features
        poolées (B, S, Pv, feature_dim) si le prédicteur a des tokens visuels -> h, out (B, S, K, dim)."""
        B, S, K, D = states.shape
        steps = self.step_embedding(torch.arange(S, device=states.device))[None, :, None]          # (1, S, 1, D)
        state_tokens = self.state_projection(states) + self.type_embedding[0] + steps
        motion_tokens = self.motion_queries.expand(B, S, K, D) + self.type_embedding[1] + steps
        blocks = [state_tokens, motion_tokens]
        if self.num_visual:
            blocks.insert(0, self.visual_projection(visual) + self.visual_position + self.type_embedding[2] + steps)
        block = self.num_visual + 2 * K
        x = torch.cat(blocks, dim=2).flatten(1, 2)                                                 # (B, S * block, D)
        offset = 0
        with_task = self.task_embedding is not None and task_ids is not None
        if with_task:
            task = self.task_embedding(task_ids % self.task_embedding.num_embeddings)[:, None]
            x, offset = torch.cat([task, x], dim=1), 1
        mask = self.attention_mask(S, states.device, with_task)

        motion = lambda x: x[:, offset:].unflatten(1, (S, block))[:, :, -K:]                      # (B, S, K, D)
        h = None
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x, src_mask=mask)
            if i == self.align_layer:
                h = motion(x)
        return h, motion(self.norm(x))


class ActionHead(nn.Module):
    """K tokens (B, S, K, dim) -> déplacement des 42 points de la main (B, S, 42, 3)."""

    def __init__(self, dim, num_queries, hidden_dim=1024, num_keypoints=NUM_HAND_KEYPOINTS):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.net = nn.Sequential(nn.LayerNorm(num_queries * dim), nn.Linear(num_queries * dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, num_keypoints * 3))

    def forward(self, tokens):
        return self.net(tokens.flatten(-2)).unflatten(-1, (self.num_keypoints, 3))


class JALA(nn.Module):
    def __init__(self,
                 feature_extractor: nn.Module,
                 feature_dim: int = 768,
                 image_size=(224, 384),
                 patch_size: int = 16,
                 dim: int = 512,
                 num_queries: int = 8,
                 perceiver_layers: int = 2,
                 perceiver_heads: int = 8,
                 predictor_layers: int = 6,
                 predictor_heads: int = 8,
                 align_layer: int = 4,
                 max_steps: int = 16,
                 num_tasks: int = 0,
                 dropout: float = 0.0,
                 pair_fusion: str = 'patch',
                 predictor_visual_pool: int = 2,
                 ema_decay: float = 0.999,
                 align_weight: float = 0.5,
                 align_norm: bool = True,
                 action_weight: float = 1.0,
                 labeled_fraction: float = 1.0,
                 probe_weight: float = 1.0,
                 z_action_weight: float = 0.0,
                 variance_weight: float = 0.0):
        """
        num_queries : K actions latentes par transition (moitié main gauche, moitié main droite)
        pair_fusion : combinaison des deux frames dans LAP / LSP, voir PerceiverBackbone
        predictor_visual_pool : le prédicteur voit aussi les patchs DINO de v_t, moyennés par blocs de pool x pool
                                (comme le VLM de JALA voit les tokens visuels) ; 0 : seulement les tokens LSP
        align_layer : couche du prédicteur dont l'état caché est h (le papier prend la couche 19 sur 28)
        num_tasks : taille de l'embedding de tâche (indice de tâche EgoDex modulo num_tasks) ; 0 le désactive
        labeled_fraction : part des clips dont la loss d'action est gardée ; < 1 simule l'entraînement hybride de JALA
                           (vidéos sans annotation de main : seule L_align)
        probe_weight : poids de la sonde sur z détaché (n'agit pas sur le modèle)
        Garde-fous contre l'effondrement, absents du papier (0 = JALA tel quel) :
        z_action_weight : tête d'action sur z *non détaché* (clips annotés) : le gradient des vraies actions atteint
                          les queries de LAP et ancre z sur le mouvement réel, y compris sa part imprévisible depuis le
                          passé, que L_align seule pousse à oublier
        variance_weight : terme de variance VICReg sur z (std par dimension sur le batch >= 1), sans annotation
        """
        super().__init__()
        assert num_queries % 2 == 0, "num_queries doit être pair (moitié par main)"
        image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        num_patches = grid_size[0] * grid_size[1]

        self.feature_extractor = feature_extractor
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        perceiver_kwargs = dict(feature_dim=feature_dim, out_dim=dim, grid_size=grid_size, num_layers=perceiver_layers,
                                num_heads=perceiver_heads, dropout=dropout, pair_fusion=pair_fusion)
        self.lsp = Perceiver(num_queries, dim, **perceiver_kwargs)
        self.lap = copy.deepcopy(self.lsp)                     # même architecture et mêmes poids au départ
        # découplage : backbone appris côté LSP, queries apprises côté LAP ; l'autre moitié suit en EMA
        self.lap.backbone.requires_grad_(False)
        self.lsp.queries.requires_grad_(False)

        self.grid_size, self.visual_pool = grid_size, predictor_visual_pool
        visual_grid = None
        if predictor_visual_pool:
            assert grid_size[0] % predictor_visual_pool == 0 and grid_size[1] % predictor_visual_pool == 0, (
                f"grille {grid_size} non divisible par predictor_visual_pool={predictor_visual_pool}")
            visual_grid = (grid_size[0] // predictor_visual_pool, grid_size[1] // predictor_visual_pool)
        self.predictor = LatentPredictor(dim, num_queries, predictor_layers, predictor_heads, align_layer, max_steps,
                                         num_tasks, dropout, feature_dim, visual_grid)
        self.action_head = ActionHead(dim, num_queries)
        self.probe = ActionHead(dim, num_queries)
        self.z_action_head = ActionHead(dim, num_queries) if z_action_weight > 0 else None

        self.num_patches = num_patches
        self.ema_decay = ema_decay
        self.align_weight, self.align_norm = align_weight, align_norm
        self.action_weight, self.labeled_fraction, self.probe_weight = action_weight, labeled_fraction, probe_weight
        self.z_action_weight, self.variance_weight = z_action_weight, variance_weight

    @torch.no_grad()
    def ema_update(self):
        """EMA asymétrique de JALA (éq. 4-5), à appeler après chaque pas d'optimiseur."""
        a = self.ema_decay
        for target, source in zip(self.lap.backbone.parameters(), self.lsp.backbone.parameters()):
            target.lerp_(source.detach(), 1 - a)
        for target, source in zip(self.lap.backbone.buffers(), self.lsp.backbone.buffers()):
            target.copy_(source)
        self.lsp.queries.lerp_(self.lap.queries.detach(), 1 - a)

    @torch.no_grad()
    def extract_features(self, images):
        features = self.feature_extractor(images).float()                         # (B, T, P, feature_dim)
        assert features.shape[2] == self.num_patches, (
            f"{features.shape[2]} patchs DINO, {self.num_patches} attendus : vérifier image_size / patch_size")
        return features

    def latent_actions(self, features):
        """LAP sur chaque transition : features (B, T, P, F) -> z (B, T-1, K, dim)."""
        B, T = features.shape[:2]
        z = self.lap(features[:, :-1].flatten(0, 1), features[:, 1:].flatten(0, 1))
        return z.unflatten(0, (B, T - 1))

    def latent_states(self, features):
        """LSP sur la frame dupliquée (v_t, v_t), pour t = 0..T-2 : -> (B, T-1, K, dim)."""
        B, T = features.shape[:2]
        current = features[:, :-1].flatten(0, 1)
        return self.lsp(current, current).unflatten(0, (B, T - 1))

    def visual_tokens(self, features):
        """Patchs DINO des frames 0..T-2 moyennés par blocs pool x pool : (B, T-1, Pv, F), ou None."""
        if not self.visual_pool:
            return None
        B, T, P, F_ = features.shape
        x = features[:, :-1].flatten(0, 1).transpose(1, 2).unflatten(2, self.grid_size)            # (B(T-1), F, h, w)
        x = F.avg_pool2d(x, self.visual_pool).flatten(2).transpose(1, 2)                          # (B(T-1), Pv, F)
        return x.unflatten(0, (B, T - 1))

    def _normalize(self, x):
        return F.layer_norm(x, x.shape[-1:]) if self.align_norm else x

    def align_loss(self, h, z):
        """L1 par token (somme sur les dimensions dans le papier ; ici moyenne, l'échelle étant portée par align_weight)."""
        return (self._normalize(h) - self._normalize(z)).abs().mean()

    def forward(self, images, actions=None, task_ids=None, labeled=None):
        """images (B, T, 3, H, W) ; actions EgoDex (B, T-1, 88, 3) ou None ; task_ids (B,) ; labeled (B,) booléen ou
        None (tirage avec labeled_fraction à l'entraînement, tout annoté sinon)."""
        features = self.extract_features(images)
        z = self.latent_actions(features)                          # cible : gradient vers les queries de LAP seulement
        states = self.latent_states(features)                      # contexte : gradient vers le backbone de LSP
        h, out = self.predictor(states, task_ids, self.visual_tokens(features))

        loss_align = self.align_loss(h, z)
        output = dict(z=z, h=h, loss_align=loss_align, features=features)
        loss = self.align_weight * loss_align
        if self.variance_weight > 0:
            std = z.float().flatten(0, 2).std(0)                                             # par dimension
            loss_variance = F.relu(1 - std).mean()
            output['loss_variance'] = loss_variance
            loss = loss + self.variance_weight * loss_variance

        if actions is not None:
            target = hand_motion_target(actions.float())                                     # (B, S, 42, 3) cm
            if labeled is None:
                labeled = (torch.rand(images.shape[0], device=images.device) < self.labeled_fraction) if self.training \
                    else torch.ones(images.shape[0], dtype=torch.bool, device=images.device)
            prediction = self.action_head(out)
            error = (prediction - target).abs().mean((1, 2, 3))                              # (B,)
            weight = labeled.float()
            loss_action = (weight * error).sum() / weight.sum().clamp(min=1)
            probe_error = (self.probe(z.detach()) - target).abs()
            loss_probe = probe_error.mean()
            loss = loss + self.action_weight * loss_action + self.probe_weight * loss_probe
            if self.z_action_head is not None:
                z_error = (self.z_action_head(z) - target).abs().mean((1, 2, 3))
                loss_z_action = (weight * z_error).sum() / weight.sum().clamp(min=1)
                output['loss_z_action'] = loss_z_action
                loss = loss + self.z_action_weight * loss_z_action
            output.update(loss_action=loss_action, loss_probe=loss_probe, action_prediction=prediction,
                          action_target=target, labeled_frac=weight.mean(),
                          action_mae=error.mean().detach(),                                 # sur tous les clips, en cm
                          action_mae_zero=target.abs().mean().detach(),                     # référence : main immobile
                          probe_mae=probe_error.mean().detach())
        output['loss'] = loss
        return output

    @torch.no_grad()
    def diagnostics(self, output):
        """Indicateurs d'effondrement et de spécificité de l'alignement."""
        z, h = output['z'].float(), output['h'].float()
        zn, hn = self._normalize(z), self._normalize(h)
        B = z.shape[0]
        metrics = dict(
            z_std=z.flatten(0, 1).std(0).mean(),                           # ~0 : actions latentes constantes
            h_std=h.flatten(0, 1).std(0).mean(),
            z_norm=z.norm(dim=-1).mean(),
            z_h_cosine=F.cosine_similarity(z, h, dim=-1).mean(),
            # std de z entre transitions d'un même clip, rapportée à la std totale : ~0 si z ne décrit que la scène
            z_temporal_std_ratio=(zn.std(1).mean() / zn.flatten(0, 1).std(0).mean().clamp(min=1e-8)) if z.shape[1] > 1
            else torch.tensor(float('nan')),
        )
        if B > 1:
            shuffled = zn[torch.roll(torch.arange(B, device=z.device), 1)]
            metrics['loss_align_shuffled'] = (hn - shuffled).abs().mean()
            # < 1 : h est plus proche de son propre z que de celui d'un autre clip ; ~1 : alignement non spécifique
            metrics['align_specificity'] = (hn - zn).abs().mean() / metrics['loss_align_shuffled'].clamp(min=1e-8)
        return metrics
