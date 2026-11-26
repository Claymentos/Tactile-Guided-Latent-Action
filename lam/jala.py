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

Deux options indépendantes choisissent entre la version du papier et la version compacte :
  predictor
    'vlm'         : le papier. InternVL3-2B (lam.jala_vlm.VLMPredictor) voit l'instruction, la frame initiale v_0 et les
                    tokens LSP(v_0, v_0), puis les tokens de mouvement de chaque chunk ; h = couche 19 sur 28.
    'transformer' : LatentPredictor, un petit transformer causal dans le temps : à la transition t, il voit les patchs
                    DINO poolés et les tokens d'état LSP des frames 0..t (et la tâche EgoDex, à la place de
                    l'instruction), et K tokens de mouvement dont l'état caché à la couche align_layer est h_t.
  action_objective
    'mcp'         : le papier (motion-token prediction, éq. 2). Les chunks de mouvement des mains sont tokenisés par un
                    tokenizer GRVQ figé (lam.motion_tokenizer, appris avant), les tokens entrent dans le prédicteur,
                    masqués selon le masquage hybride du papier (un chunk cible masqué à un taux aléatoire, les chunks
                    précédents intacts, les suivants masqués à 5 %), et la cross-entropy porte sur les tokens masqués.
                    K = nombre de tokens du tokenizer (128 dans le papier). En validation, tout est masqué : h et la
                    prédiction viennent seulement du passé.
    'regression'  : version compacte. Les tokens de mouvement sont K queries apprises et la loss est une régression L1
                    du déplacement 3D des 42 points des mains EgoDex entre t et t+δ (en cm, repère caméra de la
                    première frame), prédit depuis la sortie du prédicteur.
align_norm normalise h et z (LayerNorm sans paramètres) avant la L1 : sans le VLM pré-entraîné qui fixe l'échelle de h,
les deux côtés pourraient sinon rétrécir ensemble vers 0. align_norm=False donne la L1 brute du papier.
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
WRIST = (0, 21)                                         # poignet gauche, poignet droit
FINGERTIPS = tuple(o + k for o in (0, 21) for k in (4, 8, 12, 16, 20))
FINGERS = tuple(i for i in range(NUM_HAND_KEYPOINTS) if i not in WRIST)


def finger_motion(motion):
    """Déplacement (..., 42, 3) -> déplacement des 40 points des doigts relatif au poignet de leur main (..., 40, 3) :
    l'articulation des doigts, sans la translation / rotation globale de la main portée par le poignet."""
    wrist = torch.cat([motion[..., :1, :].expand(*motion.shape[:-2], 20, 3),
                       motion[..., 21:22, :].expand(*motion.shape[:-2], 20, 3)], dim=-2)
    return motion[..., FINGERS, :] - wrist


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
        self.keep_attention, self.attention = False, None

    def forward(self, queries, x):
        x = self.norm_x(x)
        q = self.norm_q(queries)
        attended, weights = self.cross_attention(q, x, x, need_weights=self.keep_attention, average_attn_weights=True)
        if self.keep_attention:
            self.attention = weights                                     # (B, K, N), moyenne sur les têtes
        queries = queries + attended
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

    def forward(self, states, task_ids=None, visual=None, motion_inputs=None):
        """states : (B, S, K, dim) tokens LSP ; task_ids : (B,) ou None (pas de token de tâche) ; visual : features
        poolées (B, S, Pv, feature_dim) si le prédicteur a des tokens visuels ; motion_inputs (B, S, K, dim) :
        embeddings des tokens de mouvement (MCP), ou None (queries apprises) -> h, out (B, S, K, dim)."""
        B, S, K, D = states.shape
        steps = self.step_embedding(torch.arange(S, device=states.device))[None, :, None]          # (1, S, 1, D)
        state_tokens = self.state_projection(states) + self.type_embedding[0] + steps
        motion = self.motion_queries.expand(B, S, K, D) if motion_inputs is None else motion_inputs + self.motion_queries
        motion_tokens = motion + self.type_embedding[1] + steps
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
    """K tokens (B, S, K, dim) -> déplacement des 42 points de la main (B, S, 42, 3). Si K x dim dépasse max_input
    (K = 128 tokens de 1536 avec le VLM), chaque token est d'abord projeté en petite dimension."""

    def __init__(self, dim, num_queries, hidden_dim=1024, num_keypoints=NUM_HAND_KEYPOINTS, max_input=8192):
        super().__init__()
        self.num_keypoints = num_keypoints
        token_dim = dim if num_queries * dim <= max_input else max(8, max_input // num_queries)
        self.token_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, token_dim)) if token_dim != dim else None
        self.net = nn.Sequential(nn.LayerNorm(num_queries * token_dim), nn.Linear(num_queries * token_dim, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, num_keypoints * 3))

    def forward(self, tokens):
        if self.token_projection is not None:
            tokens = self.token_projection(tokens)
        return self.net(tokens.flatten(-2)).unflatten(-1, (self.num_keypoints, 3))


class MotionTokens(nn.Module):
    """Entrées et sortie des tokens de mouvement de la loss MCP : embedding des codes GRVQ (une table par partie,
    poignet / doigts, plus un token [MASK]) et tête de classification sur le codebook de la partie de chaque token."""

    def __init__(self, dim, token_part, codebook_size, num_parts=2):
        super().__init__()
        self.codebook_size, self.num_parts = codebook_size, num_parts
        self.register_buffer('token_part', token_part.clone().long(), persistent=False)            # (K,)
        self.mask_index = num_parts * codebook_size
        self.embedding = nn.Embedding(num_parts * codebook_size + 1, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_parts * codebook_size)

    def embed(self, tokens, masked):
        """tokens (B, S, K) indices dans le codebook de leur partie ; masked (B, S, K) booléen -> (B, S, K, dim)."""
        index = tokens + self.token_part * self.codebook_size
        return self.embedding(torch.where(masked, torch.full_like(index, self.mask_index), index))

    def logits(self, out):
        """(B, S, K, dim) -> logits (B, S, K, codebook_size) sur le codebook de la partie de chaque token."""
        logits = self.head(self.norm(out)).unflatten(-1, (self.num_parts, self.codebook_size))
        part = self.token_part.view(1, 1, -1, 1, 1).expand(*logits.shape[:3], 1, self.codebook_size)
        return logits.gather(3, part).squeeze(3)


def hybrid_mask(batch_size, steps, num_tokens, later_ratio=0.05, device=None):
    """Masquage hybride du papier : par clip, un chunk cible tiré au hasard est masqué à un taux aléatoire (au moins un
    token), les chunks précédents restent intacts et les suivants sont masqués à later_ratio -> (B, S, K) booléen."""
    target = torch.randint(steps, (batch_size,), device=device)
    ratio = torch.rand(batch_size, device=device)
    draw = torch.rand(batch_size, steps, num_tokens, device=device)
    step = torch.arange(steps, device=device)[None, :, None]
    target_mask = draw < ratio[:, None, None]
    forced = torch.zeros_like(target_mask)
    forced[torch.arange(batch_size, device=device), :, torch.randint(num_tokens, (batch_size,), device=device)] = True
    target_mask = target_mask | forced
    t = target[:, None, None]
    return torch.where(step == t, target_mask, (step > t) & (draw < later_ratio))


class ForwardProbe(nn.Module):
    """Sonde de dynamique directe : (features DINO normalisées de v_t, actions latentes z_t) -> features normalisées de
    v_t+1. Entraînée sur des entrées détachées, elle n'agit pas sur JALA (qui n'a pas de générateur) ; elle sert au test
    de cycle-consistance, où il faut appliquer les actions d'une vidéo à une autre. Prédit un résidu sur les patchs."""

    def __init__(self, feature_dim, latent_dim, grid_size, dim=256, n_layer=4, num_heads=8):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, dim)
        self.latent_projection = nn.Linear(latent_dim, dim)
        self.type_embedding = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.register_buffer('position_embedding', sincos_2d(grid_size, dim), persistent=False)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, num_heads, 4 * dim, 0.0, activation='gelu',
                                                                batch_first=True, norm_first=True) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, feature_dim)
        nn.init.zeros_(self.output.weight), nn.init.zeros_(self.output.bias)      # départ : v_t+1 = v_t

    def forward(self, features, z):
        """features (N, P, F) normalisées ; z (N, K, latent_dim) -> features normalisées prédites (N, P, F)."""
        P = features.shape[1]
        x = torch.cat([self.input_projection(features) + self.position_embedding + self.type_embedding[0],
                       self.latent_projection(z) + self.type_embedding[1]], dim=1)
        for layer in self.layers:
            x = layer(x)
        return features + self.output(self.norm(x[:, :P]))


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
                 variance_weight: float = 0.0,
                 forward_probe_weight: float = 1.0,
                 forward_probe_dim: int = 256,
                 forward_probe_layers: int = 4,
                 predictor: str = 'transformer',
                 action_objective: str = 'regression',
                 motion_tokenizer_path: str = None,
                 mcp_later_mask_ratio: float = 0.05,
                 vlm_path: str = 'OpenGVLab/InternVL3-2B-hf',
                 vlm_freeze_vision: bool = True,
                 vlm_gradient_checkpointing: bool = True,
                 vlm_image_size: int = 448):
        """
        predictor : 'transformer' (LatentPredictor) ou 'vlm' (InternVL3, lam.jala_vlm.VLMPredictor, comme le papier) ;
                    avec 'vlm', align_layer est la couche du LLM (19 dans le papier), dim reste la largeur des
                    Perceivers et z / h ont la dimension du VLM (1536) ; predictor_layers / heads / visual_pool,
                    max_steps et num_tasks sont ignorés (l'instruction remplace la tâche)
        action_objective : 'regression' (loss_action L1 sur le déplacement des 42 points) ou 'mcp' (cross-entropy sur
                    les tokens GRVQ masqués, comme le papier ; action_weight est alors le poids de L_MCP)
        motion_tokenizer_path : checkpoint de lam.lightning.motion_tokenizer.MotionTokenizerTraining (MCP) ; num_queries
                    doit valoir le nombre de tokens du tokenizer
        mcp_later_mask_ratio : taux de masquage des chunks après le chunk cible (5 % dans le papier)
        vlm_path : InternVL3 au format transformers (dossier local hors ligne, ou identifiant Hugging Face)
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
        forward_probe_weight : poids de la sonde de dynamique directe (ForwardProbe, entrées détachées : n'agit pas sur
                               JALA) nécessaire au test de cycle-consistance ; 0 la désactive (pas de cycle)
        """
        super().__init__()
        assert num_queries % 2 == 0, "num_queries doit être pair (moitié par main)"
        assert predictor in ('transformer', 'vlm'), predictor
        assert action_objective in ('regression', 'mcp'), action_objective
        self.predictor_type, self.action_objective = predictor, action_objective
        self.mcp_later_mask_ratio = mcp_later_mask_ratio
        self.motion_tokenizer = None
        if action_objective == 'mcp':
            from lam.motion_tokenizer import MotionTokenizer
            assert motion_tokenizer_path, "action_objective='mcp' : motion_tokenizer_path nécessaire (tokenizer GRVQ appris)"
            self.motion_tokenizer = MotionTokenizer.from_checkpoint(motion_tokenizer_path)
            assert num_queries == self.motion_tokenizer.num_tokens, (
                f"num_queries={num_queries} : le tokenizer produit {self.motion_tokenizer.num_tokens} tokens par chunk")
        image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        num_patches = grid_size[0] * grid_size[1]

        self.feature_extractor = feature_extractor
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        if predictor == 'vlm':
            from lam.jala_vlm import VLMPredictor
            self.predictor = VLMPredictor(vlm_path, num_queries, align_layer, freeze_vision=vlm_freeze_vision,
                                          gradient_checkpointing=vlm_gradient_checkpointing, image_size=vlm_image_size)
            latent_dim = self.predictor.dim               # les têtes de LAP / LSP projettent dans l'espace du VLM
        else:
            latent_dim = dim
        self.latent_dim = latent_dim
        perceiver_kwargs = dict(feature_dim=feature_dim, out_dim=latent_dim, grid_size=grid_size, num_layers=perceiver_layers,
                                num_heads=perceiver_heads, dropout=dropout, pair_fusion=pair_fusion)
        self.lsp = Perceiver(num_queries, dim, **perceiver_kwargs)
        self.lap = copy.deepcopy(self.lsp)                     # même architecture et mêmes poids au départ
        # découplage : backbone appris côté LSP, queries apprises côté LAP ; l'autre moitié suit en EMA
        self.lap.backbone.requires_grad_(False)
        self.lsp.queries.requires_grad_(False)

        self.grid_size, self.visual_pool = grid_size, predictor_visual_pool if predictor == 'transformer' else 0
        if predictor == 'transformer':
            visual_grid = None
            if predictor_visual_pool:
                assert grid_size[0] % predictor_visual_pool == 0 and grid_size[1] % predictor_visual_pool == 0, (
                    f"grille {grid_size} non divisible par predictor_visual_pool={predictor_visual_pool}")
                visual_grid = (grid_size[0] // predictor_visual_pool, grid_size[1] // predictor_visual_pool)
            self.predictor = LatentPredictor(dim, num_queries, predictor_layers, predictor_heads, align_layer, max_steps,
                                             num_tasks, dropout, feature_dim, visual_grid)
        self.action_head = ActionHead(latent_dim, num_queries) if action_objective == 'regression' else None
        self.motion_tokens = MotionTokens(latent_dim, self.motion_tokenizer.token_part, self.motion_tokenizer.codebook_size) \
            if action_objective == 'mcp' else None
        self.probe = ActionHead(latent_dim, num_queries)
        self.z_action_head = ActionHead(latent_dim, num_queries) if z_action_weight > 0 else None
        self.forward_probe = ForwardProbe(feature_dim, latent_dim, grid_size, forward_probe_dim, forward_probe_layers) \
            if forward_probe_weight > 0 else None
        self.forward_probe_weight = forward_probe_weight

        self.num_patches = num_patches
        self.ema_decay = ema_decay
        self.align_weight, self.align_norm = align_weight, align_norm
        self.action_weight, self.labeled_fraction, self.probe_weight = action_weight, labeled_fraction, probe_weight
        self.z_action_weight, self.variance_weight = z_action_weight, variance_weight

    def train(self, mode=True):
        super().train(mode)
        if self.motion_tokenizer is not None:
            self.motion_tokenizer.eval()          # figé : pas de mise à jour EMA / k-means des codebooks GRVQ
        return self

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

    def motion_inputs(self, hand_chunks, labeled, batch_size, steps, device):
        """MCP : tokens GRVQ des chunks (B, S, K), masque (B, S, K) et embeddings d'entrée du prédicteur. Masquage
        hybride à l'entraînement, tout masqué en évaluation et pour les clips non annotés."""
        K = self.motion_tokenizer.num_tokens
        if hand_chunks is None:
            tokens = torch.zeros(batch_size, steps, K, dtype=torch.long, device=device)
            masked = torch.ones_like(tokens, dtype=torch.bool)
        else:
            tokens = self.motion_tokenizer.encode(hand_chunks.to(device).float())
            masked = hybrid_mask(batch_size, steps, K, self.mcp_later_mask_ratio, device) if self.training \
                else torch.ones_like(tokens, dtype=torch.bool)
            masked = masked | ~labeled[:, None, None]
        return tokens, masked, self.motion_tokens.embed(tokens, masked)

    def forward(self, images, actions=None, task_ids=None, labeled=None, hand_chunks=None, instructions=None):
        """images (B, T, 3, H, W) ; actions EgoDex (B, T-1, 88, 3) ou None ; task_ids (B,) ; labeled (B,) booléen ou
        None (tirage avec labeled_fraction à l'entraînement, tout annoté sinon) ; hand_chunks (B, T-1, C, 42, 3) :
        trajectoires des mains (metadata['hand_chunks'], nécessaire pour MCP) ; instructions : B textes (VLM)."""
        features = self.extract_features(images)
        z = self.latent_actions(features)                          # cible : gradient vers les queries de LAP seulement
        B, S = z.shape[:2]
        if labeled is None:
            labeled = (torch.rand(B, device=images.device) < self.labeled_fraction) if self.training \
                else torch.ones(B, dtype=torch.bool, device=images.device)
        mcp_tokens = mcp_masked = motion_inputs = None
        if self.action_objective == 'mcp':
            mcp_tokens, mcp_masked, motion_inputs = self.motion_inputs(hand_chunks, labeled, B, S, images.device)
        if self.predictor_type == 'vlm':
            initial = features[:, 0]
            states = self.lsp(initial, initial)                    # contexte LSP(v_0, v_0) : gradient vers le backbone de LSP
            h, out = self.predictor(states, images[:, 0], instructions, motion_inputs, steps=S)
            h, out = h.float(), out.float()
        else:
            states = self.latent_states(features)                  # contexte : gradient vers le backbone de LSP
            h, out = self.predictor(states, task_ids, self.visual_tokens(features), motion_inputs)

        loss_align = self.align_loss(h, z)
        output = dict(z=z, h=h, loss_align=loss_align, features=features)
        loss = self.align_weight * loss_align
        mcp_prediction = None
        if self.action_objective == 'mcp' and hand_chunks is not None:
            logits = self.motion_tokens.logits(out).float()                                  # (B, S, K, codebook)
            selected = mcp_masked & labeled[:, None, None]
            loss_mcp = F.cross_entropy(logits[selected], mcp_tokens[selected]) if selected.any() else logits.sum() * 0
            predicted = logits.argmax(-1)
            loss = loss + self.action_weight * loss_mcp
            output.update(loss_mcp=loss_mcp, mcp_tokens=mcp_tokens, mcp_masked=mcp_masked,
                          mcp_accuracy=(predicted == mcp_tokens)[mcp_masked].float().mean().detach(),
                          mcp_masked_frac=mcp_masked.float().mean().detach())
            # prédiction en un passage (argmax sur les tokens masqués), décodée en déplacement à la fin du chunk
            mcp_prediction = self.motion_tokenizer.decode(torch.where(mcp_masked, predicted, mcp_tokens))[..., -1, :, :]
            output['tokenizer_reconstruction'] = self.motion_tokenizer.decode(mcp_tokens)[..., -1, :, :]
        if self.variance_weight > 0:
            std = z.float().flatten(0, 2).std(0)                                             # par dimension
            loss_variance = F.relu(1 - std).mean()
            output['loss_variance'] = loss_variance
            loss = loss + self.variance_weight * loss_variance

        if self.forward_probe is not None:
            normed = F.layer_norm(features, features.shape[-1:])
            B, S = z.shape[:2]
            predicted = self.forward_probe(normed[:, :-1].flatten(0, 1), z.detach().flatten(0, 1))
            loss_forward_probe = F.mse_loss(predicted, normed[:, 1:].flatten(0, 1))
            output.update(loss_forward_probe=loss_forward_probe,
                          forward_probe_copy=F.mse_loss(normed[:, :-1], normed[:, 1:]).detach())   # référence : v_t+1 = v_t
            loss = loss + self.forward_probe_weight * loss_forward_probe

        if actions is not None:
            target = hand_motion_target(actions.float())                                     # (B, S, 42, 3) cm
            weight = labeled.float()
            if self.action_objective == 'regression':
                prediction = self.action_head(out)
                error = (prediction - target).abs().mean((1, 2, 3))                          # (B,)
                loss_action = (weight * error).sum() / weight.sum().clamp(min=1)
                loss = loss + self.action_weight * loss_action
                output['loss_action'] = loss_action
            else:
                prediction = mcp_prediction if mcp_prediction is not None else torch.zeros_like(target)
                error = (prediction - target).abs().mean((1, 2, 3))
                if 'tokenizer_reconstruction' in output:       # plancher : erreur du tokenizer seul (tokens réels)
                    output['tokenizer_mae'] = (output['tokenizer_reconstruction'] - target).abs().mean().detach()
            probe_prediction = self.probe(z.detach())
            probe_error = (probe_prediction - target).abs()
            loss_probe = probe_error.mean()
            loss = loss + self.probe_weight * loss_probe
            if self.z_action_head is not None:
                z_error = (self.z_action_head(z) - target).abs().mean((1, 2, 3))
                loss_z_action = (weight * z_error).sum() / weight.sum().clamp(min=1)
                output['loss_z_action'] = loss_z_action
                loss = loss + self.z_action_weight * loss_z_action
            relative = finger_motion(target)
            output.update(probe_finger_mae=(finger_motion(probe_prediction) - relative).abs().mean().detach(),
                          finger_motion_zero=relative.abs().mean().detach(),           # référence : doigts figés
                          probe_wrist_mae=(probe_prediction[..., WRIST, :] - target[..., WRIST, :]).abs().mean().detach(),
                          wrist_motion_zero=target[..., WRIST, :].abs().mean().detach())
            output.update(loss_probe=loss_probe, action_prediction=prediction,
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

    @torch.no_grad()
    def lap_attention(self, features):
        """Attention croisée des queries de LAP sur les patchs, moyennée sur les têtes et les couches Perceiver :
        (B, T-1, K, P), chaque query sommant à 1 sur les patchs (avec pair_fusion='sequence', les deux frames sont
        additionnées)."""
        layers = self.lap.backbone.layers
        for layer in layers:
            layer.keep_attention = True
        try:
            self.latent_actions(features)
            attention = torch.stack([layer.attention.float() for layer in layers]).mean(0)    # (B(T-1), K, N)
        finally:
            for layer in layers:
                layer.keep_attention, layer.attention = False, None
        P = self.num_patches
        if attention.shape[-1] == 2 * P:
            attention = attention[..., :P] + attention[..., P:]
        B, T = features.shape[:2]
        return attention.unflatten(0, (B, T - 1))

    @torch.no_grad()
    def cycle_consistency(self, images_a, images_b, actions_a=None):
        """Cycle A -> B -> A des actions latentes (figure 5 b de arXiv 2601.05230), adapté à JALA.

        1. z_A = LAP(A_t, A_t+1)
        2. B'_t+1 = ForwardProbe(B_t, z_A,t) : les actions de A appliquées à la scène B (un pas depuis la vraie frame B_t)
        3. z_cycle = LAP(B_t, B'_t+1)
        4. on décode z_A, z_cycle et z_hasard (z d'un autre clip du batch) en mouvement des mains avec la sonde, et on
           compare au vrai mouvement de A ; et on prédit A_t+1 avec la ForwardProbe pour chacun.
        Renvoie les tenseurs nécessaires aux métriques et aux visualisations.
        """
        assert self.forward_probe is not None, "cycle-consistance : forward_probe_weight > 0 nécessaire"
        fa, fb = self.extract_features(images_a), self.extract_features(images_b)
        na, nb = F.layer_norm(fa, fa.shape[-1:]), F.layer_norm(fb, fb.shape[-1:])
        B, T = fa.shape[:2]
        z_a = self.latent_actions(fa)                                                      # (B, S, K, D)
        transferred = self.forward_probe(nb[:, :-1].flatten(0, 1), z_a.flatten(0, 1)).unflatten(0, (B, T - 1))
        # LAP sur (B_t, B'_t+1) ; les features normalisées passent telles quelles (LayerNorm idempotente)
        z_cycle = self.lap(nb[:, :-1].flatten(0, 1), transferred.flatten(0, 1)).unflatten(0, (B, T - 1))
        z_random = z_a[torch.roll(torch.arange(B, device=z_a.device), max(1, B // 3))]
        latents = dict(original=z_a, cycle=z_cycle, random=z_random)

        out = dict(z_original=z_a, z_cycle=z_cycle, transferred=transferred, features_b=nb,
                   z_cosine_cycle=F.cosine_similarity(z_a, z_cycle, dim=-1).mean((1, 2)),
                   z_cosine_random=F.cosine_similarity(z_a, z_random, dim=-1).mean((1, 2)),
                   transfer_feature_error=((transferred - nb[:, 1:]) ** 2).mean((1, 2, 3)))  # B' vs vraie B_t+1
        for name, z in latents.items():
            predicted = self.forward_probe(na[:, :-1].flatten(0, 1), z.flatten(0, 1)).unflatten(0, (B, T - 1))
            out[f'feature_error_{name}'] = ((predicted - na[:, 1:]) ** 2).mean((1, 2, 3))      # (B,)
            out[f'hands_{name}'] = self.probe(z)                                               # (B, S, 42, 3) cm
        if actions_a is not None:
            target = hand_motion_target(actions_a.float())
            out['hands_target'] = target
            for name in latents:
                out[f'hand_error_{name}'] = (out[f'hands_{name}'] - target).abs().mean((1, 2, 3))
        return out


CYCLE_METRICS = ('hand_error_original', 'hand_error_cycle', 'hand_error_random', 'feature_error_original',
                 'feature_error_cycle', 'feature_error_random', 'z_cosine_cycle', 'z_cosine_random',
                 'transfer_feature_error')


def cycle_summary(means):
    """Moyennes des métriques par clip -> indicateurs. hausse_relative : (cycle - origine) / (hasard - origine), 0 =
    transfert parfait, 1 = pas mieux qu'une action d'un autre clip."""
    summary = dict(means)
    for space in ('hand', 'feature'):
        o, c, r = (means.get(f'{space}_error_{k}') for k in ('original', 'cycle', 'random'))
        if o is None:
            continue
        summary[f'{space}_hausse_erreur'] = c - o
        summary[f'{space}_hausse_relative'] = (c - o) / (r - o) if r - o > 0 else float('nan')
    return summary


def keypoint_patch_masks(pixels, valid, grid, patch_size, dilation=0):
    """Pixels (..., 42, 2) et validité (..., 42) -> masques de patchs (..., P) pour la main entière, les doigts, les
    bouts des doigts, le poignet, la main gauche et la main droite : un patch est marqué s'il contient un point
    (élargi de `dilation` patchs)."""
    gh, gw = grid
    col = (pixels[..., 0] / patch_size).floor().long().clamp(0, gw - 1)
    row = (pixels[..., 1] / patch_size).floor().long().clamp(0, gh - 1)
    index = row * gw + col                                                                  # (..., 42)

    def mask(points):
        m = torch.zeros(*pixels.shape[:-2], gh * gw, device=pixels.device)
        idx, ok = index[..., list(points)], valid[..., list(points)].float()
        m.scatter_add_(-1, idx, ok)
        m = (m > 0).float()
        if dilation:
            shape = m.shape
            m = F.max_pool2d(m.reshape(-1, 1, gh, gw), 2 * dilation + 1, 1, dilation).reshape(shape)
        return m

    return dict(hand=mask(range(42)), fingers=mask(FINGERS), fingertips=mask(FINGERTIPS), wrist=mask(WRIST),
                left=mask(range(21)), right=mask(range(21, 42)))


def attention_focus(attention, masks):
    """attention (B, S, K, P), masques (B, S, P) -> par région : masse d'attention (moyenne des queries), surface
    (part des patchs) et lift = masse / surface (1 = pas mieux qu'une attention uniforme). Les queries sont séparées en
    deux groupes (première moitié : tête gauche, seconde : tête droite) pour mesurer la spécialisation par main.
    Renvoie des sommes et un compte, pour une moyenne sur plusieurs batches."""
    K = attention.shape[2]
    mean = attention.mean(2)                                                                # (B, S, P)
    has_hand = masks['hand'].sum(-1) > 0                                                    # transitions avec une main visible
    out = {}
    for region in ('hand', 'fingers', 'fingertips', 'wrist'):
        m = masks[region]
        mass, area = (mean * m).sum(-1), m.mean(-1)
        ok = has_hand & (area > 0)
        out[f'{region}_mass'] = (mass[ok].sum(), ok.sum())
        out[f'{region}_lift'] = ((mass[ok] / area[ok]).sum(), ok.sum())
    left_q, right_q = attention[:, :, :K // 2].mean(2), attention[:, :, K // 2:].mean(2)
    both = (masks['left'].sum(-1) > 0) & (masks['right'].sum(-1) > 0)
    # queries gauches : part de leur attention sur la main gauche parmi les deux mains (0.5 = pas de spécialisation)
    for name, q, own, other in (('left_queries_own_hand', left_q, 'left', 'right'),
                                ('right_queries_own_hand', right_q, 'right', 'left')):
        a, b = (q * masks[own]).sum(-1), (q * masks[other]).sum(-1)
        share = a / (a + b).clamp(min=1e-8)
        out[name] = (share[both].sum(), both.sum())
    return out
