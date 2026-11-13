import torch

from torch import nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, model_dim, d_feedforward, dropout):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, d_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_feedforward, model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class SpatialTemporalLayer(nn.Module):
    """Une couche du transformer spatio-temporel (factorisé, comme dans Genie).

    Chaque token est un patch d'une frame. La couche enchaîne trois sous-blocs, chacun en "pre-norm" + résiduel
    (x = x + bloc(LayerNorm(x))) :
        1. attention spatiale  : chaque token regarde les tokens de SA frame (mêmes t, tous les patchs)
        2. attention temporelle : chaque token regarde le MÊME patch aux autres frames (causale : seulement le passé)
        3. feed-forward         : MLP appliqué à chaque token séparément
    La forme d'entrée est conservée : (B, T, P, D) -> (B, T, P, D).

    Exemple utilisé dans les commentaires (train_tactile_lam, encodeur) : B=2 clips, T=5 frames, P=81 tokens
    (80 patchs + 1 readout), D=512.
    """

    def __init__(self, model_dim, num_spatial_heads, num_temporal_heads, dropout, is_causal):
        super().__init__()
        self.is_causal = is_causal  # True : un token de la frame t ne voit que les frames <= t
        # nn.MultiheadAttention calcule softmax(Q K^T / sqrt(d)) V sur l'axe 1 de son entrée ; batch_first=True signifie
        # que l'entrée est (lot, séquence, D). Q, K, V sont des projections apprises de D -> D, découpées en
        # num_heads têtes de D / num_heads dimensions (512 / 8 = 64). Le dropout s'applique aux poids d'attention.
        self.temporal_attention = nn.MultiheadAttention(model_dim, num_temporal_heads, dropout=dropout, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(model_dim, num_spatial_heads, dropout=dropout, batch_first=True)
        # MLP par token : D -> 4D -> D (512 -> 2048 -> 512) avec ReLU et dropout
        self.feed_forward = FeedForward(model_dim, 4*model_dim, dropout=dropout)

        # Une LayerNorm avant chaque sous-bloc (pre-norm). Note : ce n'est pas la normalisation QK de Genie, mais une
        # normalisation classique de chaque token (moyenne 0, variance 1 sur ses D valeurs, puis échelle et biais appris).
        self.norm1 = nn.LayerNorm(model_dim)          # Genie uses QK normalization
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)


    def forward(self, x, attention_mask=None):
        '''
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, patches, features)
        attention_mask: torch.BoolTensor (batch_size, seq_len, patches) ou None, True = token ignoré par l'attention
                        spatiale (utilisé seulement pour le dropout des registres, cf. LAM.forward)
        '''
        b, t, p, f = x.shape                                                            # (2, 5, 81, 512)

        # ---------- 1. Attention spatiale : les tokens d'une même frame échangent de l'information ----------
        # On range (clip, frame) dans l'axe "lot" pour que l'attention se fasse sur l'axe des patchs :
        # 10 séquences indépendantes de 81 tokens. Une frame ne voit pas les autres frames ici.
        x = rearrange(x, 'b t p f -> (b t) p f')                                        # (10, 81, 512)
        # Même réarrangement pour le masque : un masque de 81 booléens par (clip, frame)
        attention_mask = rearrange(attention_mask, 'b t p -> (b t) p') if attention_mask is not None else None   # (10, 81)
        x_norm = self.norm1(x)                                                          # pre-norm, (10, 81, 512)
        # Self-attention : query = key = value = x_norm. Pour chaque token i : poids a_ij = softmax_j(q_i.k_j / sqrt(64))
        # sur les 81 tokens j de la frame (par tête), puis sortie_i = somme_j a_ij v_j, têtes concaténées et projetées.
        # key_padding_mask : les tokens j à True reçoivent un score -inf, donc un poids 0 (personne ne les écoute).
        # [0] : la sortie ; need_weights=False évite de renvoyer (et calculer) la carte d'attention.
        attn_output = self.spatial_attention(x_norm, x_norm, x_norm, need_weights=False, key_padding_mask=attention_mask)[0]   # (10, 81, 512)
        x = x + attn_output                                                             # résiduel : on ajoute l'info reçue

        # ---------- 2. Attention temporelle : un même patch échange de l'information à travers les frames ----------
        # On range (clip, patch) dans l'axe "lot" pour que l'attention se fasse sur l'axe du temps :
        # 2 * 81 = 162 séquences de 5 tokens (le patch k aux frames 0..4). Le readout regarde les readouts des autres frames.
        x = rearrange(x, '(b t) p f -> (b p) t f', b=b, t=t)                            # (162, 5, 512)
        x_norm = self.norm2(x)                                                          # pre-norm, (162, 5, 512)
        if self.is_causal:
            # Masque (t, t) : 0 sur et sous la diagonale, -inf au-dessus. Ajouté aux scores avant le softmax, il donne un
            # poids 0 au futur : le token de la frame 3 n'écoute que les frames 0, 1, 2, 3. Ainsi le readout de la frame t
            # ne peut décrire que la transition (t-1 -> t), sans voir la frame t+1.
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(t, device=x.device)   # (5, 5)
            # is_causal=True indique à PyTorch que attn_mask est causal (il peut alors utiliser un noyau optimisé)
            attn_output = self.temporal_attention(x_norm, x_norm, x_norm, attn_mask=causal_mask, is_causal=True, need_weights=False)[0]   # (162, 5, 512)
        else:
            # Non causal : chaque frame voit toutes les frames, passées et futures
            attn_output = self.temporal_attention(x_norm, x_norm, x_norm, need_weights=False)[0]   # (162, 5, 512)

        x = x + attn_output                                                             # résiduel

        # ---------- 3. Feed-forward : chaque token "digère" ce qu'il a reçu, sans échange entre tokens ----------
        x_norm = self.norm3(x)                                                          # pre-norm, (162, 5, 512)
        ff_output = self.feed_forward(x_norm)                                           # 512 -> 2048 -> 512, (162, 5, 512)
        x = x + ff_output                                                               # résiduel

        # Retour à la forme d'entrée
        x = rearrange(x, '(b p) t f -> b t p f', b=b, p=p)                              # (2, 5, 81, 512)

        return x


class STTransformer(nn.Module):
    """Transformer spatio-temporel : une pile de SpatialTemporalLayer appliquée à une vidéo de tokens (B, T, P, D).

    Deux usages dans LAM :
      - encodeur (add_readout_token=True) : un token "readout", initialisé à zéro, est ajouté à chaque frame après ses
        patchs. Après les couches, le readout de la frame t résume ce qui a changé entre t-1 et t : c'est l'action latente
        (avant quantification). forward renvoie (tous les tokens, readouts).
      - décodeur (add_readout_token=False, actions données) : l'action est ajoutée à chaque patch de sa frame, et les
        tokens de sortie servent à prédire la frame suivante. forward renvoie les tokens.

    Exemple utilisé dans les commentaires (encodeur de train_tactile_lam) : images 128x160, patchs de 16 -> 8x10 = 80
    patchs, D=512, B=2 clips, T=5 frames.
    """

    def __init__(self,
                 model_dim,
                 n_layer,
                 num_spatial_heads,
                 num_temporal_heads,
                 dropout,
                 image_size,
                 patch_size,
                 max_seq_len=32,
                 add_readout_token=False,
                 is_causal=True):
        super().__init__()
        # image_size et patch_size acceptent un entier (carré) ou un couple (hauteur, largeur)
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)   # (128, 160)
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)   # (16, 16)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # Nombre de patchs par frame : (128 / 16) * (160 / 16) = 8 * 10 = 80
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.add_readout_token = add_readout_token #add a readout token per frame , a null vector. If the image has 80 patches, the readout token will be considered as the 81th. It will describe the action

        # Le readout occupe une position spatiale de plus (la 81e) : il lui faut sa propre ligne dans la table des
        # positions. Comme il est créé à zéro dans forward, c'est cette ligne (+ la position temporelle) qui lui donne
        # son identité.
        if add_readout_token:
            num_patches += 1                                                            # 81

        # Positions apprises : une table (num_patches, D) indexée par l'emplacement du token dans la frame, et une table
        # (max_seq_len, D) indexée par le numéro de frame. L'attention ignore l'ordre des tokens : sans ces vecteurs, le
        # modèle ne saurait ni où est un patch dans l'image, ni à quelle frame il appartient.
        self.spatial_pos_embedding = nn.Embedding(num_patches, model_dim)               # (81, 512)
        self.temporal_pos_embedding = nn.Embedding(max_seq_len, model_dim)              # (max_seq_len, 512), T <= max_seq_len

        self.transformer = nn.ModuleDict(dict(
            # n_layer couches identiques (mais avec des poids différents), appliquées l'une après l'autre
            layers  = nn.ModuleList([SpatialTemporalLayer(model_dim,
                                                          num_spatial_heads,
                                                          num_temporal_heads,
                                                          dropout,
                                                          is_causal=is_causal) for _ in range(n_layer)]),
            # LayerNorm finale : nécessaire en pre-norm, car la sortie de la dernière couche n'est pas normalisée
            final_norm = nn.LayerNorm(model_dim),
        ))


    def forward(self, x, actions=None, registers=None, attention_mask=None):
        '''
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        actions: torch.Tensor (batch_size, seq_len, model_dim) ou None, une action par frame (usage décodeur)
        registers: torch.Tensor (batch_size, seq_len, num_registers, model_dim) ou None, tokens supplémentaires placés
                   avant les patchs (sans embedding de position)
        attention_mask: torch.BoolTensor (batch_size, seq_len, num_registers + num_patches) ou None, True = ignoré ;
                        il doit déjà couvrir les registres (LAM.forward le construit ainsi)
        returns:
        x (batch_size, seq_len, tokens, model_dim), et si add_readout_token les readouts (batch_size, seq_len, model_dim)
        '''
        b, t, p, f = x.shape                                                            # (2, 5, 80, 512)

        if self.add_readout_token:
            # Encodeur : un token nul par frame, concaténé après les 80 patchs (axe 2 = axe des tokens)
            readout_token = torch.zeros((b, t, 1, f), device=x.device, dtype=x.dtype)  # (2, 5, 1, 512)
            x = torch.cat([x, readout_token], dim=2)                                    # (2, 5, 81, 512)
            # Le masque reçoit une colonne False pour le readout : il n'est jamais masqué
            attention_mask = torch.cat([attention_mask, torch.zeros((b, t, 1), dtype=torch.bool, device=x.device)], dim=-1) if attention_mask is not None else None   # (2, 5, 81)
        elif actions is not None:
            # Décodeur : la même action (vecteur de D) est ajoutée à tous les patchs de sa frame, pour conditionner la
            # prédiction de la frame suivante. Ici T vaut le nombre de transitions (4) et p le nombre de patchs (80).
            actions = actions.unsqueeze(2)                                              # (2, 4, 512) -> (2, 4, 1, 512)
            actions = actions.repeat(1, 1, p, 1)                                        # (2, 4, 80, 512)
            x = x + actions                                                             # (2, 4, 80, 512)

        # Relecture des dimensions : p inclut maintenant le readout (81) dans l'encodeur
        b, t, p, f = x.shape                                                            # (2, 5, 81, 512)

        # Indices de position : [0, 1, ..., p-1] pour les emplacements et [0, ..., t-1] pour les frames.
        # long : nn.Embedding attend des indices entiers ; device : sur le même GPU que x.
        spatial_pos = torch.arange(0, p, dtype=torch.long, device=x.device)            # [0, ..., 80], le 80 est le readout
        temporal_pos = torch.arange(0, t, dtype=torch.long, device=x.device)           # [0, 1, 2, 3, 4]

        # Lecture des lignes correspondantes dans les tables apprises
        spatial_pos_embd = self.spatial_pos_embedding(spatial_pos)                      # (81, 512)
        temporal_pos_embd = self.temporal_pos_embedding(temporal_pos)                   # (5, 512)

        # Position spatiale : on regroupe (clip, frame) pour que l'addition de (81, 512) se diffuse sur les 10 frames ;
        # le token k de chaque frame reçoit la ligne k
        x = rearrange(x, 'b t p f -> (b t) p f')                                        # (10, 81, 512)
        x = x + spatial_pos_embd                                                        # (10, 81, 512) + (81, 512)

        # Position temporelle : on regroupe (clip, patch) pour que l'addition de (5, 512) se diffuse sur les 162
        # séquences ; chaque token de la frame t reçoit la ligne t
        x = rearrange(x, '(b t) p f -> (b p) t f', b=b, t=t)                            # (162, 5, 512)
        x = x + temporal_pos_embd                                                       # (162, 5, 512) + (5, 512)

        # Retour à la forme (clip, frame, token, D)
        x = rearrange(x, '(b p) t f -> b t p f', b=b, p=p)                              # (2, 5, 81, 512)

        # Registres (optionnels, désactivés si num_register_tokens = 0) : tokens appris ajoutés AVANT les patchs, sans
        # position. Le readout reste donc le dernier token de chaque frame.
        if registers is not None:
            x = torch.cat([registers, x], dim=2)                                        # (2, 5, R + 81, 512)

        # Les n_layer couches : la forme ne change pas, le contenu des tokens s'enrichit. Le même masque sert à toutes.
        for layer in self.transformer.layers:
            x = layer(x, attention_mask=attention_mask)                                 # (2, 5, 81, 512)

        x = self.transformer.final_norm(x)                                              # LayerNorm de chaque token

        if self.add_readout_token:
            # Le readout est le dernier token de chaque frame : c'est l'action latente (non quantifiée) de la frame
            readout_output = x[:, :, -1, :]                                             # (2, 5, 512)
            return x, readout_output

        return x
    

class TransformerLayer(nn.Module):
    def __init__(self, model_dim, num_spatial_heads, dropout):
        super().__init__()
        self.spatial_attention = nn.MultiheadAttention(model_dim, num_spatial_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForward(model_dim, 4*model_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(model_dim)          # Genie uses QK normalization
        self.norm2 = nn.LayerNorm(model_dim)


    def forward(self, x, attention_mask=None):
        '''
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, patches, features)
        '''
        b, t, p, f = x.shape
        x = rearrange(x, 'b t p f -> (b t) p f')
        attention_mask = rearrange(attention_mask, 'b t p -> (b t) p') if attention_mask is not None else None
        x_norm = self.norm1(x)
        attn_output = self.spatial_attention(x_norm, x_norm, x_norm, need_weights=False, key_padding_mask=attention_mask)[0]
        x = x + attn_output

        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + ff_output

        x = rearrange(x, '(b t) p f -> b t p f', b=b, t=t)

        return x
    

class Transformer(nn.Module):
    def __init__(self, 
                 model_dim, 
                 n_layer, 
                 num_spatial_heads,
                 dropout,
                 image_size, 
                 patch_size):
        '''
        For the case with only 2 images on the encoder, the decoder will receive a single image and the action token.
        No need for temporal attention in this case. So we use a simple Transformer.
        '''
        super().__init__()
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.spatial_pos_embedding = nn.Embedding(num_patches, model_dim)

        self.transformer = nn.ModuleDict(dict(
            layers  = nn.ModuleList([TransformerLayer(model_dim, 
                                                      num_spatial_heads, 
                                                      dropout) for _ in range(n_layer)]),
            final_norm = nn.LayerNorm(model_dim),
        ))
        

    def forward(self, x, actions=None, registers=None, attention_mask=None):
        '''
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        '''
        b, t, p, f = x.shape

        if actions is not None:
            actions = actions.unsqueeze(2)
            actions = actions.repeat(1, 1, p, 1)
            x = x + actions

        b, t, p, f = x.shape
        
        spatial_pos = torch.arange(0, p, dtype=torch.long, device=x.device)
        spatial_pos_embd = self.spatial_pos_embedding(spatial_pos)

        x = rearrange(x, 'b t p f -> (b t) p f')
        x = x + spatial_pos_embd

        x = rearrange(x, '(b t) p f -> b t p f', b=b, t=t)

        if registers is not None:
            x = torch.cat([registers, x], dim=2)

        for layer in self.transformer.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.transformer.final_norm(x)

        return x