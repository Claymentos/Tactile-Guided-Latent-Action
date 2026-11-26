"""Prédicteur VLM de JALA (arXiv 2602.21736) : InternVL3-2B, dont l'état caché des tokens de mouvement à la couche 19
sur 28 donne les embeddings prédictifs h alignés sur les actions latentes de LAP.

Séquence par clip (un seul contexte visuel, la frame initiale, comme le papier) :
    [instruction] [256 tokens image de v_0] [K tokens d'état LSP(v_0, v_0)]
    <mot> [K tokens de mouvement du chunk 1] </mot>  <mot> [chunk 2] </mot>  ...
Attention causale (celle du LLM), sauf entre les K tokens d'un même chunk, qui se voient tous (prédiction masquée
bidirectionnelle à l'intérieur du chunk, causale entre chunks). Les tokens de mouvement sont les embeddings des codes
GRVQ (loss MCP, masqués en partie) ou K queries apprises (loss_action par régression).
Les tokens LSP entrent tels quels : les têtes de sortie de LAP / LSP projettent déjà dans l'espace d'embedding du VLM.
"""

import torch
import torch.nn.functional as F
from torch import nn


class VLMPredictor(nn.Module):
    def __init__(self, path: str, num_queries: int, align_layer: int = 19, freeze_vision: bool = True,
                 freeze_text_embeddings: bool = True, gradient_checkpointing: bool = True, image_size: int = 448,
                 attn_implementation: str = 'sdpa', prompt: str = 'Task: {instruction}\n'):
        """path : dossier local ou identifiant Hugging Face d'un InternVL3 au format transformers (« -hf »).
        freeze_vision : fige l'encodeur visuel et le projecteur (le papier fige l'encodeur visuel en post-training) ;
        freeze_text_embeddings : fige la table d'embedding du texte (150k x 1536, seulement lue pour l'instruction)."""
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        full = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.float32,
                                                           attn_implementation=attn_implementation)
        self.vlm = full.model                        # vision_tower, multi_modal_projector, language_model (sans lm_head)
        del full
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.prompt = prompt
        text_config = self.vlm.config.text_config
        self.dim, self.num_layers = text_config.hidden_size, text_config.num_hidden_layers
        assert 1 <= align_layer <= self.num_layers, f"align_layer={align_layer} hors de [1, {self.num_layers}]"
        self.num_queries, self.align_layer, self.image_size = num_queries, align_layer, image_size
        self.freeze_vision = freeze_vision
        if freeze_vision:
            self.vlm.vision_tower.requires_grad_(False)
            self.vlm.multi_modal_projector.requires_grad_(False)
        if freeze_text_embeddings:
            self.vlm.language_model.get_input_embeddings().requires_grad_(False)
        if gradient_checkpointing:
            self.vlm.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        scale = self.vlm.language_model.get_input_embeddings().weight.std().item()   # échelle des embeddings du LLM
        self.motion_queries = nn.Parameter(torch.randn(num_queries, self.dim) * scale)   # entrées en régression
        self.slot_embedding = nn.Parameter(torch.randn(num_queries, self.dim) * scale)   # position du token dans le chunk
        self.markers = nn.Parameter(torch.randn(2, self.dim) * scale)                    # <mot>, </mot>

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vision:
            self.vlm.vision_tower.eval()
        return self

    def image_tokens(self, image):
        """image (B, 3, H, W) normalisée ImageNet (la normalisation d'InternVL3) -> (B, 256, dim)."""
        image = F.interpolate(image.float(), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_vision):
            features = self.vlm.get_image_features(pixel_values=image, vision_feature_layer=self.vlm.config.vision_feature_layer,
                                                   vision_feature_select_strategy=self.vlm.config.vision_feature_select_strategy)
        if not torch.is_tensor(features):                       # transformers >= 5 : sortie structurée
            features = features.pooler_output
        return features

    def text_tokens(self, instructions, batch_size, device):
        if instructions is None:
            return torch.zeros(batch_size, 0, self.dim, device=device), torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        text = [self.prompt.format(instruction=i) for i in instructions]
        encoded = self.tokenizer(text, padding=True, add_special_tokens=False, return_tensors='pt').to(device)
        embeddings = self.vlm.language_model.get_input_embeddings()(encoded.input_ids)
        return embeddings, encoded.attention_mask.bool()

    def forward(self, states, image, instructions=None, motion_inputs=None, steps=None):
        """states (B, K, dim) : tokens LSP de v_0 ; image (B, 3, H, W) : v_0 ; instructions : liste de B textes ou None ;
        motion_inputs (B, S, K, dim) (embeddings des codes, MCP) ou None (queries apprises, `steps` chunks)
        -> h (couche align_layer), out (sortie normalisée de la dernière couche), (B, S, K, dim)."""
        B, K, D = states.shape
        device = states.device
        if motion_inputs is None:
            motion_inputs = self.motion_queries.expand(B, steps, K, D)
        S = motion_inputs.shape[1]
        motion = motion_inputs + self.slot_embedding
        opening, closing = (marker.expand(B, S, 1, D) for marker in self.markers)
        chunks = torch.cat([opening, motion, closing], dim=2)                                           # (B, S, K+2, D)

        text, text_valid = self.text_tokens(instructions, B, device)
        image = self.image_tokens(image)
        prefix = [text, image.to(text.dtype), states.to(text.dtype)]
        x = torch.cat(prefix + [chunks.flatten(1, 2).to(text.dtype)], dim=1)
        L, P = x.shape[1], sum(p.shape[1] for p in prefix)

        valid = torch.cat([text_valid, torch.ones(B, L - text.shape[1], dtype=torch.bool, device=device)], dim=1)
        segment = torch.zeros(L, dtype=torch.long, device=device)              # > 0 : tokens de mouvement du chunk i
        block = torch.zeros(K + 2, dtype=torch.long, device=device)
        block[1:K + 1] = 1
        segment[P:] = (block[None] * torch.arange(1, S + 1, device=device)[:, None]).flatten()
        index = torch.arange(L, device=device)
        allowed = (index[None, :] <= index[:, None]) | ((segment[:, None] == segment[None, :]) & (segment[:, None] > 0))
        allowed = allowed[None] & (valid[:, None, :] | torch.eye(L, dtype=torch.bool, device=device)[None])   # sans padding
        position_ids = (valid.long().cumsum(1) - 1).clamp(min=0)

        output = self.vlm.language_model(inputs_embeds=x, attention_mask=allowed[:, None], position_ids=position_ids,
                                         output_hidden_states=True, use_cache=False)
        select = lambda hidden: hidden[:, P:].unflatten(1, (S, K + 2))[:, :, 1:K + 1]
        return select(output.hidden_states[self.align_layer]), select(output.last_hidden_state)
