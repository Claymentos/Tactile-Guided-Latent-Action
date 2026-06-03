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
    def __init__(self, model_dim, num_spatial_heads, num_temporal_heads, dropout, is_causal):
        super().__init__()
        self.is_causal = is_causal
        self.temporal_attention = nn.MultiheadAttention(model_dim, num_temporal_heads, dropout=dropout, batch_first=True)
        self.spatial_attention = nn.MultiheadAttention(model_dim, num_spatial_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForward(model_dim, 4*model_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(model_dim)          # Genie uses QK normalization
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)


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

        x = rearrange(x, '(b t) p f -> (b p) t f', b=b, t=t)
        x_norm = self.norm2(x)
        if self.is_causal:
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
            attn_output = self.temporal_attention(x_norm, x_norm, x_norm, attn_mask=causal_mask, is_causal=True, need_weights=False)[0]
        else:
            attn_output = self.temporal_attention(x_norm, x_norm, x_norm, need_weights=False)[0]        

        x = x + attn_output
        x_norm = self.norm3(x)
        ff_output = self.feed_forward(x_norm)
        x = x + ff_output

        x = rearrange(x, '(b p) t f -> b t p f', b=b, p=p)

        return x


class STTransformer(nn.Module):
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
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.add_readout_token = add_readout_token

        if add_readout_token:
            num_patches += 1

        self.spatial_pos_embedding = nn.Embedding(num_patches, model_dim)
        self.temporal_pos_embedding = nn.Embedding(max_seq_len, model_dim)

        self.transformer = nn.ModuleDict(dict(
            layers  = nn.ModuleList([SpatialTemporalLayer(model_dim, 
                                                          num_spatial_heads, 
                                                          num_temporal_heads, 
                                                          dropout,
                                                          is_causal=is_causal) for _ in range(n_layer)]),
            final_norm = nn.LayerNorm(model_dim),
        ))
        

    def forward(self, x, actions=None, registers=None, attention_mask=None):
        '''
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, num_patches, model_dim)
        '''
        b, t, p, f = x.shape

        if self.add_readout_token:
            readout_token = torch.zeros((b, t, 1, f), device=x.device, dtype=x.dtype)
            x = torch.cat([x, readout_token], dim=2)
            attention_mask = torch.cat([attention_mask, torch.zeros((b, t, 1), dtype=torch.bool, device=x.device)], dim=-1) if attention_mask is not None else None
        elif actions is not None:
            actions = actions.unsqueeze(2)
            actions = actions.repeat(1, 1, p, 1)
            x = x + actions

        b, t, p, f = x.shape
        
        spatial_pos = torch.arange(0, p, dtype=torch.long, device=x.device)
        temporal_pos = torch.arange(0, t, dtype=torch.long, device=x.device)

        spatial_pos_embd = self.spatial_pos_embedding(spatial_pos)
        temporal_pos_embd = self.temporal_pos_embedding(temporal_pos)

        x = rearrange(x, 'b t p f -> (b t) p f')
        x = x + spatial_pos_embd

        x = rearrange(x, '(b t) p f -> (b p) t f', b=b, t=t)
        x = x + temporal_pos_embd

        x = rearrange(x, '(b p) t f -> b t p f', b=b, p=p)

        if registers is not None:
            x = torch.cat([registers, x], dim=2)

        for layer in self.transformer.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.transformer.final_norm(x)

        if self.add_readout_token:
            readout_output = x[:, :, -1, :]
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
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
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