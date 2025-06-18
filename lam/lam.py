from torch import nn
from einops.layers.torch import Rearrange


class LAM(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 vq,
                 image_encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
        self.image_encoder = image_encoder


    def forward(self, x):
        '''
        arguments:
        x: torch.Tensor, shape (batch_size, seq_len, channels, height, width)
        '''
        features = self.image_encoder(x)                                            # Extract image features or simply patchfy and project if working in image space
        
        logits, readout = self.encode(features)
        readout = readout[:, 1:, :]                                                 # Ignore the first action token, it doesn't represent any transition
        action_tokens, indices, commit_loss = self.vq.quantize(readout)

        logits = self.decode(features[:, :-1], action_tokens)                                             # Ignore the last frame, since there is no target for it
        recon_loss, recon_video = self.image_encoder.compute_loss(logits, x[:, 1:], features[:, 1:])      # The first frame of x is not used as a target
        
        return recon_loss, commit_loss, indices, features, logits, recon_video

        
    def encode(self, x):
        return self.encoder(x)
    

    def decode(self, x, actions):
        logits = self.decoder(x, actions)
     
        return logits


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