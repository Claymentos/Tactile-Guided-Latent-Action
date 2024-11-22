from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange


class LAM(nn.Module):
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
        readout = readout[:, 1:, :]                                                 # Ignore the first action token, it doesn't represent any transition
        readout = rearrange(readout, 'b t f -> (b t) f')

        action_tokens, indices, commit_loss = self.vq(readout)
        action_tokens = rearrange(action_tokens, '(b t) f -> b t f', b=x.size(0), t=x.size(1)-1)
        indices = rearrange(indices, '(b t) -> b t', b=x.size(0), t=x.size(1)-1)

        recon_video = self.decode(x, action_tokens)

        recon_loss = nn.functional.mse_loss(recon_video, x[:, 1:, :, :, :])        # The first frame of x is not used as a target
        
        return recon_loss, commit_loss, recon_video

        
    def encode(self, x):
        return self.encoder(x)
    

    def decode(self, x, actions):
        if x.size(1) >  1:
            x = x[:, :-1, :, :, :]         # Ignore the last frame
            
        logits = self.decoder(x, actions)

        reconstruction = self.to_image(logits)
        reconstruction = nn.functional.sigmoid(reconstruction)
     
        return reconstruction
