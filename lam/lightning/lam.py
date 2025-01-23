import torch
import lightning as L
import wandb

from torch import nn
from einops import rearrange
from torchvision.utils import make_grid
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


class LAMTraining(L.LightningModule):
    def __init__(self,  
                 model: nn.Module, 
                 learning_rate: float = 1e-3, 
                 weight_decay: float = 0.01,
                 scheduler_max_steps: int = 200_000,
                 codebook_replace_freq: float = 0.05
                ):
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = scheduler_max_steps
        self.codebook_replace_freq = codebook_replace_freq      # Only used in NSVQ


    def training_step(self, batch, batch_idx):
        '''
        arguments:
        batch: torch.Tensor, shape: (batch_size, seq_len, channels, height, width)
        '''
        batch = self.preprocess(batch)
        recon_loss, commit_loss, recon_video, indices = self.model(batch)

        loss = recon_loss + commit_loss     # Add VQ-VAE loss
                                            # Check if it is necessary with FSQ

        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_commit_loss", commit_loss)

        with torch.no_grad():
            self.compute_metrics(batch, recon_video, indices, prefix="train")

        self.replace_unused_codebook_entries()      # Only used in NSVQ

        return loss

    
    def validation_step(self, batch, batch_idx): 
        '''
        arguments:
        batch: torch.Tensor, shape: (batch_size, seq_len, channels, height, width)
        '''
        batch = self.preprocess(batch)
        recon_loss, commit_loss, recon_video, indices = self.model(batch)

        loss = recon_loss + commit_loss

        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_commit_loss", commit_loss)

        random_reconstruction = self.compute_metrics(batch, recon_video, indices, prefix="val", return_random_reconstruction=True)

        imagination = self.imagine_with_random_actions(batch, times=min(4, recon_video.size(1)))
        delta_psnr = self.psnr(batch[:, 1:5], recon_video[:, :4]) - self.psnr(batch[:, 1:5], imagination)
        self.log("val_delta_psnr_imagination_t=4", delta_psnr)

        if batch_idx == 0:
            x = [make_grid([
                (batch[0][t+1]*255).to('cpu', torch.uint8), (recon_video[0][t]*255).to('cpu', torch.uint8), (imagination[0][t]*255).to('cpu', torch.uint8), (random_reconstruction[0][t]*255).to('cpu', torch.uint8),
                (batch[1][t+1]*255).to('cpu', torch.uint8), (recon_video[1][t]*255).to('cpu', torch.uint8), (imagination[1][t]*255).to('cpu', torch.uint8), (random_reconstruction[1][t]*255).to('cpu', torch.uint8),
                (batch[2][t+1]*255).to('cpu', torch.uint8), (recon_video[2][t]*255).to('cpu', torch.uint8), (imagination[2][t]*255).to('cpu', torch.uint8), (random_reconstruction[2][t]*255).to('cpu', torch.uint8),
                (batch[3][t+1]*255).to('cpu', torch.uint8), (recon_video[3][t]*255).to('cpu', torch.uint8), (imagination[3][t]*255).to('cpu', torch.uint8), (random_reconstruction[3][t]*255).to('cpu', torch.uint8)
            ], nrow=4) for t in range(imagination.size(1))]

            x = torch.stack(x, dim=0)
            video = wandb.Video(x, fps=4, caption="video_reconstruction_imagination with random actions_reconstruction with random actions")
            self.logger.experiment.log({"val_videos": video})

        return loss
    
    def preprocess(self, batch):
        if isinstance(batch, dict):
            batch = torch.flip(batch['obs']['agentview_rgb'], dims=(-2,)) / 255.0           # Images on Libero are upside down

        return batch

    def reconstruct_with_random_actions(self, batch):
        random_indices = torch.randint(0, self.model.vq.codebook_size, (batch.size(0), batch.size(1)-1), device=batch.device)
        random_action_tokens = self.model.vq.dequantize(random_indices) 
        recon_video = self.model.decode(batch[:, :-1, :, :, :], random_action_tokens)

        return recon_video
    

    def imagine_with_random_actions(self, batch, times=4):
        x = batch[:, 0].unsqueeze(1)
        imagination = x.clone()

        for t in range(times):
            random_indices = torch.randint(0, self.model.vq.codebook_size, (batch.size(0), 1), device=batch.device)
            random_action_tokens = self.model.vq.dequantize(random_indices) 
            recon_video = self.model.decode(x, random_action_tokens)
            imagination = torch.cat([imagination, recon_video], dim=1)
            x = recon_video

        return imagination[:, 1:]

    def compute_metrics(self, batch, recon_video, indices, prefix, return_random_reconstruction=False):
        imagination = self.reconstruct_with_random_actions(batch)
        delta_psnr = self.psnr(batch[:, 1:], recon_video) - self.psnr(batch[:, 1:], imagination)
        self.log(f"{prefix}_delta_psnr", delta_psnr)
        
        ssim_score = ssim(rearrange(recon_video, 'b t c h w -> (b t) c h w'), rearrange(batch[:, 1:], 'b t c h w -> (b t) c h w'))
        self.log(f"{prefix}_ssim", ssim_score)

        codebook_usage, codebook_entropy, codebook_cv = self.codebook_usage(indices)
        self.log(f"{prefix}_codebook_usage", codebook_usage)
        self.log(f"{prefix}_codebook_entropy", codebook_entropy)
        self.log(f"{prefix}_codebook_cv", codebook_cv)

        distinctiveness_cosine = self.distinctiveness_cosine(self.model.vq.get_codebook())
        self.log(f"{prefix}_distinctiveness_cosine", distinctiveness_cosine)

        if return_random_reconstruction:
            return imagination

    def psnr(self, image, reconstruction):
        return 10 * torch.log10((1.0**2) / torch.nn.functional.mse_loss(reconstruction, image)).item()

    def codebook_usage(self, indices):
        values, counts = torch.unique(indices, return_counts=True)
        total_counts = torch.full((self.model.vq.codebook_size,), 0, device=indices.device)
        total_counts[values] = counts

        codebook_usage =  torch.mean((total_counts != 0).float())
        codebook_entropy = self.entropy(total_counts)
        codebook_cv = self.coefficient_of_variation(total_counts)

        return codebook_usage, codebook_entropy, codebook_cv
    
    def entropy(self, histogram):
        histogram = histogram + 1e-10                   # Avoid log(0)
        probabilities = histogram / histogram.sum()
        return -torch.sum(probabilities * torch.log(probabilities))
    
    def  coefficient_of_variation(self, histogram):
        return histogram.float().std() / histogram.float().mean()
    
    def distinctiveness_cosine(self, codes):
        '''Measure how distinct the codes are using cosine similarity'''
        normalized_vectors = codes / torch.norm(codes, dim=1, keepdim=True)

        similarity_matrix = normalized_vectors @ normalized_vectors.T

        mask = ~torch.eye(codes.shape[0], dtype=bool)
        average_similarity = similarity_matrix[mask].mean()

        return 1.0 - average_similarity

    def replace_unused_codebook_entries(self):
        if self.global_step == 0:
            return
        
        replace_interval = int(self.steps * self.codebook_replace_freq)

        if ((self.global_step % replace_interval) == 0) and (self.global_step <= self.steps - 2 * replace_interval):
            self.model.vq.replace_codebook(replace_interval)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]