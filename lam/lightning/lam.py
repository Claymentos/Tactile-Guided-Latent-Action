import torch
import lightning as L
import wandb

from torch import nn
from torchvision.utils import make_grid


class LAMTraining(L.LightningModule):
    def __init__(self,  
                 model: nn.Module, 
                 learning_rate: float = 1e-3, 
                 weight_decay: float = 0.01,
                 scheduler_max_steps: int = 200_000,
                ):
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = scheduler_max_steps


    def training_step(self, batch, batch_idx):
        '''
        arguments:
        batch: torch.Tensor, shape: (batch_size, seq_len, channels, height, width)
        '''
        batch = self.preprocess(batch)
        recon_loss, commit_loss, recon_video = self.model(batch)

        loss = recon_loss + commit_loss     # Add loss to help with VQ-VAE training
                                            # Aparently not necessary with FSQ

        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_commit_loss", commit_loss)

        with torch.no_grad():
            imagination = self.reconstruct_with_random_actions(batch)
            delta_psnr = self.psnr(batch[:, 1:], recon_video) - self.psnr(batch[:, 1:], imagination)
            self.log("train_delta_psnr", delta_psnr)
       

        return loss

    
    def validation_step(self, batch, batch_idx): 
        '''
        arguments:
        batch: torch.Tensor, shape: (batch_size, seq_len, channels, height, width)
        '''
        batch = self.preprocess(batch)
        recon_loss, commit_loss, recon_video = self.model(batch)

        loss = recon_loss + commit_loss

        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_commit_loss", commit_loss)

        random_reconstruction = self.reconstruct_with_random_actions(batch)
        delta_psnr = self.psnr(batch[:, 1:], recon_video) - self.psnr(batch[:, 1:], random_reconstruction)
        self.log("val_delta_psnr_t=2", delta_psnr)

        imagination = self.imagine_with_random_actions(batch, times=4)
        delta_psnr = self.psnr(batch[:, 1:5], recon_video[:, :4]) - self.psnr(batch[:, 1:5], imagination)
        self.log("val_delta_psnr_imagination_t=4", delta_psnr)

        if batch_idx == 0:
            x = make_grid([
                (batch[0][0]*255).to('cpu', torch.uint8), (batch[0][1]*255).to('cpu', torch.uint8), (recon_video[0][0]*255).to('cpu', torch.uint8), (random_reconstruction[0][0]*255).to('cpu', torch.uint8),
                (batch[1][0]*255).to('cpu', torch.uint8), (batch[1][1]*255).to('cpu', torch.uint8), (recon_video[1][0]*255).to('cpu', torch.uint8), (random_reconstruction[1][0]*255).to('cpu', torch.uint8),
                (batch[2][0]*255).to('cpu', torch.uint8), (batch[2][1]*255).to('cpu', torch.uint8), (recon_video[2][0]*255).to('cpu', torch.uint8), (random_reconstruction[2][0]*255).to('cpu', torch.uint8),
                (batch[3][0]*255).to('cpu', torch.uint8), (batch[3][1]*255).to('cpu', torch.uint8), (recon_video[3][0]*255).to('cpu', torch.uint8), (random_reconstruction[3][0]*255).to('cpu', torch.uint8)
            ], nrow=4)
            video = wandb.Image(x, caption="frame1_frame2_reconstruction_random aciton")
            self.logger.experiment.log({"val_images": video})

            x = [make_grid([
                (batch[0][t+1]*255).to('cpu', torch.uint8), (recon_video[0][t]*255).to('cpu', torch.uint8), (imagination[0][t]*255).to('cpu', torch.uint8),
                (batch[1][t+1]*255).to('cpu', torch.uint8), (recon_video[1][t]*255).to('cpu', torch.uint8), (imagination[1][t]*255).to('cpu', torch.uint8),
                (batch[2][t+1]*255).to('cpu', torch.uint8), (recon_video[2][t]*255).to('cpu', torch.uint8), (imagination[2][t]*255).to('cpu', torch.uint8),
                (batch[3][t+1]*255).to('cpu', torch.uint8), (recon_video[3][t]*255).to('cpu', torch.uint8), (imagination[3][t]*255).to('cpu', torch.uint8)
            ], nrow=3) for t in range(imagination.size(1))]

            x = torch.stack(x, dim=0)
            video = wandb.Video(x, fps=4, caption="video_reconstruction_imagination with random actions")
            self.logger.experiment.log({"val_videos": video})

        return loss
    
    def preprocess(self, batch):
        if isinstance(batch, dict):
            batch = torch.flip(batch['obs']['agentview_rgb'], dims=(-2,)) / 255.0           # Images on Libero are upside down

        return batch

    def reconstruct_with_random_actions(self, batch):
        random_indices = torch.randint(0, self.model.vq.codebook_size, (batch.size(0), batch.size(1)-1), device=batch.device)
        random_action_tokens = self.model.vq.project_out(self.model.vq.codebook[random_indices]) 
        recon_video = self.model.decode(batch, random_action_tokens)

        return recon_video
    

    def imagine_with_random_actions(self, batch, times=4):
        x = batch[:, 0].unsqueeze(1)
        imagination = x.clone()

        for t in range(times):
            random_indices = torch.randint(0, self.model.vq.codebook_size, (batch.size(0), 1), device=batch.device)
            random_action_tokens = self.model.vq.project_out(self.model.vq.codebook[random_indices]) 
            recon_video = self.model.decode(x, random_action_tokens)
            imagination = torch.cat([imagination, recon_video], dim=1)
            x = recon_video

        return imagination[:, 1:]

    def psnr(self, image, reconstruction):
        return 10 * torch.log10((1.0**2) / torch.nn.functional.mse_loss(reconstruction, image)).item()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]