import torch
import lightning as L
import wandb
import h5py

from torch import nn
from sklearn.model_selection import train_test_split
from torchvision.utils import make_grid

from lam.data import AtariDataset, CoinRunDataset


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
        recon_loss, commit_loss, recon_video = self.model(batch)

        loss = recon_loss + commit_loss     # Add loss to help with VQ-VAE training
                                            # Aparently not necessary with FSQ

        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_commit_loss", commit_loss)

        random_reconstruction = self.reconstruct_with_random_actions(batch)
        delta_psnr = self.psnr(batch[:, 1:], recon_video) - self.psnr(batch[:, 1:], random_reconstruction)
        self.log("val_delta_psnr", delta_psnr)

        imagination = self.imagine_with_random_actions(batch, times=4)
        delta_psnr = self.psnr(batch[:, 1:5], recon_video[:, :4]) - self.psnr(batch[:, 1:5], imagination)
        self.log("val_delta_psnr_imagination", delta_psnr)

        if batch_idx == 0:
            # x = [make_grid([
            #     (batch[0][t]*255).to('cpu', torch.uint8), (recon_video[0][t]*255).to('cpu', torch.uint8), (imagination[0][t]*255).to('cpu', torch.uint8),
            #     (batch[1][t]*255).to('cpu', torch.uint8), (recon_video[1][t]*255).to('cpu', torch.uint8), (imagination[1][t]*255).to('cpu', torch.uint8),
            #     (batch[2][t]*255).to('cpu', torch.uint8), (recon_video[2][t]*255).to('cpu', torch.uint8), (imagination[2][t]*255).to('cpu', torch.uint8),
            #     (batch[3][t]*255).to('cpu', torch.uint8), (recon_video[3][t]*255).to('cpu', torch.uint8), (imagination[3][t]*255).to('cpu', torch.uint8)
            # ], nrow=3) for t in range(imagination.size(1))]

            # x = torch.stack(x, dim=0)
            # video = wandb.Video(x, fps=4, caption="video_reconstruction_imagination")

            x = make_grid([
                (batch[0][0]*255).to('cpu', torch.uint8), (batch[0][1]*255).to('cpu', torch.uint8), (recon_video[0][0]*255).to('cpu', torch.uint8), (random_reconstruction[0][0]*255).to('cpu', torch.uint8),
                (batch[1][0]*255).to('cpu', torch.uint8), (batch[1][1]*255).to('cpu', torch.uint8), (recon_video[1][0]*255).to('cpu', torch.uint8), (random_reconstruction[1][0]*255).to('cpu', torch.uint8),
                (batch[2][0]*255).to('cpu', torch.uint8), (batch[2][1]*255).to('cpu', torch.uint8), (recon_video[2][0]*255).to('cpu', torch.uint8), (random_reconstruction[2][0]*255).to('cpu', torch.uint8),
                (batch[3][0]*255).to('cpu', torch.uint8), (batch[3][1]*255).to('cpu', torch.uint8), (recon_video[3][0]*255).to('cpu', torch.uint8), (random_reconstruction[3][0]*255).to('cpu', torch.uint8)
            ], nrow=4)
            video = wandb.Image(x, caption="images_reconstruction_random aciton")
            self.logger.experiment.log({"val_images": video})

            x = [make_grid([
                (batch[0][t+1]*255).to('cpu', torch.uint8), (recon_video[0][t]*255).to('cpu', torch.uint8), (imagination[0][t]*255).to('cpu', torch.uint8),
                (batch[1][t+1]*255).to('cpu', torch.uint8), (recon_video[1][t]*255).to('cpu', torch.uint8), (imagination[1][t]*255).to('cpu', torch.uint8),
                (batch[2][t+1]*255).to('cpu', torch.uint8), (recon_video[2][t]*255).to('cpu', torch.uint8), (imagination[2][t]*255).to('cpu', torch.uint8),
                (batch[3][t+1]*255).to('cpu', torch.uint8), (recon_video[3][t]*255).to('cpu', torch.uint8), (imagination[3][t]*255).to('cpu', torch.uint8)
            ], nrow=3) for t in range(imagination.size(1))]

            x = torch.stack(x, dim=0)
            video = wandb.Video(x, fps=4, caption="video_reconstruction_imagination")
            self.logger.experiment.log({"val_videos": video})

        return loss
    

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
    

class AtariDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 validation_split: float = 0.2,
                 batch_size: int = 32,
                 num_workers: int = 6):
        super().__init__()
        self.data_path = data_path
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        train_indices, val_indices = self.get_split_indices()

        self.train_dataset = AtariDataset(self.data_path, train_indices)
        self.val_dataset = AtariDataset(self.data_path, val_indices)


    def get_split_indices(self):
        assert self.validation_split < 1.0 and self.validation_split >= 0.0, "Validation split must be between 0.0 and 1.0"

        with h5py.File(self.data_path, 'r') as f:
            dataset_idx = list(f.keys())

        train_indices, val_indices = train_test_split(dataset_idx, test_size=self.validation_split, shuffle=True)

        return train_indices, val_indices


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           prefetch_factor=2, 
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers, 
                                           shuffle=True)
    

class CoinRunModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 validation_path: str,
                 window_size: int = 16,
                 batch_size: int = 32,
                 num_workers: int = 6):
        super().__init__()
        self.data_path = data_path
        self.validation_path = validation_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.train_dataset = CoinRunDataset(self.data_path, self.window_size)
        self.val_dataset = CoinRunDataset(self.validation_path, self.window_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           prefetch_factor=2, 
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers, 
                                           shuffle=True)