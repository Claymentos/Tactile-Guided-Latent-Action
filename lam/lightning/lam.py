import torch
import lightning as L
import wandb

from torch import nn
from torchvision.utils import make_grid

from lam.metrics import _compute_metrics

class LAMTraining(L.LightningModule):
    def __init__(self,  
                 model: nn.Module, 
                 learning_rate: float = 1e-3, 
                 weight_decay: float = 0.01,
                 scheduler_max_steps: int = 200_000,
                 codebook_replace_freq: float = 0.05,
                 metrics: dict = {},
                 compute_metrics_every_n_steps: int = 100,
                 commit_loss_weight: float = 1.0,
                 perceptual_loss_weight: float = 1.0,
                ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = scheduler_max_steps
        self.codebook_replace_freq = codebook_replace_freq      # Only used in NSVQ
        self.metrics = metrics
        self.compute_metrics_every_n_steps = compute_metrics_every_n_steps
        self.commit_loss_weight = commit_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight


    def training_step(self, batch, batch_idx):
        '''
        arguments:
        batch: torch.Tensor, shape: (batch_size, seq_len, channels, height, width)
                or Tuple[torch.Tensor, torch.Tensor], shape of each: (batch_size, seq_len, channels, height, width)
        '''
        batch = self.preprocess(batch)

        if "images_secondary" in batch.keys():
            source_view, target_view = batch["images"], batch["images_secondary"]           # Multiview case, source view is different from target view
            recon_loss, commit_loss, recon_video, indices = self.model(source_view, target_view)
        else:
            target_view = batch["images"]                                                   # Traditional case, source view is equal to target view
            output = self.model(target_view)

        loss = output["recon_loss"] + self.commit_loss_weight * output["commit_loss"] + self.perceptual_loss_weight * output["perceptual_loss"]

        self.log("train/loss", loss)
        self.log("train/recon_loss", output["recon_loss"])
        self.log("train/perceptual_loss", output["perceptual_loss"])
        self.log("train/commit_loss", output["commit_loss"])

        if batch_idx % self.compute_metrics_every_n_steps == 0:
            with torch.no_grad():
                self.compute_metrics(target_view, output["recon_video"], output["features"], output["recon_features"], output["indices"], output["action_tokens"], prefix="train")

        self.replace_unused_codebook_entries()      # Only used in NSVQ

        return loss

    
    def validation_step(self, batch, batch_idx): 
        '''
        arguments:
        batch: torch.Tensor, shape: (batch_size, seq_len, channels, height, width)
        '''
        batch = self.preprocess(batch)
        
        if "images_secondary" in batch.keys():
            source_view, target_view = batch["images"], batch["images_secondary"]           # Multiview case, source view is different from target view
            recon_loss, commit_loss, recon_video, indices = self.model(source_view, target_view)
        else:
            target_view = batch["images"]                                                   # Traditional case, source view is equal to target view
            output = self.model(target_view)

        loss = output["recon_loss"] + output["commit_loss"] + output["perceptual_loss"]

        self.log("val/loss", loss)
        self.log("val/recon_loss", output["recon_loss"])
        self.log("val/perceptual_loss", output["perceptual_loss"])
        self.log("val/commit_loss", output["commit_loss"])

        random_reconstruction = self.compute_metrics(target_view, output["recon_video"], output["features"], output["recon_features"], output["indices"], output["action_tokens"], prefix="val", return_random_reconstruction=True)

        if batch_idx == 0:
            self.log_videos(target_view, output["recon_video"], random_reconstruction)

        return loss
    
    def preprocess(self, batch):
        if 'obs' in batch.keys():
            batch["images"] = torch.flip(batch['obs']['agentview_rgb'], dims=(-2,)) / 255.0           # Images on Libero are upside down

        return batch
    
    def log_videos(self, target_view, recon_video, random_reconstruction):
        recon_video = torch.cat([target_view[:, :1], recon_video], dim=1)
        
        if random_reconstruction is not None:
            random_reconstruction = torch.cat([target_view[:, :1], random_reconstruction], dim=1)

            x = [make_grid([
                (target_view[0][t]*255).to('cpu', torch.uint8), (recon_video[0][t]*255).to('cpu', torch.uint8), (random_reconstruction[0][t]*255).to('cpu', torch.uint8),
                (target_view[1][t]*255).to('cpu', torch.uint8), (recon_video[1][t]*255).to('cpu', torch.uint8), (random_reconstruction[1][t]*255).to('cpu', torch.uint8),
                (target_view[2][t]*255).to('cpu', torch.uint8), (recon_video[2][t]*255).to('cpu', torch.uint8), (random_reconstruction[2][t]*255).to('cpu', torch.uint8),
                (target_view[3][t]*255).to('cpu', torch.uint8), (recon_video[3][t]*255).to('cpu', torch.uint8), (random_reconstruction[3][t]*255).to('cpu', torch.uint8)
            ], nrow=3) for t in range(random_reconstruction.size(1))]

            caption="video | reconstruction | imagination"

        else:
            x = [make_grid([
                (target_view[0][t]*255).to('cpu', torch.uint8), (recon_video[0][t]*255).to('cpu', torch.uint8),
                (target_view[1][t]*255).to('cpu', torch.uint8), (recon_video[1][t]*255).to('cpu', torch.uint8),
                (target_view[2][t]*255).to('cpu', torch.uint8), (recon_video[2][t]*255).to('cpu', torch.uint8),
                (target_view[3][t]*255).to('cpu', torch.uint8), (recon_video[3][t]*255).to('cpu', torch.uint8)
            ], nrow=2) for t in range(recon_video.size(1))]

            caption="video | reconstruction"

        x = torch.stack(x, dim=0)
        video = wandb.Video(x, fps=2, caption=caption)
        self.logger.experiment.log({"val_videos": video})

    def reconstruct_with_random_actions(self, batch, features, action_tokens, return_video=True):
        if self.model.vq.get_codebook() is None:
            idx = torch.randperm(action_tokens.shape[0])
            random_action_tokens = action_tokens[idx]
        else:
            random_indices = torch.randint(0, self.model.vq.codebook_size, (batch.size(0), batch.size(1)-1), device=batch.device)
            random_action_tokens = self.model.vq.dequantize(random_indices)

        recon_features = self.model.decode(features[:, :-1], random_action_tokens)

        if not return_video:
            return recon_features
        
        recon_video = self.model.feature_extractor._post_process(recon_features)  

        return recon_features, recon_video

    def imagine_with_random_actions(self, batch, features, times=4):
        x = batch[:, 0].unsqueeze(1)
        imagination = x.clone()

        for t in range(times):
            random_indices = torch.randint(0, self.model.vq.codebook_size, (batch.size(0), 1), device=batch.device)
            random_action_tokens = self.model.vq.dequantize(random_indices) 
            recon_video = self.model.decode(features, random_action_tokens)
            imagination = torch.cat([imagination, recon_video], dim=1)
            x = recon_video

        return imagination[:, 1:]

    def reconstruct_with_same_action(self, features, indices, action_tokens):
        if indices:
            half_indices = indices[:indices.shape[0]//2]
            duplicate_indices = half_indices.repeat((2, 1))
            action_tokens = self.model.vq.dequantize(duplicate_indices)
        else:
            half_action_tokens = action_tokens[:action_tokens.shape[0]//2]
            duplicate_action_tokens = half_action_tokens.repeat((2, 1, 1))
            action_tokens = duplicate_action_tokens
            
        recon_features = self.model.decode(features[:, :-1], action_tokens)
        recon_video = self.model.feature_extractor._post_process(recon_features)

        return recon_video

    def compute_metrics(self, batch, recon_video, features, recon_features, indices, action_tokens, prefix, return_random_reconstruction=False):
        if recon_video is not None:
            feature_imagination, video_imagination = self.reconstruct_with_random_actions(batch, features, action_tokens, return_video=True)
            same_action_recon = self.reconstruct_with_same_action(features, indices, action_tokens)
        else:
            feature_imagination = self.reconstruct_with_random_actions(batch, features, action_tokens, return_video=False)
            same_action_recon = None

        metrics = _compute_metrics(self.metrics, batch, recon_video, features, recon_features, indices, feature_imagination, video_imagination, self.model.vq.codebook_size, self.model.vq.get_codebook(), same_action_recon)

        for key, value in metrics.items():
            self.log(f"{prefix}/{key}", value)

        if return_random_reconstruction and recon_video is not None:
            return video_imagination
    
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