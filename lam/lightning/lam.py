import torch
import lightning as L
import wandb
from typing import Callable, Dict, Iterable, Tuple
from os import listdir, makedirs, path

from torch import nn
from torchvision.utils import make_grid
from torch.optim import AdamW, Optimizer
from torch import Tensor

from lam.metrics import _compute_metrics
import matplotlib
import numpy as np
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
OptimizerCallable = Callable[[Iterable], Optimizer]
import matplotlib.pyplot as plt
from accelerate import PartialState


class ResidualActVAETraining(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 weight_decay: float=0.01,
                 scheduler_max_steps: int=200_000):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.weight_decay = weight_decay
        self.steps = scheduler_max_steps
        
    def process_batch(self, batch):
        if isinstance(batch, list):
            actions = batch[0]  
        if isinstance(batch, dict):
            actions = batch["actions"]
        return actions
    
    def training_step(self, batch, batch_idx):

        batch = self.process_batch(batch)
        output = self.model(batch)
        loss = output["loss"]

        self.log("train/loss", output["loss"])
        self.log("train/recon_loss", output["recon_loss"])
        self.log("train/vq_loss", output["vq_loss"])

        return loss
    
    def validation_step(self, batch, batch_idx):

        batch = self.process_batch(batch)
        output = self.model(batch)

        loss = output["recon_loss"] + self.vq_beta * output["commit_loss"] + output["codebook_loss"]
        self.log("train/loss", loss)
        self.log("train/recon_loss", output["recon_loss"])
        self.log("train/commit_loss", output["commit_loss"])
        self.log("train/codebook_loss", output["codebook_loss"])

        return loss
    

    def configure_optimizers(self):
            vq_params = list(self.model.vq.parameters())
            base_params = [p for n, p in self.model.named_parameters() if 'vq' not in n]

            optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': 2e-5},
                {'params': vq_params, 'lr': 1e-3}
            ], weight_decay=self.weight_decay)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wandb
import lightning as L

class ActVAETraining(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.01,
                 scheduler_max_steps: int = 200_000,
                 vis_every_n_epochs: int = 5, # Fréquence des visualisations lourdes
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = scheduler_max_steps
        self.vis_every_n_epochs = vis_every_n_epochs

        # Listes pour accumuler les données sur toute l'époque de validation
        self.val_indices_epoch = []
        self.val_actions_epoch = []

    def on_validation_epoch_start(self):
        self.val_indices_epoch.clear()
        self.val_actions_epoch.clear()

    def process_batch(self, batch):
        if isinstance(batch, list):
            return batch[0]
        elif isinstance(batch, dict):
            return batch["actions"]
        return batch

    def training_step(self, batch, batch_idx):
        batch = self.process_batch(batch)
        output = self.model(batch)
        self.log("train/loss", output["loss"])
        self.log("train/recon_loss", output["recon_loss"])
        self.log("train/vq_loss", output["vq_loss"])
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        batch = self.process_batch(batch)
        output = self.model(batch)

        # Extraction sécurisée vers le CPU pour éviter les fuites de mémoire GPU
        # indices shape attendu: (B, T) -> on l'aplatit en (B*T)
        self.val_indices_epoch.append(output["indices"].detach().cpu().view(-1))
        # batch shape attendu: (B, T, 88, 3) -> on extrait les poses individuelles (B*T, 88, 3)
        self.val_actions_epoch.append(batch.detach().cpu().view(-1, 88, 3))

        # Logs standards
        self.log("val/loss", output["loss"])
        self.log("val/recon_loss", output["recon_loss"])
        self.log("val/vq_loss", output["vq_loss"])


        return output["loss"]

    def on_validation_epoch_end(self):
        if not self.val_indices_epoch:
            return

        # Concaténation globale des résultats de validation
        all_indices = torch.cat(self.val_indices_epoch)
        all_gt_poses = torch.cat(self.val_actions_epoch)
        
        codebook_size = self.model.vq.codebook_size
        counts = torch.bincount(all_indices, minlength=codebook_size).float()
        
        # --- 1. Calcul et log des métriques de base ---
        num_active_codes = torch.unique(all_indices).numel()
        usage_ratio = (num_active_codes / codebook_size) * 100.0
        probs = counts / counts.sum()
        entropy = -torch.sum(probs[probs > 0] * torch.log(probs[probs > 0]))
        perplexity = torch.exp(entropy)

        self.log("val/codebook_active_codes", float(num_active_codes))
        self.log("val/codebook_usage_ratio", usage_ratio)
        self.log("val/codebook_perplexity", perplexity)

        # --- 2. Calcul de la variance Intra-Code ---
        intra_code_variances = []
        for code_idx in range(codebook_size):
            mask = (all_indices == code_idx)
            if mask.sum() > 5: # Validité statistique minimale
                variance = torch.var(all_gt_poses[mask], dim=0).mean()
                intra_code_variances.append(variance)
        
        if intra_code_variances:
            mean_intra_variance = torch.stack(intra_code_variances).mean()
            self.log("val/codebook_intra_cluster_variance", mean_intra_variance)

        # --- 3. Visualisations avancées (Toutes les N époques) ---
        if (self.current_epoch + 1) % self.vis_every_n_epochs == 0 and self.logger and hasattr(self.logger, "experiment"):
            
            # A. Log de l'histogramme de distribution
            self.logger.experiment.log({"val/codebook_distribution": wandb.Histogram(all_indices.numpy())})
            
            # B. Génération du t-SNE du Codebook
            fig_tsne = self._generate_tsne_plot()
            self.logger.experiment.log({"val/codebook_tsne": wandb.Image(fig_tsne)})
            plt.close(fig_tsne)
            
            # C. Décodage et affichage des "Poses Atomes" les plus utilisées
            # On récupère les indices des 3 codes les plus activés durant cette validation
            top_codes = torch.topk(counts, k=3).indices.tolist()
            fig_poses = self._generate_poses_plot(top_codes)
            self.logger.experiment.log({"val/codebook_top_decoded_poses": wandb.Image(fig_poses)})
            plt.close(fig_poses)

        # Nettoyage obligatoire de la mémoire
        self.val_indices_epoch.clear()
        self.val_actions_epoch.clear()


    def _generate_tsne_plot(self):
        codebook_weights = self.model.vq._codebook.embed.detach().cpu().numpy().squeeze()
        codebook_size = self.model.vq.codebook_size
        
        tsne = TSNE(n_components=2, perplexity=min(30, codebook_size - 1), random_state=42)
        codebook_2d = tsne.fit_transform(codebook_weights)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(codebook_2d[:, 0], codebook_2d[:, 1], alpha=0.6, c='royalblue', edgecolors='k')
        ax.set_title(f"Topologie du Codebook (Époque {self.current_epoch})")
        ax.grid(True, linestyle='--', alpha=0.5)
        return fig

    def _generate_poses_plot(self, code_indices: list):
        fig = plt.figure(figsize=(15, 5))
        
        for idx, code_idx in enumerate(code_indices):
            # Extraction du vecteur latent pur du dictionnaire
            with torch.no_grad():
                embedding = self.model.vq._codebook.embed.squeeze()[code_idx] # Shape: (512,)
                z_q = embedding.unsqueeze(0).unsqueeze(0).to(self.device) # Shape: (1, 1, 512)
                
                # Passage dans ton décodeur personnalisé
                decoded_pose = self.model.decode(z_q) # Shape: (1, 1, 88, 3)
                pose_np = decoded_pose.squeeze().cpu().numpy() # Shape: (88, 3)
            
            # Subplot en 3D
            ax = fig.add_subplot(1, len(code_indices), idx + 1, projection='3d')
            
            # Affichage des 88 points clés sous forme de nuage de points
            ax.scatter(pose_np[:, 0], pose_np[:, 1], pose_np[:, 2], c='crimson', marker='o', s=15)
            
            ax.set_title(f"Pose du Code n°{code_idx}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Pour garder une proportion visuelle correcte
            ax.set_box_aspect([1,1,1]) 
            
        plt.suptitle(f"Reconstruction Sémantique des Top Codes (Époque {self.current_epoch})")
        plt.tight_layout()
        return fig


        
    def configure_optimizers(self):
            vq_params = list(self.model.vq.parameters())
            base_params = [p for n, p in self.model.named_parameters() if 'vq' not in n]

            optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': 2e-5},
                {'params': vq_params, 'lr': 1e-3}
            ], weight_decay=self.weight_decay)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    
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
        if indices is not None:
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
    
class VDVAETraining(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 vae_learning_rate: float=2e-4,
                 codebook_learning_rate: float=1e-4,
                 weight_decay: float=0.01,
                 scheduler_max_steps: int=200_000,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.vae_learning_rate = vae_learning_rate
        self.codebook_learning_rate = codebook_learning_rate
        self.weight_decay = weight_decay
        self.steps = scheduler_max_steps


    def training_step(self, batch, batch_idx):
        frames, actions, rel_t, _, _, _ = batch
        B, T_plus_1, C, H, W = frames.shape
        T = T_plus_1 - 1
        
        # Flatten transitions: (B, T, ...) -> (B*T, ...)
        frame_t = frames[:, :-1].reshape(B * T, C, H, W)
        frame_t_H = frames[:, 1:].reshape(B * T, C, H, W)
        actions_t = actions.reshape(B * T, actions.shape[-2], actions.shape[-1])
        
        output = self.model(frame_t, frame_t_H, actions=actions_t)
        
        self.log("train/loss", output["loss"])
        self.log("train/loss_recon", output["loss_recon"])
        self.log("train/loss_vq_act", output["loss_vq_act"])
        self.log("train/loss_vq_env", output["loss_vq_env"])
        self.log("train/loss_contrastive", output["loss_contrastive"])
        self.log("train/loss_reg", output["loss_reg"])
        
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        frames, actions, rel_t, _, _, _ = batch
        B, T_plus_1, C, H, W = frames.shape
        T = T_plus_1 - 1
        
        frame_t = frames[:, :-1].reshape(B * T, C, H, W)
        frame_t_H = frames[:, 1:].reshape(B * T, C, H, W)
        actions_t = actions.reshape(B * T, actions.shape[-2], actions.shape[-1])
        
        output = self.model(frame_t, frame_t_H, actions=actions_t)
        
        self.log("val/loss", output["loss"])
        self.log("val/loss_recon", output["loss_recon"])
        self.log("val/loss_vq_act", output["loss_vq_act"])
        self.log("val/loss_vq_env", output["loss_vq_env"])
        self.log("val/loss_contrastive", output["loss_contrastive"])

        # if batch_idx == 0:
        #     self.log_videos()

        
        return output["loss"]
    
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


    def configure_optimizers(self):
        # Separate learning rates for model and env_vq codebook as requested
        vq_params = list(self.model.env_vq.parameters())
        base_params = [p for n, p in self.model.named_parameters() if 'env_vq' not in n]

        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': self.hparams.vae_learning_rate},
            {'params': vq_params, 'lr': self.hparams.codebook_learning_rate}
        ], weight_decay=self.hparams.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.scheduler_max_steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]


def viridis_colormap() -> Tensor:
    """(256, 3) RGB lookup table, to color the tactile maps."""
    return torch.tensor(matplotlib.colormaps['viridis'](np.linspace(0, 1, 256))[:, :3], dtype=torch.float32)


def tactile_to_rgb(maps, mask, size, colormap):
    """(B, T, 2, 21, 21) maps in [0, 1] and (B, 2, 21, 21) glove masks -> (B, T, 3, H, W), the left glove on the left,
    black off the glove, on a grey background."""
    rgb = colormap[(maps.clamp(0, 1) * 255).round().long()] * mask[:, None, ..., None]   # (B, T, 2, 21, 21, 3)
    cells = maps.shape[-1]
    scale = min(size[0], size[1] // 2) // cells
    rgb = rgb.repeat_interleave(scale, dim=-3).repeat_interleave(scale, dim=-2).permute(0, 1, 2, 5, 3, 4)
    canvas = rgb.new_full((*maps.shape[:2], 3, *size), 0.25)   # grey, to tell the panels apart
    side, top = cells * scale, (size[0] - cells * scale) // 2
    for glove in range(2):
        left = glove * size[1] // 2 + (size[1] // 2 - side) // 2
        canvas[..., top:top + side, left:left + side] = rgb[:, :, glove]
    return canvas


class TactileVDVAETraining(L.LightningModule):
    """Trains lam.lam.TactileVDVAE on lam.lightning.egotouch.EgoTouchDataModule batches."""
    LOGGED = ('loss', 'loss_recon', 'loss_recon_copy', 'loss_tactile', 'loss_pressure', 'loss_contact', 'loss_bend',
              'loss_vq_act', 'loss_vq_env', 'loss_reg', 'pressure_mae', 'pressure_mae_copy', 'bend_mae', 'bend_mae_copy',
              'contact_f1', 'contact_f1_copy', 'act_codes_used', 'env_codes_used')
    LOGGED_SHUFFLED = ('loss_recon', 'pressure_mae', 'bend_mae', 'contact_f1')

    def __init__(self,
                 model: nn.Module,
                 camera: str = 'chest',
                 learning_rate: float = 2e-4,
                 weight_decay: float = 1e-4,
                 scheduler_max_steps: int = 100_000,
                 num_logged_samples: int = 4,
                 image_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),   # the dataset's normalization, to show the frames
                 image_std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.register_buffer('colormap', viridis_colormap(), persistent=False)

    def shared_step(self, batch, **kwargs):
        videos, tactile, _, rel_t, _ = batch
        frames = videos[self.hparams.camera]
        return self.model(frames, tactile, rel_t, **kwargs), frames.shape[0]

    def training_step(self, batch, batch_idx):
        output, batch_size = self.shared_step(batch)
        for key in self.LOGGED:
            self.log(f"train/{key}", output[key], batch_size=batch_size)
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output, batch_size = self.shared_step(batch)
        for key in self.LOGGED:
            self.log(f"val/{key}", output[key], batch_size=batch_size)

        # with the latents of other transitions: the gap to the real ones is what the latents carry
        shuffled, _ = self.shared_step(batch, shuffle_latent=True)
        for key in self.LOGGED_SHUFFLED:
            self.log(f"val/shuffled_latent_{key}", shuffled[key], batch_size=batch_size)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            self.log_videos(batch)
        return output["loss"]

    @torch.no_grad()
    def log_videos(self, batch):
        """One video frame per transition, one row per clip: the frame t+H, then the DINOv3 patches of t+H (real,
        predicted from the inferred action, predicted from a random action), then the gloves at t+H (same order).
        There is no DINOv3 decoder back to pixels: the patches are drawn as their 3 principal components."""
        videos, tactile, tactile_mask, _, _ = batch
        n = self.hparams.num_logged_samples
        frames, tactile, tactile_mask = videos[self.hparams.camera][:n], tactile[:n], tactile_mask[:n]
        outputs = self.model.visualize(frames, tactile)

        mean = torch.tensor(self.hparams.image_mean, device=frames.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(self.hparams.image_std, device=frames.device).view(1, 1, 3, 1, 1)
        images = (frames[:, 1:].float() * std + mean).clamp(0, 1)
        size = images.shape[-2:]
        with torch.autocast(device_type=frames.device.type, enabled=False):  # no QR decomposition in bfloat16
            features = self.features_to_rgb([outputs[f"features_{k}"] for k in ("target", "recon", "imagined")], size)
        touch = [self.tactile_to_rgb(outputs[f"tactile_{k}"], tactile_mask, size) for k in ("target", "recon", "imagined")]

        panels = [images, *features, *touch]
        B, T = images.shape[:2]
        x = torch.stack([make_grid([panel[b, t] for b in range(B) for panel in panels], nrow=len(panels)) for t in range(T)])
        caption = "frame t+H | DINOv3 PCA: real, reconstructed, imagined | touch: real, reconstructed, imagined"
        self.logger.experiment.log({"val_videos": wandb.Video((x * 255).to('cpu', torch.uint8).numpy(), fps=2, caption=caption)})

    @staticmethod
    def features_to_rgb(features, size):
        """[(B, T, P, F)] -> [(B, T, 3, H, W)]: the 3 principal components of each clip's real patches (the first
        tensor), colors scaled on them, so that the predictions are drawn in the same colors."""
        B, T, P, _ = features[0].shape
        patch = round((size[0] * size[1] / P) ** 0.5)
        grid = (size[0] // patch, size[1] // patch)
        images = [[] for _ in features]
        for b in range(B):
            target = features[0][b].flatten(0, 1).float()
            center = target.mean(0)
            _, _, components = torch.pca_lowrank(target - center, q=3, center=False)
            projected = [(f[b].float() - center) @ components for f in features]       # (T, P, 3) each
            low, high = projected[0].flatten(0, 1).min(0).values, projected[0].flatten(0, 1).max(0).values
            for i, p in enumerate(projected):
                rgb = ((p - low) / (high - low).clamp(min=1e-6)).clamp(0, 1)
                rgb = rgb.transpose(1, 2).unflatten(-1, grid)                            # (T, 3, h, w)
                images[i].append(F.interpolate(rgb, size=size, mode='nearest'))
        return [torch.stack(i) for i in images]

    def tactile_to_rgb(self, maps, mask, size):
        return tactile_to_rgb(maps, mask, size, self.colormap)

    def configure_optimizers(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.scheduler_max_steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]


class TactileLAMTraining(LAMTraining):
    """Trains lam.lam.TactileLAM on lam.lightning.egotouch.EgoTouchDataModule batches (frames in [0, 1]: normalize=False).

    LAMTraining's loss, metrics and validation videos are kept; the tactile loss is added with tactile_loss_weight. In
    validation, the tactile predictions with random action tokens are logged as val/imagined_* (the tactile counterpart
    of delta_psnr), and a second video shows the gloves at t+1: real, reconstructed, imagined with random actions.
    """
    TACTILE_LOGGED = ('loss_tactile', 'loss_pressure', 'loss_contact', 'loss_bend', 'pressure_mae', 'pressure_mae_copy',
                      'bend_mae', 'bend_mae_copy', 'contact_f1', 'contact_f1_copy')
    TACTILE_IMAGINED = ('pressure_mae', 'bend_mae', 'contact_f1')

    def __init__(self,
                 model: nn.Module,
                 camera: str = 'chest',
                 tactile_loss_weight: float = 0.01,
                 num_logged_samples: int = 4,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.camera = camera
        self.tactile_loss_weight = tactile_loss_weight
        self.num_logged_samples = num_logged_samples
        self.register_buffer('colormap', viridis_colormap(), persistent=False)

    def shared_step(self, batch, prefix):
        videos, tactile, tactile_mask, _, _ = batch
        images = videos[self.camera]
        output = self.model(images, tactile)
        loss = (output["recon_loss"] + self.commit_loss_weight * output["commit_loss"]
                + self.perceptual_loss_weight * output["perceptual_loss"] + self.tactile_loss_weight * output["loss_tactile"])

        batch_size = images.shape[0]
        self.log(f"{prefix}/loss", loss, batch_size=batch_size)
        for key in ("recon_loss", "perceptual_loss", "commit_loss", *self.TACTILE_LOGGED):
            self.log(f"{prefix}/{key}", output[key], batch_size=batch_size)
        return images, tactile, tactile_mask, output, loss

    def training_step(self, batch, batch_idx):
        images, _, _, output, loss = self.shared_step(batch, "train")

        if batch_idx % self.compute_metrics_every_n_steps == 0 and output["recon_video"] is not None:
            with torch.no_grad():
                self.compute_metrics(images, output["recon_video"], output["features"], output["recon_features"], output["indices"], output["action_tokens"], prefix="train")

        self.replace_unused_codebook_entries()      # Only used in NSVQ
        return loss

    def validation_step(self, batch, batch_idx):
        images, tactile, tactile_mask, output, loss = self.shared_step(batch, "val")

        imagined = self.model.tactile_with_random_actions(output["features"], tactile)
        imagined_metrics = self.model.tactile_losses(imagined, tactile[:, :-1].flatten(0, 1), tactile[:, 1:].flatten(0, 1))
        for key in self.TACTILE_IMAGINED:
            self.log(f"val/imagined_{key}", imagined_metrics[key], batch_size=images.shape[0])

        # without a pixel decoder (DINO features and no decoder_path), there is no video to compute the image metrics on
        random_reconstruction = None
        if output["recon_video"] is not None:
            random_reconstruction = self.compute_metrics(images, output["recon_video"], output["features"], output["recon_features"], output["indices"], output["action_tokens"], prefix="val", return_random_reconstruction=True)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            if output["recon_video"] is not None and images.shape[0] >= 4:      # LAMTraining.log_videos shows 4 clips
                self.log_videos(images, output["recon_video"], random_reconstruction)
            self.log_tactile_videos(images, tactile, tactile_mask, output["tactile_prediction"], imagined)
        return loss

    @torch.no_grad()
    def log_tactile_videos(self, images, tactile, tactile_mask, tactile_prediction, imagined):
        """One video frame per transition, one row per clip: the frame t+1, then the gloves at t+1 (real, reconstructed
        from the inferred action, imagined from a random action)."""
        n = self.num_logged_samples
        B, T = tactile.shape[:2]
        tactile_t = tactile[:, :-1].flatten(0, 1)
        maps = {"real": tactile[:, 1:],
                "reconstructed": self.model.predicted_maps(tactile_prediction, tactile_t).unflatten(0, (B, T - 1)),
                "imagined": self.model.predicted_maps(imagined, tactile_t).unflatten(0, (B, T - 1))}
        frames = images[:n, 1:].float().clamp(0, 1)
        size = frames.shape[-2:]
        panels = [frames] + [tactile_to_rgb(m[:n].float(), tactile_mask[:n], size, self.colormap) for m in maps.values()]

        x = torch.stack([make_grid([panel[b, t] for b in range(frames.shape[0]) for panel in panels], nrow=len(panels))
                         for t in range(T - 1)])
        caption = "frame t+1 | touch: " + ", ".join(maps)
        self.logger.experiment.log({"val_tactile_videos": wandb.Video((x * 255).to('cpu', torch.uint8).numpy(), fps=2, caption=caption)})
