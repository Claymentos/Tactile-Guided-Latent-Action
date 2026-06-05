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
        self.log("val/loss", output["loss"], sync_dist=True)
        self.log("val/recon_loss", output["recon_loss"], sync_dist=True)
        self.log("val/vq_loss", output["vq_loss"], sync_dist=True)


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

        self.log("val/codebook_active_codes", float(num_active_codes), sync_dist=True)
        self.log("val/codebook_usage_ratio", usage_ratio, sync_dist=True)
        self.log("val/codebook_perplexity", perplexity, sync_dist=True)

        # --- 2. Calcul de la variance Intra-Code ---
        intra_code_variances = []
        for code_idx in range(codebook_size):
            mask = (all_indices == code_idx)
            if mask.sum() > 5: # Validité statistique minimale
                variance = torch.var(all_gt_poses[mask], dim=0).mean()
                intra_code_variances.append(variance)
        
        if intra_code_variances:
            mean_intra_variance = torch.stack(intra_code_variances).mean()
            self.log("val/codebook_intra_cluster_variance", mean_intra_variance, sync_dist=True)

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
        
        