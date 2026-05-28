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
from lam.lam import UncontrolledDINOLatentActionModel, ControllableDINOLatentActionModel
import matplotlib.pyplot as plt
from accelerate import PartialState

class DINO_LAM(L.LightningModule):

    """
    A latent action model operates at the DINO latent space
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            stage: str = 'stage-1',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            stage_one_ckpt: str = None,
    ) -> None:
        super(DINO_LAM, self).__init__()
        assert stage in ['stage-1', 'stage-2']

        lam = UncontrolledDINOLatentActionModel if stage == 'stage-1' else ControllableDINOLatentActionModel

        self.lam = lam(
                    in_dim=image_channels,
                    model_dim=lam_model_dim,
                    latent_dim=lam_latent_dim,
                    num_latents=lam_num_latents,
                    patch_size=lam_patch_size,
                    enc_blocks=lam_enc_blocks,
                    dec_blocks=lam_dec_blocks,
                    num_heads=lam_num_heads,
                    dropout=lam_dropout,
                )
        
        if stage_one_ckpt and path.exists(stage_one_ckpt):
            lam_ckpt = torch.load(stage_one_ckpt)['state_dict']
            stage1_ckpt = {}
            for key in lam_ckpt.keys():
                if 'vq' in key or 'action_latent' in key:
                    stage1_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
            self.lam.load_state_dict(stage1_ckpt, strict=False)


        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']

        outputs = self.lam(batch)
        gt_future_frames = outputs["target"]

        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        loss = mse_loss + q_loss + self.vq_beta * commit_loss
        
        # Optimize uncontrollable queries in stage-2 (the codebook is frozen though)
        if "z_q_uncontrol" in outputs.keys():
            q_loss_uncontrol = ((outputs["emb_uncontrol"].detach() - outputs["z_uncontrol"]) ** 2).mean()
            commit_loss_uncontrol = ((outputs["emb_uncontrol"]- outputs["z_uncontrol"].detach()) ** 2).mean()
            loss = loss + q_loss_uncontrol + self.vq_beta * commit_loss_uncontrol

        # Compute code usage
        unique, counts = torch.unique(outputs["indices"], return_counts=True)
        index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()

        loss_logs = (
            ("mse_loss", mse_loss),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("code_usage", code_usage),
        )

        if "indices_uncontrol" in outputs.keys():
            unique, counts = torch.unique(outputs["indices_uncontrol"], return_counts=True)
            index_counts = torch.zeros(32, dtype=torch.long).cuda()
            index_counts[unique] = counts
            uncontrol_code_usage = (index_counts != 0).float().mean()

            loss_logs = (
                ("mse_loss", mse_loss),
                ("q_loss", q_loss),
                ("commit_loss", commit_loss),
                ("q_loss_uncontrol", q_loss_uncontrol),
                ("commit_loss_uncontrol", commit_loss_uncontrol),
                ("code_usage", code_usage),
                ("code_usage_uncontrol", uncontrol_code_usage),
            )

        return outputs, loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)


        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        return loss

    def validation_step(self, batch, batch_idx):
        output, loss, aux_losses = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(
            {f"val/{k}": v for k, v in aux_losses},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True)
        if batch_idx == 0:
            self.log_videos(output["target"], output["recon"])
        
    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def on_train_epoch_end(self):
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

    #     self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
    #     self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    # def plot_usage_distribution(self, usage, filename):
    #     data = usage.cpu().numpy()
    #     n = 1
    #     for n in range(1, 10):
    #         if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
    #             break
    #     data = data.reshape(2 ** n, -1)
    #     fig, ax = plt.subplots()
    #     cax = ax.matshow(data, interpolation="nearest")
    #     fig.colorbar(cax)
    #     plt.axis("off")
    #     plt.gca().set_axis_off()
    #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #     plt.margins(0, 0)
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #     plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
    #     plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim

    def log_videos(self, target_view, recon_video):
        recon_video = torch.cat([target_view[:,:1], recon_video], dim=1)
        x = [make_grid([
            (target_view[0][t]*255).to('cpu', torch.uint8), (recon_video[0][t]*255).to('cpu', torch.uint8),
            (target_view[1][t]*255).to('cpu', torch.uint8), (recon_video[1][t]*255).to('cpu', torch.uint8),
            (target_view[2][t]*255).to('cpu', torch.uint8), (recon_video[2][t]*255).to('cpu', torch.uint8),
            (target_view[3][t]*255).to('cpu', torch.uint8), (recon_video[3][t]*255).to('cpu', torch.uint8)
        ], nrow=2) for t in range(recon_video.shape[1])]
        caption = "video | reconstruction"

        x = torch.stack(x, dim=0)
        video = wandb.Video(x, fps=2, caption=caption)
        self.logger.experiment.log({"val_videos": video})


class LAMVAETraining(L.LightningModule):
    def __init__(self,
            model: nn.Module,
            beta: float = 0.01,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            optimizer: OptimizerCallable = AdamW):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.beta = beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        target_view = batch["images"]
        output = self.model(target_view)

        loss = output["recon_loss"] + self.beta * output["kl_loss"]

        self.log("train/loss", loss)
        self.log("train/recon_loss", output["recon_loss"])
        self.log("train/kl_loss", output["kl_loss"])

        if batch_idx % self.log_interval == 0:
            with torch.no_grad():
                self.log_images(target_view, output["recon"])

        return loss

    def validation_step(self, batch, batch_idx):
        target_view = batch["images"]
        output = self.model(target_view)

        loss = output["recon_loss"] + self.beta * output["kl_loss"]

        self.log("val/loss", loss)
        self.log("val/recon_loss", output["recon_loss"])
        self.log("val/kl_loss", output["kl_loss"])

        if batch_idx == 0:
            self.log_images(target_view, output["recon"])

        return loss

    def log_images(self, target_view, recon):
        # target_view : (B, T, C, H, W)
        # recon       : (B, T-1, C, H, W) — frames futures reconstruites

        # Le modèle prédit x[:,1:], on aligne target sur les mêmes frames
        future_frames = target_view[:, 1:]

        x = [make_grid([
            (future_frames[0][t] * 255).to('cpu', torch.uint8),
            (recon[0][t].clamp(0, 1) * 255).to('cpu', torch.uint8),
            (future_frames[1][t] * 255).to('cpu', torch.uint8),
            (recon[1][t].clamp(0, 1) * 255).to('cpu', torch.uint8),
            (future_frames[2][t] * 255).to('cpu', torch.uint8),
            (recon[2][t].clamp(0, 1) * 255).to('cpu', torch.uint8),
            (future_frames[3][t] * 255).to('cpu', torch.uint8),
            (recon[3][t].clamp(0, 1) * 255).to('cpu', torch.uint8),
        ], nrow=2) for t in range(recon.size(1))]

        x = torch.stack(x, dim=0)
        video = wandb.Video(x, fps=2, caption="target | reconstruction")
        self.logger.experiment.log(
            {"val_images" if not self.model.training else "train_images": video},
            step=self.global_step
        )

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

class ActVAETraining(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 vq_beta: float=0.25,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.01,
                 scheduler_max_steps: int = 200_000,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.vq_beta = vq_beta
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = scheduler_max_steps

    def process_batch(self, batch):
        if isinstance(batch, list):
            actions = batch[1]
        elif isinstance(batch, dict):
            actions = batch["actions"]
        return actions
    
    def training_step(self, batch, batch_idx):

        batch = self.process_batch(batch)
        output = self.model(batch)

        loss = output["recon_loss"] + self.vq_beta * output["commit_loss"] + output["codebook_loss"]
        self.log("train/loss", loss)
        self.log("train/recon_loss", output["recon_loss"])
        self.log("train/commit_loss", output["commit_loss"])
        self.log("train/codebook_loss", output["codebook_loss"])

        return loss
    

    def validation_step(self, batch, batch_idx):

        batch = self.process_batch(batch)
        output = self.model(batch)

        loss = output["recon_loss"] + self.vq_beta * output["commit_loss"] + output["codebook_loss"]
        self.log("train/loss", loss)
        self.log("train/recon_loss", output["recon_loss"])
        self.log("train/commit_loss", output["commit_loss"])
        self.log("train/codebook_loss", output["codebook_loss"])

        if batch_idx == 0:
            self.log_trajectory_plot(batch, output["recon_actions"])


        return loss
    

    def log_trajectory_plot(self, gt_actions, recon_actions):
        """
        Génère un graphique de trajectoire en convertissant les tenseurs en float32.
        """
        # Ajout du .float() avant le .cpu().numpy()
        gt_sample = gt_actions[0, :, 0, :].float().cpu().numpy()          # Shape: (T, 3)
        recon_sample = recon_actions[0, :, 0, :].float().cpu().numpy()    # Shape: (T, 3)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        coords = ['X', 'Y', 'Z']
        
        for i in range(3):
            axes[i].plot(gt_sample[:, i], label="Ground Truth", color='black', linestyle='--')
            axes[i].plot(recon_sample[:, i], label="Reconstructed", color='crimson')
            axes[i].set_ylabel(f"Coord {coords[i]}")
            axes[i].grid(True)
            if i == 0:
                axes[i].legend()
                
        axes[-1].set_xlabel("Time step (Sequence frames)")
        plt.suptitle("Trajectory Alignment: Keypoint 0 (GT vs ActVAE Reconstruction)")
        
        # Vérification que le logger est disponible (pour éviter les erreurs lors du sanity check)
        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.log({"val/trajectory_match": wandb.Image(fig)})
        
        plt.close(fig)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
    
    