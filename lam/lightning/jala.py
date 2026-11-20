"""Entraînement de lam.jala.JALA sur les batches de lam.lightning.egodex.EgoDexDataModule :
(frames (B, T, 3, H, W) normalisées ImageNet, actions (B, T-1, 88, 3), rel_t, heatmaps, valid_kp, metadata)."""

import math

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_warn
from torch import nn


class JALATraining(L.LightningModule):
    LOGGED = ('loss', 'loss_align', 'loss_action', 'loss_probe', 'loss_z_action', 'loss_variance', 'action_mae',
              'action_mae_zero', 'probe_mae', 'labeled_frac')

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.05,
                 betas=(0.9, 0.95),
                 warmup_ratio: float = 0.05,
                 scheduler_max_steps: int = 100_000,
                 use_task: bool = True):
        """Optimiseur et schedule de JALA : AdamW β = (0.9, 0.95), weight decay 0.05, 5 % de warmup puis cosinus.
        use_task : passe metadata['task_id'] au prédicteur (le modèle doit avoir num_tasks > 0)."""
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate, self.weight_decay, self.betas = learning_rate, weight_decay, tuple(betas)
        self.warmup_ratio, self.steps = warmup_ratio, scheduler_max_steps
        self.use_task = use_task

    def on_fit_start(self):
        num_tasks = getattr(self.trainer.datamodule, 'num_tasks', None) if self.trainer.datamodule is not None else None
        embedding = self.model.predictor.task_embedding
        if self.use_task and num_tasks and embedding is not None and num_tasks > embedding.num_embeddings:
            rank_zero_warn(f"{num_tasks} tâches EgoDex pour num_tasks={embedding.num_embeddings} : des tâches partagent "
                           f"un embedding (algo.model.num_tasks={num_tasks} l'évite)")

    def shared_step(self, batch, prefix):
        frames, actions, _, _, _, metadata = batch
        task_ids = metadata['task_id'].to(self.device).long() if self.use_task and 'task_id' in metadata else None
        output = self.model(frames, actions, task_ids)
        batch_size = frames.shape[0]
        for key in self.LOGGED:
            if key in output:
                self.log(f"{prefix}/{key}", output[key], batch_size=batch_size, prog_bar=key == 'loss')
        if prefix == 'val' or self.global_step % max(self.trainer.log_every_n_steps, 1) == 0:
            for key, value in self.model.diagnostics(output).items():
                self.log(f"{prefix}/{key}", value, batch_size=batch_size)
        return output

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')['loss']

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model.ema_update()

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')['loss']

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue        # DINOv3 figé, backbone de LAP et queries de LSP (mis à jour en EMA)
            (no_decay if parameter.ndim < 2 or 'embedding' in name or 'queries' in name else decay).append(parameter)
        optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': self.weight_decay},
                                       {'params': no_decay, 'weight_decay': 0.0}], lr=self.learning_rate, betas=self.betas)
        warmup = max(1, int(self.warmup_ratio * self.steps))

        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, self.steps - warmup))
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]
