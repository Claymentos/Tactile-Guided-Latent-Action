"""Entraînement de lam.jala.JALA sur les batches de lam.lightning.egodex.EgoDexDataModule :
(frames (B, T, 3, H, W) normalisées ImageNet, actions (B, T-1, 88, 3), rel_t, heatmaps, valid_kp, metadata)."""

import math

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_warn
from torch import nn

from lam.jala import CYCLE_METRICS, attention_focus, cycle_summary, keypoint_patch_masks
from lam.metrics import partner_indices


class JALATraining(L.LightningModule):
    LOGGED = ('loss', 'loss_align', 'loss_action', 'loss_probe', 'loss_z_action', 'loss_variance', 'action_mae',
              'action_mae_zero', 'probe_mae', 'labeled_frac', 'loss_forward_probe', 'forward_probe_copy',
              'probe_finger_mae', 'finger_motion_zero', 'probe_wrist_mae', 'wrist_motion_zero', 'loss_mcp',
              'mcp_accuracy', 'mcp_masked_frac', 'tokenizer_mae')

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.05,
                 betas=(0.9, 0.95),
                 warmup_ratio: float = 0.05,
                 scheduler_max_steps: int = 100_000,
                 use_task: bool = True,
                 cycle_consistency_batches: int = 8,
                 num_logged_samples: int = 4,
                 arrow_scale: float = 3.0,
                 finger_focus_batches: int = 8):
        """Optimiseur et schedule de JALA : AdamW β = (0.9, 0.95), weight decay 0.05, 5 % de warmup puis cosinus.
        use_task : passe metadata['task_id'] au prédicteur (le modèle doit avoir num_tasks > 0).
        cycle_consistency_batches : batches de validation (les premiers) sur lesquels le test de cycle A -> B -> A est
            calculé (val/cycle/*) ; 0 le désactive. Au premier batch, la vidéo val/cycle_video (lam.jala_viz) et sa
            première frame val/cycle_image sont envoyées à W&B pour num_logged_samples paires de clips.
        arrow_scale : agrandissement des flèches de mouvement des mains dans les visualisations.
        finger_focus_batches : batches de validation sur lesquels on mesure où regarde LAP (val/focus/*) ; au premier,
            la vidéo val/finger_attention_video (lam.jala_viz.finger_attention_video). 0 le désactive."""
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate, self.weight_decay, self.betas = learning_rate, weight_decay, tuple(betas)
        self.warmup_ratio, self.steps = warmup_ratio, scheduler_max_steps
        self.use_task = use_task
        self.cycle_consistency_batches = cycle_consistency_batches
        self.num_logged_samples, self.arrow_scale = num_logged_samples, arrow_scale
        self._cycle_sums = None
        self.finger_focus_batches = finger_focus_batches
        self._focus_sums = {}

    def on_fit_start(self):
        num_tasks = getattr(self.trainer.datamodule, 'num_tasks', None) if self.trainer.datamodule is not None else None
        embedding = getattr(self.model.predictor, 'task_embedding', None)      # absent avec le VLM (instruction)
        if self.use_task and num_tasks and embedding is not None and num_tasks > embedding.num_embeddings:
            rank_zero_warn(f"{num_tasks} tâches EgoDex pour num_tasks={embedding.num_embeddings} : des tâches partagent "
                           f"un embedding (algo.model.num_tasks={num_tasks} l'évite)")

    def shared_step(self, batch, prefix):
        frames, actions, _, _, _, metadata = batch
        task_ids = metadata['task_id'].to(self.device).long() if self.use_task and 'task_id' in metadata else None
        output = self.model(frames, actions, task_ids, hand_chunks=metadata.get('hand_chunks'),
                            instructions=metadata.get('instruction'))
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
        output = self.shared_step(batch, 'val')
        self.focus_step(batch, batch_idx, output)
        self.cycle_step(batch, batch_idx)
        return output['loss']

    def on_validation_epoch_start(self):
        self._cycle_sums, self._focus_sums = None, {}

    @torch.no_grad()
    def focus_step(self, batch, batch_idx, output):
        """Les actions latentes regardent-elles les doigts ? Attention des queries de LAP comparée aux patchs où se
        projettent les doigts, les bouts des doigts et le poignet (metadata['cam0_to_camt'] nécessaire)."""
        frames, actions, _, _, _, metadata = batch
        if batch_idx >= self.finger_focus_batches or not isinstance(metadata, dict) or 'cam0_to_camt' not in metadata:
            return
        from lam.jala import NUM_HAND_KEYPOINTS
        from lam.jala_viz import project
        attention = self.model.lap_attention(output['features'])                          # (B, S, K, P)
        S = attention.shape[1]
        cameras = metadata['cam0_to_camt'][:, :S].to(frames.device).float()
        pixels, valid = project(actions[:, :, :NUM_HAND_KEYPOINTS].float(), cameras, frames.shape[-2:])
        grid = self.model.grid_size
        masks = keypoint_patch_masks(pixels, valid, grid, frames.shape[-2] // grid[0])
        for key, (total, count) in attention_focus(attention, masks).items():
            previous = self._focus_sums.get(key, (0.0, 0.0))
            self._focus_sums[key] = (previous[0] + total.float(), previous[1] + count.float())

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            import wandb
            from lam.jala_viz import finger_attention_video
            video = finger_attention_video(frames, actions, metadata['cam0_to_camt'], attention,
                                           self.model.probe(output['z']), grid, self.num_logged_samples, self.arrow_scale)
            caption = ("1 : attention moyenne des queries de LAP (part et lift sur doigts / bouts / poignet) ; "
                       "2-3 : queries des têtes gauche / droite (part sur chaque main) ; "
                       "4 : articulation des doigts relative au poignet, réelle (vert) et décodée de z (bleu).")
            self.logger.experiment.log({'val/finger_attention_video': wandb.Video(video, fps=1, caption=caption),
                                        'val/finger_attention_image': wandb.Image(video[0].transpose(1, 2, 0), caption=caption),
                                        'trainer/global_step': self.global_step})

    @torch.no_grad()
    def cycle_step(self, batch, batch_idx):
        """Test de cycle : chaque clip A est apparié à un clip B d'un autre épisode du batch."""
        if self.model.forward_probe is None or batch_idx >= self.cycle_consistency_batches or batch[0].shape[0] < 2:
            return
        frames, actions, _, _, _, metadata = batch
        partner = partner_indices(metadata.get('episode')) if isinstance(metadata, dict) else None
        if partner is None:
            partner = torch.roll(torch.arange(frames.shape[0]), frames.shape[0] // 2)
        partner = partner.to(frames.device)
        cycle = self.model.cycle_consistency(frames, frames[partner], actions)
        sums = torch.stack([cycle[k].float().sum() for k in CYCLE_METRICS] +
                           [torch.tensor(float(frames.shape[0]), device=frames.device)])
        self._cycle_sums = sums if self._cycle_sums is None else self._cycle_sums + sums

        if batch_idx == 0 and isinstance(self.logger, WandbLogger) and 'cam0_to_camt' in metadata:
            self.log_cycle_video(frames, actions, metadata['cam0_to_camt'], partner, cycle)

    def log_cycle_video(self, frames, actions, cameras, partner, cycle):
        import wandb
        from lam.jala_viz import cycle_video
        video = cycle_video(frames, frames[partner], actions, actions[partner], cameras, cameras[partner.to(cameras.device)],
                            cycle, self.model.grid_size, self.num_logged_samples, self.arrow_scale)
        caption = ("ligne = paire (A, B). A : mouvement reel (vert), decode de z_A (bleu), de z_cycle (rouge). "
                   "B : reel (vert), action de A transferee (bleu). Puis B t+1 reel et B' = B_t + z_A en PCA DINO.")
        self.logger.experiment.log({'val/cycle_video': wandb.Video(video, fps=1, caption=caption),
                                    'val/cycle_image': wandb.Image(video[0].transpose(1, 2, 0), caption=caption),
                                    'trainer/global_step': self.global_step})

    def on_validation_epoch_end(self):
        for key, (total, count) in self._focus_sums.items():
            pair = self.trainer.strategy.reduce(torch.stack([total, count]), reduce_op='sum')
            if pair[1] > 0:
                self.log(f'val/focus/{key}', float(pair[0] / pair[1]))
        self._focus_sums = {}
        if self._cycle_sums is None:
            return
        sums = self.trainer.strategy.reduce(self._cycle_sums, reduce_op='sum')      # somme sur les GPU
        means = {k: float(v) for k, v in zip(CYCLE_METRICS, (sums[:-1] / sums[-1]).tolist())}
        for key, value in cycle_summary(means).items():
            self.log(f'val/cycle/{key}', value)
        self._cycle_sums = None

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
