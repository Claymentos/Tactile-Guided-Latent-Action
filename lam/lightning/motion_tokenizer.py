"""Apprentissage du tokenizer de mouvement (lam.motion_tokenizer) sur metadata['hand_chunks'] d'EgoDex : première étape
de JALA avec la loss MCP (le tokenizer est ensuite figé, algo.model.motion_tokenizer_path)."""

import lightning as L
import torch

from lam.motion_tokenizer import MotionTokenizer


class MotionTokenizerTraining(L.LightningModule):
    def __init__(self, tokenizer: dict, learning_rate: float = 3e-4, weight_decay: float = 0.0,
                 finger_weight: float = 1.0, scheduler_max_steps: int = 50_000):
        """tokenizer : arguments de MotionTokenizer (sauvegardés dans le checkpoint, relus par from_checkpoint) ;
        finger_weight : poids de la reconstruction des doigts (mouvements plus petits que ceux du poignet)."""
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = MotionTokenizer(**tokenizer)
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.finger_weight, self.steps = finger_weight, scheduler_max_steps

    def shared_step(self, batch, prefix):
        metadata = batch[-1]
        chunks = metadata['hand_chunks'].float()                                # (B, S, C, 42, 3)
        out = self.tokenizer(chunks)
        loss = out['loss_wrist'] + self.finger_weight * out['loss_fingers'] + out['loss_commit']
        B = chunks.shape[0]
        for key in ('loss_wrist', 'loss_fingers', 'loss_commit', 'end_mae', 'end_zero'):
            self.log(f'{prefix}/{key}', out[key], batch_size=B)
        self.log(f'{prefix}/loss', loss, batch_size=B, prog_bar=True)
        # usage du codebook : part des codes utilisés dans le batch, par partie
        parts = self.tokenizer.token_part
        for i, name in enumerate(('wrist', 'fingers')):
            used = out['indices'][:, parts == i].unique().numel() / self.tokenizer.codebook_size
            self.log(f'{prefix}/codebook_usage_{name}', used, batch_size=B)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
