import os
import torch
import hydra

from omegaconf import OmegaConf
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities import rank_zero_only
from hydra.utils import instantiate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(config):
    seed_everything(42, workers=True)

    lightning_module = instantiate(config.lightning_module)
    data_module = instantiate(config.data.data_module)
    
    trainer = instantiate(config.trainer)
    trainer.logger.log_hyperparams(config)

    trainer.checkpoint_callback.dirpath = f"{trainer.logger.experiment.dir}/../checkpoints/"
    save_config(config, trainer.checkpoint_callback.dirpath)

    trainer.fit(model=lightning_module, datamodule=data_module, ckpt_path=config.checkpoint_path)


@rank_zero_only
def save_config(config, path):
    os.makedirs(path, exist_ok=True)

    with open(f"{path}model_config.yaml", "w") as f:
        OmegaConf.save(config, f)


if __name__ == "__main__":
    train()