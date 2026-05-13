import os
import torch
import hydra

from omegaconf import OmegaConf, DictConfig
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities import rank_zero_only
from hydra.utils import instantiate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="conf", config_name="train_stage_1")
def train(config):
    seed_everything(42, workers=True)

    lightning_module = instantiate(config.lightning_module)
    data_module = instantiate(config.data.data_module)
    
    trainer = instantiate(config.trainer)
    trainer.logger.log_hyperparams(flatten_config(config))

    trainer.checkpoint_callback.dirpath = f"{trainer.logger.experiment.dir}/../checkpoints/"
    save_config(config, trainer.checkpoint_callback.dirpath)

    trainer.fit(model=lightning_module, datamodule=data_module, ckpt_path=config.checkpoint_path)


@rank_zero_only
def save_config(config, path):
    os.makedirs(path, exist_ok=True)

    with open(f"{path}model_config.yaml", "w") as f:
        OmegaConf.save(config, f)


def flatten_config(config, parent_key='', sep='.'):
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    items = {}

    if isinstance(config, dict):
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_config(v, new_key, sep=sep))
    elif isinstance(config, list):
        for i, v in enumerate(config):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_config(v, new_key, sep=sep))
    else:
        items[parent_key] = config

    return items


if __name__ == "__main__":
    train()