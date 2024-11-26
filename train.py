import torch
import hydra

from lightning.pytorch import seed_everything
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
    trainer.fit(model=lightning_module, datamodule=data_module, ckpt_path=config.checkpoint_path)


if __name__ == "__main__":
    train()