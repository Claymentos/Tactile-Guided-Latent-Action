import torch
import hydra

from lightning.pytorch import seed_everything
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(config):
    seed_everything(42, workers=True)

    if any(gpu in torch.torch.cuda.get_device_name() for gpu in ['A100', 'H100', 'RTX']):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    lightning_module = instantiate(config.lightning_module)
    data_module = instantiate(config.data.data_module)
    
    trainer = instantiate(config.trainer)
    trainer.fit(model=lightning_module, datamodule=data_module, ckpt_path=config.checkpoint_path)


if __name__ == "__main__":
    train()