import hydra
import torch
import pickle

from tqdm import tqdm
from hydra.utils import call, instantiate
from omegaconf import OmegaConf
from typing import List


def group_by_index(data: torch.Tensor, indices: torch.Tensor, N: int, result: List[torch.Tensor] = None) -> List[torch.Tensor]:
    """
    Groups data by integer indices in [0, N-1], returning a list of length N.
    If an index is missing, its group is an empty tensor of shape (0, F).
    
    Args:
        data: Tensor of shape (B, F)
        indices: Tensor of shape (B,) with values in [0, N-1]
        N: Total number of possible groups
        result: Optional preallocated list to store results. If not provided, a new list will be created.

    Returns:
        A list of N tensors; each tensor has shape (M_i, F) where M_i is number of items with index i
    """
    B, F = data.shape
    assert indices.shape[0] == B
    assert indices.dtype == torch.long

    if not result:
        result = [torch.empty(0, F, dtype=data.dtype, device="cpu") for _ in range(N)]

    for idx, sample in zip(indices, data):
        result[idx] = torch.cat((result[idx], sample.unsqueeze(0)), dim=0) if result[idx].numel() else sample.unsqueeze(0)

    return result


@hydra.main(config_path="conf", config_name="test")
def main(cfg):
    if (model_config_path := cfg.get("model_config")) is None:
        model_config_path = "/".join(cfg.lightning_module.checkpoint_path.split("/")[0:-1]) + "/model_config.yaml"

    model_config = OmegaConf.load(model_config_path)
    backbone = instantiate(model_config.algo.model)
    model = call(cfg.lightning_module, model=backbone)
    model.freeze()

    data_module = instantiate(cfg.data.data_module, window_size=model.model.encoder.temporal_pos_embedding.num_embeddings, image_size=model_config.algo.model.feature_extractor.image_size, add_actions=True)
    dataloader = data_module.train_dataloader()

    latent_to_real_actions, latent_to_real_actions_diff, latent_to_real_proprio_diff = None, None, None

    raw_data = {"indices": None,
                "proprio_diff": None,
                "action_diff": None,
                "action_0": None}

    pbar = tqdm(total=len(dataloader), desc="Processing samples", unit="sample")

    for i, sample in enumerate(dataloader):
        with torch.no_grad():
            action_tokens, indices = model.model.get_latent_actions(sample["images"].to(model.device))

        if raw_data["indices"] is None:
            raw_data["proprio_diff"] = (sample["proprio"][:, 1] - sample["proprio"][:, 0]).cpu()
            raw_data["action_diff"] = (sample["actions"][:, 1] - sample["actions"][:, 0]).cpu()
            raw_data["action_0"] = sample["actions"][:, 0].cpu()
            raw_data["indices"] = indices.cpu()
        else:
            raw_data["proprio_diff"] = torch.cat((raw_data["proprio_diff"], (sample["proprio"][:, 1] - sample["proprio"][:, 0]).cpu()), dim=0)
            raw_data["action_diff"] = torch.cat((raw_data["action_diff"], (sample["actions"][:, 1] - sample["actions"][:, 0]).cpu()), dim=0)
            raw_data["action_0"] = torch.cat((raw_data["action_0"], sample["actions"][:, 0].cpu()), dim=0)
            raw_data["indices"] = torch.cat((raw_data["indices"], indices.cpu()), dim=0)

        latent_to_real_actions = group_by_index(data=sample["actions"][:, 0].cpu(),
                                                indices=indices,
                                                N=model.model.vq.codebook_size,
                                                result=latent_to_real_actions)

        latent_to_real_actions_diff = group_by_index(data=(sample["actions"][:, 1] - sample["actions"][:, 0]).cpu(),
                                                      indices=indices,
                                                      N=model.model.vq.codebook_size,
                                                      result=latent_to_real_actions_diff)

        latent_to_real_proprio_diff = group_by_index(data=(sample["proprio"][:, 1] - sample["proprio"][:, 0]).cpu(),
                                                      indices=indices,
                                                      N=model.model.vq.codebook_size,
                                                      result=latent_to_real_proprio_diff)

        pbar.update(1)

        if (i + 1) == len(dataloader):
            break

    with open(f"latent_to_real_actions_{cfg.lightning_module.checkpoint_path.split('/')[-2]}_{cfg.data.data_module.data_mix}.pkl", "wb") as file:
        pickle.dump(latent_to_real_actions, file)

    with open(f"latent_to_real_actions_diff_{cfg.lightning_module.checkpoint_path.split('/')[-2]}_{cfg.data.data_module.data_mix}.pkl", "wb") as file:
        pickle.dump(latent_to_real_actions_diff, file)

    with open(f"latent_to_real_proprio_diff_{cfg.lightning_module.checkpoint_path.split('/')[-2]}_{cfg.data.data_module.data_mix}.pkl", "wb") as file:
        pickle.dump(latent_to_real_proprio_diff, file)

    with open(f"raw_data_{cfg.lightning_module.checkpoint_path.split('/')[-2]}_{cfg.data.data_module.data_mix}.pkl", "wb") as file:
        pickle.dump(raw_data, file)

    print("Done!")

if __name__ == "__main__":
    main()