import hydra
import torch
import json
import numpy as np
import pickle

from tqdm import tqdm
from hydra.utils import call, instantiate
from einops import rearrange
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

    # Step 1: Sort by index
    sorted_indices, sort_order = torch.sort(indices)
    sorted_data = data[sort_order]

    # Step 2: Find group boundaries
    unique_indices, counts = torch.unique_consecutive(sorted_indices, return_counts=True)
    grouped_data = torch.split(sorted_data, counts.tolist())

    # Step 3: Initialize result with empty tensors
    if not result:
        result = [torch.empty(0, F, dtype=data.dtype, device="cpu") for _ in range(N)]

    # Step 4: Fill in the known groups
    for idx, group in zip(unique_indices.tolist(), grouped_data):
        result[idx] = torch.cat((result[idx], group.to("cpu")), dim=0) if result[idx].numel() else group.to("cpu")

    return result


@hydra.main(config_path="conf", config_name="test")
def main(cfg):
    if (model_config_path := cfg.get("model_config")) is None:
        model_config_path = "/".join(cfg.lightning_module.checkpoint_path.split("/")[0:-1]) + "/model_config.yaml"

    model_config = OmegaConf.load(model_config_path)
    backbone = instantiate(model_config.algo.model)
    model = call(cfg.lightning_module, model=backbone)
    model.freeze()

    data_module = instantiate(cfg.data.data_module, window_size=model.model.encoder.temporal_pos_embedding.num_embeddings, add_actions=True)
    dataloader = data_module.train_dataloader()

    latent_to_real_actions = {"all_contexts": None,
             "only_min_context": None,
             "only_max_context": None}
    
    raw_actions = {"actions": [],
                   "indices": []}

    pbar = tqdm(total=len(dataloader), desc="Processing samples", unit="sample")

    for i, sample in enumerate(dataloader):
        images, actions = sample

        with torch.no_grad():
            logits, readout = model.model.encode(images.to(model.device))
            readout = readout[:, 1:, :]                                                 # Ignore the first action token, it doesn't represent any transition
            action_tokens, indices, commit_loss = model.model.vq.quantize(readout)

        raw_actions["actions"].append(actions)
        raw_actions["indices"].append(indices)

        for idx in range(1, indices.shape[1]):
            latent_to_real_actions["all_contexts"] = group_by_index(data=(actions[:, idx] - actions[:, idx-1]).to(model.device),
                                                   indices=indices[:, idx],
                                                   N=model.model.vq.codebook_size,
                                                   result=latent_to_real_actions["all_contexts"])

        latent_to_real_actions["only_min_context"] = group_by_index(data=(actions[:, 1] - actions[:, 0]).to(model.device),
                                                   indices=indices[:, 1],
                                                   N=model.model.vq.codebook_size,
                                                   result=latent_to_real_actions["only_min_context"])
        
        latent_to_real_actions["only_max_context"] = group_by_index(data=(actions[:, -2] - actions[:, -3]).to(model.device),
                                                   indices=indices[:, -1],
                                                   N=model.model.vq.codebook_size,
                                                   result=latent_to_real_actions["only_min_context"])
        
        pbar.update(1)

        if (i + 1) == len(dataloader):
            break


    with open(f"latent_to_real_actions_{cfg.lightning_module.checkpoint_path.split('/')[-2]}_{cfg.data.data_module.data_mix}.pkl", "wb") as file:
        pickle.dump(latent_to_real_actions, file)

    with open(f"raw_actions_{cfg.lightning_module.checkpoint_path.split('/')[-2]}_{cfg.data.data_module.data_mix}.pkl", "wb") as file:
        pickle.dump(raw_actions, file)

    print("Done!")

if __name__ == "__main__":
    main()