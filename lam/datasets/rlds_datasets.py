"""
Adapted from: https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/datasets.py

rlds_datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Protocol, Tuple, Union, Sequence

import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from PIL import Image
import numpy as np
from einops import rearrange

from lam.datasets.rlds.utils.data_utils import tree_map
from lam.datasets.rlds.dataset import make_interleaved_dataset, make_single_dataset
from lam.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from lam.datasets.rlds.oxe.materialize import get_oxe_dataset_kwargs_and_weights
from lam.datasets.rlds.utils.data_utils import NormalizationType

IGNORE_INDEX = -100


# === Interface for an Image Transform ===
class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...


class RLDSBatchTransform:
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to a format suitable for training a model."""         
        frames = rlds_batch["observation"]["image_primary"]
        sample = rearrange(torch.tensor(frames, dtype=torch.float32), "t h w c -> t c h w") / 255.0

        return sample
    

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: int = 1,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_images=True,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=False,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                            # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=False,                               # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
    

class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)


@dataclass
class PaddedCollatorForActionPrediction:
    language_model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    action_model_max_length: int = 1024

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, actions = tuple([instance[key] for instance in instances] for key in ("input_ids", "actions"))

        # Add readout token to the end of each action sequence
        actions = [torch.cat([action, torch.zeros((1, action.size(-1)))]) for action in actions]

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=IGNORE_INDEX)
        actions = pad_sequence(actions, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids = input_ids[:, : self.language_model_max_length]
        actions = actions[:, : self.action_model_max_length]

        # Get padding attention masks
        language_pad_attention_mask = input_ids.ne(IGNORE_INDEX)
        action_pad_attention_mask = actions.ne(IGNORE_INDEX).sum(-1).bool()

        # Once we have the padding mask, we can replace the padding tokens with the pad_token_id 
        # Necessary because pad_token_id is the same as eos_token_id
        input_ids.masked_fill_(~language_pad_attention_mask, self.pad_token_id)

        output = dict(
            language_tokens=input_ids,
            action_tokens=actions,
            language_pad_attention_mask=language_pad_attention_mask,
            action_pad_attention_mask=action_pad_attention_mask,
        )

        return output