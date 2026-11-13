"""EgoTouch (huggingface: zhouzhoujy/EgoTouch): RGB videos and tactile maps.

<root>/split.json lists the episodes of the official splits: train, val, test_seen, test_unseen (held-out tasks).
An episode is `<root>/<scene>/<task>/<YYYYMMDD_HHMMSS_mmm>/`, every stream on one 30 Hz timeline:
    chest.mp4, left.mp4, right.mp4   head/chest camera and two wrist cameras (4:3, 320x240 or 640x480)
    pressure_grids.npz               one 21x21 map per glove and per frame, in [0, 1], baseline-corrected, NaN off the
                                     glove. 217 cells are on the glove, 151 taxels and 15 bend sensors drawn as strips.

tactile_repr='mano_uv' (or 'mano_vertex') places the taxels on the MANO hand, the representation shared with ARCTIC
(lam/hand_contact.py): the bend sensors, which do not measure contact, are left out.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import tv_tensors

from lam.hand_contact import ManoUV, egotouch_vertex_weights, load_mano_assets
from lam.tactile import ALL_PRESSURE_CELLS, canonical_cells

CAMERAS = ('chest', 'left', 'right')
GLOVES = ('left', 'right')
SPLITS = ('train', 'val', 'test_seen', 'test_unseen')
SPLIT_FILE = 'split.json'
TACTILE_FILE = 'pressure_grids.npz'
NUM_SEGMENTS = 10  # each episode is divided into intervals so as not to load the same part of it every time
TACTILE_REPRS = ('grid', 'mano_vertex', 'mano_uv')


class EgoTouchDataset(Dataset):
    """Each item is the tuple (videos, tactile, tactile_mask, rel_t, metadata), with T = num_context + 1 frames:
        videos        {camera: FloatTensor (T, 3, H, W)}, normalised for the backbone
        tactile       FloatTensor (T, 2, 21, 21), gloves in GLOVES order, 0 off the glove
        tactile_mask  BoolTensor (2, 21, 21), True on the glove
        rel_t         array (T - 1,), number of frames between consecutive samples (0 where a short episode is padded)
        metadata      {'episode': str, 'indices': LongTensor (T,)}

    normalize=False keeps the frames in [0, 1] instead of normalizing them for the backbone.

    tactile_repr (one of TACTILE_REPRS) 'mano_vertex' or 'mano_uv' replaces the maps with the pressure of the taxels on
    the MANO hands, as ArcticDataset's contacts (contact: pressure > lam.tactile.CONTACT_THRESHOLD), using the assets
    at mano_uv_path:
        tactile       FloatTensor (T, 2, 778) on the vertices, or (T, 2, H_uv, W_uv) on the UV map
        tactile_mask  BoolTensor (2, 778) or (2, H_uv, W_uv), True under the taxels

    `split` is one of SPLITS, by default 'train' if train else 'val'. Only the split's episodes that are fully
    downloaded under root_folder are used. `train` also selects random sampling within each interval of an episode.
    """

    def __init__(self, root_folder, max_context_len=90, num_context=4, patch_size=14, img_size=224, aug=None,
                 backbone_name='dinov3', train=False, evaluate=False, cameras=CAMERAS, split=None, normalize=True,
                 tactile_repr='grid', mano_uv_path=None):
        super(EgoTouchDataset, self).__init__()
        assert not (aug and not train), "aug is only meant for the training set"
        assert max_context_len >= num_context, "max_context_len must be at least num_context"
        unknown = set(cameras) - set(CAMERAS)
        assert not unknown, f"unknown cameras {unknown}, expected some of {CAMERAS}"
        split = split or ('train' if train else 'val')
        assert split in SPLITS, f"unknown split {split!r}, expected one of {SPLITS}"
        assert tactile_repr in TACTILE_REPRS, f"unknown tactile_repr {tactile_repr!r}, expected one of {TACTILE_REPRS}"
        assert tactile_repr == 'grid' or mano_uv_path, f"tactile_repr={tactile_repr!r} needs mano_uv_path"

        self.max_context_len = max_context_len
        self.num_context = num_context
        self.frame_skip = max_context_len // num_context
        self.window = self.frame_skip * num_context
        self.patch_size = patch_size
        self.img_size = img_size
        self.train = train
        self.cameras = tuple(cameras)
        self.backbone_name = backbone_name
        self.normalize = normalize
        self.split = split
        self.episodes = self.list_episodes(root_folder)
        if not self.episodes:
            raise ValueError(f"No downloaded EgoTouch episode of split {split!r} under {root_folder}")

        # the last window of each episode when evaluating, else one interval per item
        self.segments = [1.0] if evaluate else [(i + 1) / NUM_SEGMENTS for i in range(NUM_SEGMENTS)]
        self.transform = self.build_transform(aug)

        self.tactile_repr = tactile_repr
        if tactile_repr != 'grid':
            assets = load_mano_assets(mano_uv_path)
            self.vertex_weights = torch.from_numpy(egotouch_vertex_weights(assets))  # (778, 151)
            self.uv = ManoUV(assets) if tactile_repr == 'mano_uv' else None

    def list_episodes(self, root_folder):
        """Complete episodes of the split, sorted by scene, task and recording time."""
        root = Path(root_folder).expanduser()
        split_file = root / SPLIT_FILE
        if not split_file.exists():
            raise FileNotFoundError(f"{split_file} is missing: download split.json from the zhouzhoujy/EgoTouch dataset")

        # entries are paths on the authors' machine, ending in <scene>/<task>/<stamp>.hdf5
        entries = json.loads(split_file.read_text())[self.split]
        episodes = sorted(root.joinpath(*Path(entry).with_suffix('').parts[-3:]) for entry in entries)
        required = [TACTILE_FILE] + [f'{camera}.mp4' for camera in self.cameras]
        return [episode for episode in episodes if all((episode / name).exists() for name in required)]  # downloads can be partial

    def build_transform(self, aug):
        if not self.normalize:
            normalize = transforms.Identity()  # frames stay in [0, 1], as LAM expects
        elif 'siglip' in self.backbone_name:
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not aug:
            return normalize
        return transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            normalize,
        ])

    def __len__(self):
        return len(self.episodes) * len(self.segments)

    def sample_indices(self, num_frames, segment):
        """num_context + 1 frame indices ending in the given interval of the episode, padded with frame 0 if it is short."""
        slack = num_frames - 1 - self.window
        if slack <= 0:
            last = num_frames - 1
        else:
            high = self.window + int(segment * slack)
            low = self.window + int((segment - 1 / NUM_SEGMENTS) * slack)
            last = random.randint(low, high) if self.train else high
        return np.maximum(last - self.frame_skip * np.arange(self.num_context, -1, -1), 0)

    def process_video(self, reader, indices):
        video = torch.from_numpy(reader.get_batch(indices.tolist()).asnumpy()).permute(0, 3, 1, 2)
        height, width = video.shape[-2:]
        size = [self.img_size, round(self.img_size * width / height)]
        video = TF.resize(video, size, antialias=True)
        video = TF.center_crop(video, [s // self.patch_size * self.patch_size for s in size])  # whole number of patches
        video = tv_tensors.Video(video.float() / 255.)
        return self.transform(video).as_subclass(torch.Tensor)

    def to_mano(self, tactile, tactile_mask):
        """(T, 2, 21, 21) maps and (2, 21, 21) mask -> the taxels on the MANO vertices or UV map, and where they are."""
        taxels = canonical_cells(tactile)[..., ALL_PRESSURE_CELLS]  # (T, 2, 151), both gloves on the left one's layout
        observed = canonical_cells(tactile_mask.float())[..., ALL_PRESSURE_CELLS]  # (2, 151)
        tactile = taxels @ self.vertex_weights.T  # (T, 2, 778)
        # a vertex is observed when the taxels it is made of are all on the glove
        mask = ((1 - observed) @ self.vertex_weights.T < 1e-6) & (self.vertex_weights.sum(1) > 0)
        if self.uv is not None:
            tactile, mask = self.uv(tactile), self.uv.coverage(mask)
        return tactile, mask

    def __getitem__(self, idx):
        episode = self.episodes[idx // len(self.segments)]
        segment = self.segments[idx % len(self.segments)]

        readers = {camera: VideoReader(str(episode / f'{camera}.mp4'), num_threads=1, ctx=cpu(0)) for camera in self.cameras}
        with np.load(episode / TACTILE_FILE) as grids:
            grid = np.stack([grids[f'{glove}_pressure_grid'] for glove in GLOVES], axis=1)  # (N, 2, 21, 21)

        # the streams share a timeline, but can end a few frames apart
        num_frames = min(len(grid), *(len(reader) for reader in readers.values()))
        indices = self.sample_indices(num_frames, segment)

        videos = {camera: self.process_video(reader, indices) for camera, reader in readers.items()}
        tactile_mask = torch.from_numpy(np.isfinite(grid[indices[0]]))
        tactile = torch.from_numpy(np.nan_to_num(grid[indices], nan=0.0)).float()
        if self.tactile_repr != 'grid':
            tactile, tactile_mask = self.to_mano(tactile, tactile_mask)
        rel_t = indices[1:] - indices[:-1]
        metadata = {'episode': str(episode), 'indices': torch.from_numpy(indices).long()}

        return videos, tactile, tactile_mask, rel_t, metadata
