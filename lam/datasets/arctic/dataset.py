"""ARCTIC (https://arctic.is.tue.mpg.de): egocentric RGB videos and hand-object contacts on the MANO hands.

<data_root> is ARCTIC's data/ folder:
    images/<subject>/<sequence>/0/<k:05d>.jpg   the egocentric camera (view 0), 30 Hz, 2800x2000, distorted
    splits_json/protocol_<p1|p2>.json           the splits train, val (s05) and test (s03), lists of '<subject>/<sequence>'
<contact_root> holds the contacts computed by lam/datasets/arctic/preprocess.py:
    <subject>/<sequence>.npz                    distance (N, 2, 778) of each MANO vertex to the object, per pose frame
The image k shows the pose frame k - image_offset (ARCTIC's ioi_offset, stored with the contacts).

The contacts are the vertices closer to the object than contact_threshold (EgoTac: 1 cm), on the MANO vertices or drawn
on the MANO UV map shared with EgoTouch (lam/hand_contact.py).
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from lam.hand_contact import HANDS, MANO_NUM_VERTICES, ManoUV

CAMERA = 'ego'
SPLITS = ('train', 'val', 'test')
CONTACT_REPRS = ('uv', 'vertex')
NUM_SEGMENTS = 10  # each sequence is divided into intervals so as not to load the same part of it every time


class ArcticDataset(Dataset):
    """Each item is the tuple (videos, contact, contact_mask, rel_t, metadata), with T = num_context + 1 frames:
        videos        {'ego': FloatTensor (T, 3, H, W)}, normalised for the backbone (in [0, 1] if not normalize)
        contact       FloatTensor (T, 2, 778) on the vertices, in {0, 1}, or (T, 2, H_uv, W_uv) on the UV map, in
                      [0, 1] (interpolated between the vertices), hands in HANDS order (left, right)
        contact_mask  BoolTensor (2, H_uv, W_uv) or (2, 778), True where the contact is known: the whole hand
        rel_t         array (T - 1,), number of frames between consecutive samples (0 where a short sequence is padded)
        metadata      {'episode': str, 'indices': LongTensor (T,)} the pose frames
    The arguments are those of EgoTouchDataset, plus contact_root, mano_uv_path, contact_threshold (m), contact_repr
    (one of CONTACT_REPRS) and protocol. Only the split's sequences with both images and contacts are used.
    """

    def __init__(self, root_folder, contact_root, mano_uv_path=None, max_context_len=90, num_context=4, patch_size=14,
                 img_size=224, aug=None, backbone_name='dinov3', train=False, evaluate=False, split=None,
                 normalize=True, contact_threshold=0.01, contact_repr='uv', protocol='p1'):
        super().__init__()
        assert not (aug and not train), "aug is only meant for the training set"
        assert max_context_len >= num_context, "max_context_len must be at least num_context"
        split = split or ('train' if train else 'val')
        assert split in SPLITS, f"unknown split {split!r}, expected one of {SPLITS}"
        assert contact_repr in CONTACT_REPRS, f"unknown contact_repr {contact_repr!r}, expected one of {CONTACT_REPRS}"
        assert contact_repr != 'uv' or mano_uv_path, "contact_repr='uv' needs mano_uv_path"

        self.root = Path(root_folder).expanduser()
        self.contact_root = Path(contact_root).expanduser()
        self.num_context = num_context
        self.frame_skip = max_context_len // num_context
        self.window = self.frame_skip * num_context
        self.patch_size = patch_size
        self.img_size = img_size
        self.train = train
        self.normalize = normalize
        self.backbone_name = backbone_name
        self.split = split
        self.protocol = protocol
        self.contact_threshold = contact_threshold
        self.contact_repr = contact_repr
        self.uv = ManoUV(mano_uv_path) if contact_repr == 'uv' else None

        self.sequences = self.list_sequences()
        if not self.sequences:
            raise ValueError(f"No ARCTIC sequence of split {split!r} with both images under {self.root} and contacts "
                             f"under {self.contact_root}")
        self.segments = [1.0] if evaluate else [(i + 1) / NUM_SEGMENTS for i in range(NUM_SEGMENTS)]
        self.transform = self.build_transform(aug)

    def list_sequences(self):
        """[(sequence, pose frames with an image)] for the split's sequences with both images and contacts."""
        split_file = self.root / 'splits_json' / f'protocol_{self.protocol}.json'
        sequences = []
        for sequence in json.loads(split_file.read_text())[self.split]:
            images, contacts = self.root / 'images' / sequence / '0', self.contact_root / f'{sequence}.npz'
            if not (images.is_dir() and contacts.exists()):
                continue
            with np.load(contacts) as data:
                num_frames, offset = data['distance'].shape[0], int(data['image_offset'])
            image_ids = {int(path.stem) for path in images.glob('*.jpg')}
            frames = np.array([k for k in range(num_frames) if k + offset in image_ids])
            if len(frames):
                sequences.append((sequence, frames, offset))
        return sequences

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
        return len(self.sequences) * len(self.segments)

    def sample_indices(self, num_frames, segment):
        """num_context + 1 positions ending in the given interval of the sequence, padded with 0 if it is short."""
        slack = num_frames - 1 - self.window
        if slack <= 0:
            last = num_frames - 1
        else:
            high = self.window + int(segment * slack)
            low = self.window + int((segment - 1 / NUM_SEGMENTS) * slack)
            last = random.randint(low, high) if self.train else high
        return np.maximum(last - self.frame_skip * np.arange(self.num_context, -1, -1), 0)

    def load_frame(self, path):
        """The image at img_size height, decoded at a reduced scale: ARCTIC's frames are 2800x2000 JPEGs."""
        with Image.open(path) as image:
            width, height = image.size
            size = (round(self.img_size * width / height), self.img_size)
            image.draft('RGB', size)  # JPEG decoding at 1/2, 1/4 or 1/8 of the size, never below `size`
            return torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1)

    def process_video(self, frames):
        video = torch.stack(frames)
        height, width = video.shape[-2:]
        size = [self.img_size, round(self.img_size * width / height)]
        video = TF.resize(video, size, antialias=True)
        video = TF.center_crop(video, [s // self.patch_size * self.patch_size for s in size])  # whole number of patches
        video = tv_tensors.Video(video.float() / 255.)
        return self.transform(video).as_subclass(torch.Tensor)

    def __getitem__(self, idx):
        sequence, frames, offset = self.sequences[idx // len(self.segments)]
        segment = self.segments[idx % len(self.segments)]
        positions = self.sample_indices(len(frames), segment)
        indices = frames[positions]  # pose frames

        image_folder = self.root / 'images' / sequence / '0'
        video = self.process_video([self.load_frame(image_folder / f'{k + offset:05d}.jpg') for k in indices])
        with np.load(self.contact_root / f'{sequence}.npz') as data:
            distance = torch.from_numpy(data['distance'][indices].astype(np.float32))  # (T, 2, 778)
        contact = (distance <= self.contact_threshold).float()
        mask = torch.ones(len(HANDS), MANO_NUM_VERTICES, dtype=torch.bool)
        if self.uv is not None:
            contact, mask = self.uv(contact), self.uv.coverage(mask)
        rel_t = indices[1:] - indices[:-1]
        metadata = {'episode': sequence, 'indices': torch.from_numpy(indices).long()}

        return {CAMERA: video}, contact, mask, rel_t, metadata
