"""Computes the hand-object contacts of ARCTIC on the MANO vertices, as EgoTac does (data_processing/parse_data_arctic.py).

For every frame of a sequence, both MANO hands and the articulated object are rebuilt from ARCTIC's fits, and each hand
vertex gets its distance to the object. A vertex is in contact below a threshold (EgoTac: 1 cm), applied by the dataset,
so that the threshold can change without running this again.

Difference with EgoTac: the distance is to the surface of the object, not to its nearest vertex. ARCTIC's object meshes
have edges up to 3 cm long (the box: 14 mm on median), so with a 1 cm threshold the nearest vertex misses contacts in
the middle of the faces. The surface is approximated by points sampled every `--spacing` mm. `--method vertex` gives
EgoTac's distances. Distances do not depend on the frame they are computed in: unlike EgoTac, there is no need to move
to the egocentric camera's frame.

Needs ARCTIC's data (raw_seqs/, meta/, splits_json/) and the MANO model, with smplx + chumpy + trimesh + scipy, e.g.
ARCTIC's environment:
    python lam/datasets/arctic/preprocess.py --data_root ~/arctic/unpack/arctic_data/data \
        --mano_dir ~/arctic/unpack/body_models/mano --out ~/arctic/processed/contacts
Writes <out>/<subject>/<sequence>.npz: distance (N, 2, 778) float16 in meters, hands in lam.hand_contact.HANDS order,
for the N pose frames; image k of the egocentric camera is pose frame k - image_offset.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HANDS = ('left', 'right')
ARTICULATION_AXIS = np.array([0.0, 0.0, -1.0])  # ARCTIC's common/object_tensors.py
MANO_BATCH = 512


def mano_vertices(layer, params, num_frames):
    """(N, 778, 3) in meters, in ARCTIC's world frame."""
    import torch

    out = []
    for start in range(0, num_frames, MANO_BATCH):
        frames = slice(start, min(start + MANO_BATCH, num_frames))
        n = frames.stop - frames.start
        with torch.no_grad():
            output = layer(betas=torch.as_tensor(params['shape'], dtype=torch.float32).reshape(1, 10).expand(n, 10),
                           global_orient=torch.as_tensor(params['rot'][frames], dtype=torch.float32),
                           hand_pose=torch.as_tensor(params['pose'][frames], dtype=torch.float32),
                           transl=torch.as_tensor(params['trans'][frames], dtype=torch.float32))
        out.append(output.vertices.numpy())
    return np.concatenate(out)


class ObjectSurface:
    """The two rigid parts of an ARCTIC object (top: articulated, bottom), as point clouds in the object's frame, in m."""

    def __init__(self, object_dir, method, spacing_mm, seed=0):
        import trimesh

        mesh = trimesh.load(str(object_dir / 'mesh.obj'), process=False)
        vertices, faces = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
        top = np.asarray(json.loads((object_dir / 'parts.json').read_text()), dtype=bool)
        rng = np.random.default_rng(seed)
        self.trees = {}
        for part, vertex_mask in (('top', top), ('bottom', ~top)):
            points = [vertices[vertex_mask]]
            if method == 'surface':
                part_faces = faces[vertex_mask[faces].mean(1) > 0.5]  # a face on the seam goes to its majority
                corners = vertices[part_faces]
                areas = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
                count = int(np.ceil(areas.sum() / spacing_mm ** 2))
                chosen = rng.choice(len(part_faces), size=count, p=areas / areas.sum())
                r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
                bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
                points.append(np.einsum('nk,nkd->nd', bary, corners[chosen]))
            self.trees[part] = cKDTree(np.concatenate(points) / 1000.0)  # mm -> m

    def distances(self, points, params):
        """points (P, 3) in the world frame, params ARCTIC's 7 object parameters of the frame -> (P,) distances in m."""
        angle, rotation, translation = params[0], params[1:4], params[4:7] / 1000.0
        to_object = Rotation.from_rotvec(rotation).inv()
        local = to_object.apply(points - translation)  # bottom part frame
        top_local = Rotation.from_rotvec(ARTICULATION_AXIS * angle).inv().apply(local)
        return np.minimum(self.trees['bottom'].query(local)[0], self.trees['top'].query(top_local)[0])


def process(sequence, data_root, layers, surfaces, misc, out_root, method, spacing_mm):
    subject, name = sequence.split('/')
    raw = data_root / 'raw_seqs' / subject
    mano = np.load(raw / f'{name}.mano.npy', allow_pickle=True).item()
    object_params = np.load(raw / f'{name}.object.npy', allow_pickle=True).astype(np.float64)
    num_frames = len(object_params)
    object_name = name.split('_')[0]
    if object_name not in surfaces:
        surfaces[object_name] = ObjectSurface(data_root / 'meta' / 'object_vtemplates' / object_name, method, spacing_mm)
    surface = surfaces[object_name]

    hands = np.stack([mano_vertices(layers[hand], mano[hand], num_frames) for hand in HANDS], axis=1)  # (N, 2, 778, 3)
    distance = np.empty(hands.shape[:3], dtype=np.float32)
    for frame in range(num_frames):
        distance[frame] = surface.distances(hands[frame].reshape(-1, 3), object_params[frame]).reshape(len(HANDS), -1)

    out = out_root / subject / f'{name}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, distance=distance.astype(np.float16), image_offset=int(misc[subject]['ioi_offset']),
                        object_name=object_name, method=method, spacing_mm=spacing_mm)
    return out, distance


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', type=Path, required=True, help="ARCTIC's data/ folder: raw_seqs/, meta/, splits_json/")
    parser.add_argument('--mano_dir', type=Path, required=True, help='folder of MANO_RIGHT.pkl and MANO_LEFT.pkl')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', default='p1', help='splits_json/protocol_<protocol>.json')
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--subject', default=None, help='only this subject, e.g. s01')
    parser.add_argument('--only_with_images', action='store_true', help='skip the sequences whose egocentric images are missing')
    parser.add_argument('--method', choices=('surface', 'vertex'), default='surface', help='vertex: EgoTac\'s distances')
    parser.add_argument('--spacing', type=float, default=2.0, help='mm between the points sampled on the object')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    import smplx

    data_root = args.data_root.expanduser()
    out_root = args.out.expanduser()
    protocol = json.loads((data_root / 'splits_json' / f'protocol_{args.protocol}.json').read_text())
    misc = json.loads((data_root / 'meta' / 'misc.json').read_text())
    sequences = [seq for split in args.splits.split(',') for seq in protocol[split.strip()]]
    if args.subject:
        sequences = [seq for seq in sequences if seq.startswith(f'{args.subject}/')]
    if args.only_with_images:
        sequences = [seq for seq in sequences if (data_root / 'images' / seq / '0').is_dir()]
    # ARCTIC's MANO layers (common/body_models.py: build_mano_aa)
    layers = {hand: smplx.MANO(model_path=str(args.mano_dir.expanduser()), is_rhand=hand == 'right', use_pca=False,
                               flat_hand_mean=False) for hand in HANDS}
    surfaces = {}

    print(f"{len(sequences)} sequences")
    for i, sequence in enumerate(sequences):
        if (out_root / f'{sequence}.npz').exists() and not args.overwrite:
            continue
        start = time.perf_counter()
        try:
            out, distance = process(sequence, data_root, layers, surfaces, misc, out_root, args.method, args.spacing)
        except FileNotFoundError as error:
            print(f"[{i + 1}/{len(sequences)}] {sequence}: skipped, {error}")
            continue
        contact = (distance <= 0.01).any(-1).mean(0)
        print(f"[{i + 1}/{len(sequences)}] {sequence}: {len(distance)} frames, frames with contact (1 cm) "
              f"left {contact[0]:.0%} right {contact[1]:.0%}, {time.perf_counter() - start:.1f} s -> {out}")


if __name__ == '__main__':
    main()
