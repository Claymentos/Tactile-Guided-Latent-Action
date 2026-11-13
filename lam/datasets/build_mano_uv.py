"""Builds the MANO assets of lam/hand_contact.py: the right hand at rest, its faces and the UV map.

Needs the MANO model (MANO_RIGHT.pkl, from https://mano.is.tue.mpg.de) and smplx + chumpy, e.g. ARCTIC's environment:
    python lam/datasets/build_mano_uv.py --mano_dir ~/arctic/unpack/body_models/mano --out ~/mano_assets/mano_uv.npz
The output holds MANO's geometry: keep it out of the repository, MANO's license does not allow redistributing it.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def load_hand_contact():
    """lam/hand_contact.py without importing the lam package, whose dependencies the MANO environment may miss."""
    path = Path(__file__).resolve().parents[1] / 'hand_contact.py'
    spec = importlib.util.spec_from_file_location('hand_contact', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mano_layer(mano_dir, right):
    import smplx
    return smplx.MANO(model_path=str(Path(mano_dir).expanduser()), is_rhand=right, use_pca=False, flat_hand_mean=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mano_dir', type=Path, required=True, help='folder of MANO_RIGHT.pkl and MANO_LEFT.pkl')
    parser.add_argument('--out', type=Path, required=True, help='the .npz to write')
    parser.add_argument('--size', type=int, default=128, help='height of the UV map in texels')
    args = parser.parse_args()

    hand_contact = load_hand_contact()
    right, left = mano_layer(args.mano_dir, True), mano_layer(args.mano_dir, False)
    template = right.v_template.numpy()  # the flat hand: v_template is the pose 0 of flat_hand_mean
    faces = right.faces.astype(np.int64)
    # the datasets rely on the left hand being the mirror image of the right one, vertex by vertex
    mirror = np.abs(left.v_template.numpy() * np.array([-1, 1, 1]) - template).max()
    assert mirror < 1e-6, f"the left MANO hand is not the mirror image of the right one ({mirror:.2e} m)"

    uv = hand_contact.build_mano_uv(template, faces, size=args.size)
    visible = np.zeros(hand_contact.MANO_NUM_VERTICES, dtype=bool)
    visible[uv['texel_vertices'][uv['texel_weights'] > 0.05]] = True

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    joints = (right.J_regressor.numpy() @ template).astype(np.float32)  # (16, 3) MANO's joints at rest
    np.savez_compressed(args.out.expanduser(), template=template.astype(np.float32), faces=faces, joints=joints,
                        lbs_weights=right.lbs_weights.numpy().astype(np.float32), **uv)
    print(f"wrote {args.out}: UV map {uv['texel_mask'].shape}, {visible.sum()}/{len(visible)} vertices visible")


if __name__ == '__main__':
    main()
