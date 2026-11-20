"""Choisir algo.model.recon_change_threshold : distribution de |frame t+1 - frame t| moyen par patch sur EgoTouch.

Lit les batches d'entraînement exactement comme train.py (même config hydra, même échantillonnage des frames), découpe
chaque transition en patchs comme l'extracteur Identity, et affiche :
  - les quantiles du changement moyen par patch ;
  - pour plusieurs seuils, la fraction de patchs au-dessus (ce que la loss garde) et la part des transitions où
    presque toute l'image change (mouvement de caméra : la loss redevient alors la MSE sur toute l'image).

Usage :
    python calibrate_change_threshold.py [--config-name train_tactile_lam_motion] [--batches 50] [hydra overrides...]
"""

import argparse

import numpy as np
import torch
from einops import rearrange
from hydra import compose, initialize
from hydra.utils import instantiate
from tqdm import tqdm

THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-name', default='train_tactile_lam')
    parser.add_argument('--batches', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('overrides', nargs='*', help='overrides hydra, ex. data.data_module.data_path=/chemin')
    args = parser.parse_args()

    with initialize(version_base=None, config_path='conf'):
        config = compose(config_name=args.config_name, overrides=args.overrides)
    data_module = instantiate(config.data.data_module, num_workers=args.num_workers)
    data_module.setup('fit')
    camera = config.data.data_module.camera
    patch = config.algo.feature_extractor.model.patch_size
    ph, pw = (patch, patch) if isinstance(patch, int) else tuple(patch)

    changes = []
    for i, (videos, *_) in enumerate(tqdm(data_module.train_dataloader(), total=args.batches, desc='batches')):
        if i >= args.batches:
            break
        images = videos[camera].float()                                               # (B, T, C, H, W) dans [0, 1]
        patches = rearrange(images, 'b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=ph, p2=pw)
        changes.append((patches[:, 1:] - patches[:, :-1]).abs().mean(-1).flatten(0, 1))   # (N, P)
    change = torch.cat(changes).numpy()                                               # (transitions, patchs)

    print(f"\n{change.shape[0]} transitions, {change.shape[1]} patchs par frame")
    print("quantiles du changement moyen par patch :")
    for q in (10, 25, 50, 75, 90, 95, 99):
        print(f"  {q:3d} %  {np.percentile(change, q):.4f}")
    print(f"\n{'seuil':>7s} {'patchs gardés':>14s} {'transitions sans patch':>23s} {'transitions > 90 % gardés':>26s}")
    for threshold in THRESHOLDS:
        kept = change > threshold                                                    # (N, P)
        per_frame = kept.mean(1)
        print(f"{threshold:7.3f} {kept.mean():14.1%} {(per_frame == 0).mean():23.1%} {(per_frame > 0.9).mean():26.1%}")
    print("\nUn bon seuil garde une minorité de patchs (≈ 20-40 %) ; si beaucoup de transitions gardent > 90 % des"
          " patchs, c'est le mouvement de caméra : le masque n'y isole pas la main.")


if __name__ == '__main__':
    main()
