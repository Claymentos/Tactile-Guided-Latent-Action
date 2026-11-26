"""Évaluation d'un checkpoint JALA (lam.lightning.jala.JALATraining) sur un split EgoDex, avec les métriques et les
visualisations de la validation (val/*, val/focus/*, val/cycle/*, vidéos dans un run W&B hors ligne).

    python eval_jala.py --ckpt checkpoints/jala_egodex/epoch=14-step=100000.ckpt --data ~/egodex
    python eval_jala.py --ckpt ... --tasks seen --train-tasks part1_tasks.txt

Tâches : l'indice de tâche est l'ordre alphabétique des dossiers du split d'entraînement (part1). Sur un autre split,
les indices ne correspondent pas. --tasks none (défaut) évalue sans token de tâche ; --tasks seen garde seulement les
clips des tâches vues à l'entraînement, avec leur indice d'entraînement (--train-tasks : sortie de
`ls /lustre/fsmisc/dataset/ml-egodex/part1`, une tâche par ligne).
"""

import argparse
import os

import torch
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from lam.datasets.egodex.dataset import EgoDexDataset
from lam.lightning.jala import JALATraining


class TaskRemap(Dataset):
    """Remplace metadata['task_id'] par l'indice de la tâche dans la liste d'entraînement."""

    def __init__(self, dataset, train_ids):
        self.dataset, self.train_ids = dataset, train_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        item[-1]['task_id'] = self.train_ids[item[-1]['task']]
        return item


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--config', default=None, help="model_config.yaml du run (défaut : à côté du checkpoint)")
    parser.add_argument('--data', default=os.path.expanduser('~/egodex'), help="dossier contenant test/ (ou part1/)")
    parser.add_argument('--split', default='test', choices=('test', 'part1'))
    parser.add_argument('--tasks', default='none', choices=('none', 'seen'))
    parser.add_argument('--train-tasks', default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-batches', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--out', default='runs/eval_jala')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)

    config = OmegaConf.load(args.config or os.path.join(os.path.dirname(args.ckpt), 'model_config.yaml'))
    model = instantiate(config.algo.model)
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = {k[len('model.'):]: v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith('feature_extractor.')]       # DINOv3 rechargé depuis ses poids
    assert not missing and not unexpected, f"checkpoint incompatible : manquants {missing}, inattendus {unexpected}"
    print(f"checkpoint {args.ckpt} : pas {checkpoint.get('global_step')}")

    data = config.data.data_module
    dataset = EgoDexDataset(args.data, max_context_len=data.window_size, num_context=data.get('num_context', 4),
                            patch_size=data.patch_size, backbone_name=data.get('backbone_name', 'dinov3'),
                            train=args.split == 'part1', return_heatmaps=False,
                            return_motion_chunks=config.algo.model.get('action_objective', 'regression') == 'mcp')
    use_task = False
    if args.tasks == 'seen':
        assert args.train_tasks, "--tasks seen : --train-tasks <liste des tâches de part1> nécessaire"
        names = [line.strip() for line in open(args.train_tasks) if line.strip()]
        train_ids = {name: i for i, name in enumerate(sorted(names))}
        keep = [i for i, (clip, _) in enumerate(dataset.idx_to_data)
                if dataset.all_tasks[dataset.file_tasks[clip]] in train_ids]
        seen = sorted({dataset.all_tasks[dataset.file_tasks[dataset.idx_to_data[i][0]]] for i in keep})
        print(f"{len(seen)} tâches du split {args.split} vues à l'entraînement : {seen}")
        assert keep, "aucune tâche commune"
        dataset, use_task = TaskRemap(Subset(dataset, keep), train_ids), True
    print(f"{len(dataset)} clips, {min(args.num_batches, len(dataset) // args.batch_size)} batches de {args.batch_size}"
          f" ({'avec' if use_task else 'sans'} token de tâche)")

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))
    module = JALATraining(model, use_task=use_task, **{k: v for k, v in checkpoint['hyper_parameters'].items()
                                                       if k in ('cycle_consistency_batches', 'num_logged_samples',
                                                                'arrow_scale', 'finger_focus_batches')})
    module.cycle_consistency_batches = module.finger_focus_batches = args.num_batches
    os.makedirs(args.out, exist_ok=True)
    logger = WandbLogger(name=f"eval_{os.path.basename(args.ckpt)}_{args.split}_{args.tasks}", save_dir=args.out,
                         mode='offline', project='LAM')
    trainer = Trainer(logger=logger, precision='bf16-mixed', limit_val_batches=args.num_batches, devices=1)
    metrics = trainer.validate(module, dataloaders=loader, verbose=False)[0]

    def show(title, keys):
        print(f"\n{title}")
        for key, reference in keys:
            if key in metrics:
                ref = f"   (référence {reference} : {metrics[reference]:.3f})" if reference in metrics else ''
                print(f"  {key:42s} {metrics[key]:8.3f}{ref}")

    show("Actions (cm, plus bas = mieux)", [
        ('val/action_mae', 'val/action_mae_zero'), ('val/probe_mae', 'val/action_mae_zero'),
        ('val/probe_finger_mae', 'val/finger_motion_zero'), ('val/probe_wrist_mae', 'val/wrist_motion_zero'),
        ('val/tokenizer_mae', None), ('val/mcp_accuracy', None)])
    show("Alignement et effondrement", [
        ('val/loss_align', 'val/loss_align_shuffled'), ('val/align_specificity', None), ('val/z_std', None),
        ('val/z_temporal_std_ratio', None), ('val/z_h_cosine', None)])
    show("Attention de LAP sur les mains (lift : 1 = uniforme)", [
        ('val/focus/hand_lift', None), ('val/focus/fingers_lift', None), ('val/focus/fingertips_lift', None),
        ('val/focus/wrist_lift', None), ('val/focus/fingers_mass', None),
        ('val/focus/left_queries_own_hand', None), ('val/focus/right_queries_own_hand', None)])
    show("Cycle A -> B -> A (hausse_relative : 0 = transfert parfait, 1 = pas mieux qu'une action au hasard)", [
        ('val/cycle/hand_error_original', None), ('val/cycle/hand_error_cycle', None),
        ('val/cycle/hand_error_random', None), ('val/cycle/hand_hausse_relative', None),
        ('val/cycle/feature_hausse_relative', None), ('val/cycle/z_cosine_cycle', 'val/cycle/z_cosine_random')])
    print(f"\nToutes les métriques et les vidéos : {logger.experiment.dir}  (wandb sync pour les voir en ligne ; images et "
          f"gif dans media/)")


if __name__ == '__main__':
    main()
