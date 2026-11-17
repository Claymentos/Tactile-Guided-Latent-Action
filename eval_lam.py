"""Évaluation d'un checkpoint TactileLAM : les actions latentes encodent-elles les mouvements des doigts ?

Les actions sont calculées sur un split d'EgoTouch (vidéo seule, comme à l'inférence), puis :

  1. sondes linéaires  ridge sur les actions gelées (continues z, ou code one-hot), validation croisée groupée par
                       épisode : R² sur le Δflexion de chaque doigt, le Δpression de chaque région, le mouvement global
                       de la caméra (corrélation de phase) ; AUROC des débuts / fins de contact par région.
                       Référence : la même sonde sur la différence moyenne des features (f_{t+1} - f_t).
  2. dictionnaire      pour chaque code : fréquence, Δflexion moyen par doigt, taux de débuts / fins de contact,
                       mouvement caméra ; η² (part de la variance expliquée par le code) et NMI code / événements.
  3. projection 2D     t-SNE (ou PCA) des actions continues, colorée par code, flexion, caméra, contact.
  4. interventions     chaque transition est décodée avec chacun des K codes : où l'image change (concentration,
                       corrélation avec le vrai mouvement), combien la flexion prédite change (contrôlabilité tactile),
                       et rang du code inféré parmi les K (le meilleur code = rang 1, hasard = (K+1)/2).
  5. attention         attention spatiale du token readout sur les patchs, par couche : concentration et corrélation
                       avec le mouvement réel entre t-1 et t.

Exécuter sur le run A (tactile_target=false) et le run B (true) et comparer les summary.json.

Usage :
    python eval_lam.py --checkpoint <run>/checkpoints/epoch=X-step=Y.ckpt \
        [--config <run>/checkpoints/model_config.yaml] [--data-path /chemin/egotouch/data] [--split val] \
        [--max-batches 100] [--intervention-batches 8] [--out eval/<nom>]
"""

import argparse
import warnings
import csv
import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from lam.tactile import ALL_BEND_CELLS, BEND_CELLS, CONTACT_THRESHOLD, FINGER_COLUMNS, PRESSURE_CELLS, canonical_cells

FINGERS = list(FINGER_COLUMNS)                       # pinky, ring, middle, index, thumb
REGIONS = FINGERS + ['palm']
GLOVES = ('left', 'right')
# position de chaque bend sensor dans les sorties du décodeur tactile (ALL_BEND_CELLS), regroupées par doigt
_BEND_SENSORS_OF_FINGER = [[ALL_BEND_CELLS.index(cell) for cell in cells] for cells in BEND_CELLS]


# ----------------------------------------------------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------------------------------------------------

def load_model(checkpoint, config_path, device):
    """Le TactileLAM du checkpoint Lightning, construit avec le model_config.yaml sauvegardé par train.py."""
    config_path = Path(config_path) if config_path else Path(checkpoint).parent / 'model_config.yaml'
    config = OmegaConf.load(config_path)
    model = instantiate(config.algo.model)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] clés manquantes : {missing}\n[load] clés inattendues : {unexpected}")
    return model.to(device).eval(), config


def build_loader(config, args):
    overrides = dict(batch_size=args.batch_size, num_workers=args.num_workers, val_split=args.split)
    if args.data_path:
        overrides['data_path'] = args.data_path
    data_module = instantiate(config.data.data_module, **overrides)
    assert data_module.dataset_args['tactile_repr'] == 'grid', "eval_lam lit les capteurs de flexion : tactile_repr=grid"
    data_module.setup('validate')
    return data_module.val_dataloader(), config.data.data_module.camera


# ----------------------------------------------------------------------------------------------------------------------
# Modèle : encodage, décodage, attention du readout
# ----------------------------------------------------------------------------------------------------------------------

def encode(model, images):
    """Comme TactileLAM.forward (registres jamais masqués) : features, actions continues et quantifiées, codes."""
    features = model.feature_extractor(images)                                        # (B, T, P, D)
    B, T, P, _ = features.shape
    R = model.num_register_tokens
    registers = repeat(model.register_tokens, '1 n d -> b t n d', b=B, t=T) if R else None
    attention_mask = torch.zeros((B, T, R + P), dtype=torch.bool, device=images.device)
    logits, readout = model.encode(features, registers=registers, attention_mask=attention_mask)
    readout = readout[:, 1:]                                                          # (B, T-1, D)
    tokens, indices, _ = model.vq.quantize(readout)
    register_readouts = logits[:, 1:, :R] if R else None
    return dict(features=features, readout=readout, tokens=tokens, indices=indices,
                register_readouts=register_readouts, attention_mask=attention_mask)


def decode(model, enc, images, tokens):
    """Frames t + actions -> reconstruction de t+1 : la vidéo (pixels) si l'extracteur en produit une, et les features."""
    recon_features = model.decode(enc['features'][:, :-1], actions=tokens, registers=enc['register_readouts'],
                                  attention_mask=enc['attention_mask'][:, :-1])
    recon_features = recon_features[:, :, model.num_register_tokens:]
    _, recon_video = model.feature_extractor.compute_loss(recon_features, images[:, 1:], enc['features'][:, 1:])
    return recon_video, recon_features


def predict_bend(model, enc, tactile, tokens):
    """Δflexion prédit par le décodeur tactile pour chaque transition, par gant et par doigt : (B, T-1, 2, 5)."""
    B, T = tactile.shape[:2]
    context = enc['features'][:, :-1].flatten(0, 1)
    _, _, delta_bend = model.predict_tactile(context, tactile[:, :-1].flatten(0, 1), tokens.flatten(0, 1).to(context.dtype))
    per_finger = torch.stack([delta_bend[..., s].mean(-1) for s in _BEND_SENSORS_OF_FINGER], dim=-1)
    return per_finger.float().unflatten(0, (B, T - 1))


class ReadoutAttention:
    """Attention spatiale du readout (dernier token de chaque frame) sur les patchs, recalculée dans un pre-hook de
    chaque nn.MultiheadAttention spatiale de l'encodeur (le modèle l'appelle avec need_weights=False)."""

    def __init__(self, encoder, num_registers, num_patches):
        self.R, self.P = num_registers, num_patches
        self.maps = {}
        self.enabled = False
        self.handles = [layer.spatial_attention.register_forward_pre_hook(partial(self.hook, i), with_kwargs=True)
                        for i, layer in enumerate(encoder.transformer.layers)]

    def hook(self, layer, module, args, kwargs):
        if not self.enabled:
            return
        x = args[0]                                                                   # (B*T, R + P + 1, D)
        D, H = module.embed_dim, module.num_heads
        w, b = module.in_proj_weight, module.in_proj_bias
        q = F.linear(x[:, -1:], w[:D], None if b is None else b[:D])                   # requête du readout
        k = F.linear(x, w[D:2 * D], None if b is None else b[D:2 * D])
        q = q.unflatten(-1, (H, D // H)).transpose(1, 2)                              # (BT, H, 1, d)
        k = k.unflatten(-1, (H, D // H)).transpose(1, 2)                              # (BT, H, L, d)
        scores = (q @ k.transpose(-1, -2)).squeeze(2).float() / math.sqrt(D // H)     # (BT, H, L)
        mask = kwargs.get('key_padding_mask')
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :].bool(), float('-inf'))
        self.maps[layer] = scores.softmax(-1).mean(1)[:, self.R:self.R + self.P]      # (BT, P), moyenne des têtes

    def remove(self):
        for handle in self.handles:
            handle.remove()


# ----------------------------------------------------------------------------------------------------------------------
# Quantités observées
# ----------------------------------------------------------------------------------------------------------------------

def tactile_quantities(tactile):
    """(B, T, 2, 21, 21) -> par transition : Δflexion par doigt (B, T-1, 2, 5), Δpression par région (B, T-1, 2, 6),
    débuts et fins de contact par région (B, T-1, 2, 6), bool."""
    cells = canonical_cells(tactile.float())                                          # (B, T, 2, 441)
    bend = torch.stack([cells[..., c].mean(-1) for c in BEND_CELLS], dim=-1)          # (B, T, 2, 5)
    pressure = torch.stack([cells[..., c].mean(-1) for c in PRESSURE_CELLS], dim=-1)  # (B, T, 2, 6)
    contact = [cells[..., c] > CONTACT_THRESHOLD for c in PRESSURE_CELLS]
    onset = torch.stack([(~c[:, :-1] & c[:, 1:]).any(-1) for c in contact], dim=-1)
    offset = torch.stack([(c[:, :-1] & ~c[:, 1:]).any(-1) for c in contact], dim=-1)
    return bend[:, 1:] - bend[:, :-1], pressure[:, 1:] - pressure[:, :-1], onset, offset


def camera_shift(images):
    """Translation globale (dy, dx) en pixels entre frames successives, par corrélation de phase : (B, T-1, 2)."""
    gray = images.float().mean(2)                                                     # (B, T, H, W)
    H, W = gray.shape[-2:]
    window = torch.outer(torch.hann_window(H, device=gray.device), torch.hann_window(W, device=gray.device))
    spectrum = torch.fft.rfft2((gray - gray.mean((-2, -1), keepdim=True)) * window)
    cross = spectrum[:, 1:] * spectrum[:, :-1].conj()
    correlation = torch.fft.irfft2(cross / cross.abs().clamp(min=1e-8), s=(H, W))
    peak = correlation.flatten(-2).argmax(-1)
    dy, dx = peak // W, peak % W
    dy = torch.where(dy > H // 2, dy - H, dy)
    dx = torch.where(dx > W // 2, dx - W, dx)
    return torch.stack([dy, dx], dim=-1).float()


def patch_map(pixels, patch_size):
    """(..., C, H, W) -> (..., H/p * W/p) : moyenne par patch (sur les canaux et les pixels)."""
    x = pixels.float().mean(-3)
    lead = x.shape[:-2]
    x = F.avg_pool2d(x.reshape(-1, 1, *x.shape[-2:]), patch_size)
    return x.reshape(*lead, -1)


def concentration(maps):
    """(N, P) cartes positives -> entropie normalisée (1 = uniforme, 0 = un seul patch) et part des 10 % meilleurs patchs."""
    p = maps / maps.sum(-1, keepdim=True).clamp(min=1e-12)
    entropy = -(p * p.clamp(min=1e-12).log()).sum(-1) / math.log(p.shape[-1])
    top = max(1, round(0.1 * p.shape[-1]))
    return entropy, p.topk(top, dim=-1).values.sum(-1)


def rank_of_inferred(errors, indices):
    """(K, ...) erreurs de chaque code, (...) code inféré -> rang du code inféré (1 = meilleur), ex aequo au rang moyen."""
    own = errors.gather(0, indices[None])
    return 1 + (errors < own).sum(0) + 0.5 * ((errors == own).sum(0) - 1)


def pearson(a, b):
    """Corrélation ligne à ligne de deux (N, P)."""
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-12)


# ----------------------------------------------------------------------------------------------------------------------
# Sondes et statistiques
# ----------------------------------------------------------------------------------------------------------------------

def episode_folds(groups, k, seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    fold_of = {g: i % k for i, g in enumerate(unique)}
    return np.array([fold_of[g] for g in groups])


def ridge_oof(X, Y, folds, alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)):
    """Prédictions hors-fold d'une ridge (X et Y standardisés sur le fold d'entraînement) ; alpha choisi par le R²
    moyen hors-fold (légèrement optimiste, identique pour toutes les représentations comparées)."""
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    best, best_score = None, -np.inf
    for alpha in alphas:
        pred = np.zeros_like(Y)
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-8
            my = Y[tr].mean(0)
            Xtr, Xte = (X[tr] - mx) / sx, (X[te] - mx) / sx
            A = Xtr.T @ Xtr + alpha * np.eye(X.shape[1])
            pred[te] = Xte @ np.linalg.solve(A, Xtr.T @ (Y[tr] - my)) + my
        score = r2(Y, pred).mean()
        if score > best_score:
            best, best_score = pred, score
    return best


def r2(Y, pred):
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0).clip(min=1e-12)


def auroc(labels, scores):
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos < 10 or n_neg < 10:
        return float('nan')
    order = scores.argsort(kind='mergesort')
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # rangs moyens pour les ex aequo
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def nmi(a, b):
    """Information mutuelle normalisée par sqrt(H(a) H(b))."""
    joint = np.unique(np.stack([a, b], 1), axis=0, return_counts=True)[1]
    pj = joint / joint.sum()
    ha, hb = entropy(a), entropy(b)
    mi = ha + hb - float(-(pj * np.log(pj)).sum())
    return mi / math.sqrt(ha * hb) if ha > 0 and hb > 0 else 0.0


def eta_squared(values, codes):
    """Part de la variance de chaque colonne de values expliquée par le code (rapport de corrélation)."""
    total = ((values - values.mean(0)) ** 2).sum(0)
    between = sum((codes == k).sum() * (values[codes == k].mean(0) - values.mean(0)) ** 2 for k in np.unique(codes))
    return between / total.clip(min=1e-12)


def ternary(x, tau):
    return np.where(x > tau, 2, np.where(x < -tau, 0, 1))


# ----------------------------------------------------------------------------------------------------------------------
# Collecte
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def collect(model, loader, camera, args, device, patch_size):
    K = model.vq.codebook_size
    codebook = model.vq.dequantize(torch.arange(K, device=device)[None])[0]          # (K, D)
    attention = None
    records = {k: [] for k in ('z', 'code', 'feat_diff', 'dbend', 'dpressure', 'onset', 'offset', 'shift',
                               'pixel_change', 'episode')}
    inter = {k: [] for k in ('energy_entropy', 'energy_top10', 'energy_motion_corr', 'visual_std', 'real_change',
                             'tactile_std', 'pixel_rank', 'bend_rank', 'bend_err_inferred', 'bend_err_best')}
    attn = {}
    visuals = []
    autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16 and device.type == 'cuda')

    for batch_idx, (videos, tactile, _, _, metadata) in enumerate(tqdm(loader, desc='évaluation')):
        if args.max_batches and batch_idx >= args.max_batches:
            break
        images, tactile = videos[camera].to(device), tactile.to(device)
        B, T = images.shape[:2]
        run_interventions = batch_idx < args.intervention_batches

        with autocast:
            if attention is None:
                num_patches = model.feature_extractor(images[:1, :1]).shape[2]
                attention = ReadoutAttention(model.encoder, model.num_register_tokens, num_patches)
            attention.enabled = run_interventions
            enc = encode(model, images)
            attention.enabled = False

        # ---- actions et quantités observées, pour chaque transition
        dbend, dpressure, onset, offset = tactile_quantities(tactile)
        motion = patch_map((images[:, 1:] - images[:, :-1]).abs(), patch_size)       # (B, T-1, P)
        records['z'].append(enc['readout'].float().flatten(0, 1).cpu())
        records['code'].append(enc['indices'].flatten().cpu())
        records['feat_diff'].append((enc['features'][:, 1:] - enc['features'][:, :-1]).float().mean(2).flatten(0, 1).cpu())
        records['dbend'].append(dbend.flatten(0, 1).flatten(1).cpu())                # (N, 10) gant x doigt
        records['dpressure'].append(dpressure.flatten(0, 1).flatten(1).cpu())        # (N, 12)
        records['onset'].append(onset.flatten(0, 1).flatten(1).cpu())
        records['offset'].append(offset.flatten(0, 1).flatten(1).cpu())
        records['shift'].append(camera_shift(images).flatten(0, 1).cpu())
        records['pixel_change'].append(motion.mean(-1).flatten().cpu())
        records['episode'] += [e for e in metadata['episode'] for _ in range(T - 1)]

        if not run_interventions:
            continue

        # ---- attention du readout : frame t (1..T-1) -> patchs, comparée au mouvement t-1 -> t
        for layer, maps in attention.maps.items():
            maps = maps.unflatten(0, (B, T))[:, 1:].flatten(0, 1)                    # (N, P)
            ent, top = concentration(maps)
            corr = pearson(maps, motion.flatten(0, 1))
            for name, value in (('entropy', ent), ('top10', top), ('motion_corr', corr)):
                attn.setdefault(layer, {}).setdefault(name, []).append(value.cpu())

        # ---- interventions : décodage avec chacun des K codes
        target = images[:, 1:].float()
        s1 = s2 = 0.0
        pixel_err, bend_pred = [], []
        with autocast:
            for k in range(K):
                tokens = codebook[k].expand(B, T - 1, -1).to(enc['tokens'].dtype)
                video, _ = decode(model, enc, images, tokens)
                video = video.float()
                s1, s2 = s1 + video, s2 + video ** 2
                pixel_err.append(((video - target) ** 2).mean((-3, -2, -1)))         # (B, T-1)
                bend_pred.append(predict_bend(model, enc, tactile, tokens))          # (B, T-1, 2, 5)
                if batch_idx == 0:
                    visuals.append(video[0, 0].cpu())
            inferred_video, _ = decode(model, enc, images, enc['tokens'])
        pixel_err, bend_pred = torch.stack(pixel_err), torch.stack(bend_pred)        # (K, B, T-1), (K, B, T-1, 2, 5)
        variance = (s2 / K - (s1 / K) ** 2).clamp(min=0)                             # (B, T-1, 3, H, W)
        energy = patch_map(variance, patch_size).flatten(0, 1)                       # (N, P)
        ent, top = concentration(energy)
        inter['energy_entropy'].append(ent.cpu())
        inter['energy_top10'].append(top.cpu())
        inter['energy_motion_corr'].append(pearson(energy, motion.flatten(0, 1)).cpu())
        inter['visual_std'].append(variance.mean((-3, -2, -1)).sqrt().flatten().cpu())
        inter['real_change'].append((target - images[:, :-1].float()).abs().mean((-3, -2, -1)).flatten().cpu())
        inter['tactile_std'].append(bend_pred.std(0).flatten(0, 1).cpu())             # (N, 2, 5)
        idx = enc['indices']
        inter['pixel_rank'].append(rank_of_inferred(pixel_err, idx).flatten().float().cpu())
        bend_err = (bend_pred - dbend[None]).abs().mean((-2, -1))                     # (K, B, T-1)
        inter['bend_rank'].append(rank_of_inferred(bend_err, idx).flatten().float().cpu())
        inter['bend_err_inferred'].append(bend_err.gather(0, idx[None])[0].flatten().cpu())
        inter['bend_err_best'].append(bend_err.min(0).values.flatten().cpu())

        if batch_idx == 0:
            n = min(args.num_visuals, B * (T - 1))
            last4 = torch.stack([attention.maps[l] for l in range(max(0, len(attention.maps) - 4), len(attention.maps))]).mean(0)
            last4 = last4.unflatten(0, (B, T))[:, 1:].flatten(0, 1)
            visuals = dict(codes=visuals, frames_t=images[:, :-1].flatten(0, 1)[:n].float().cpu(),
                           frames_t1=target.flatten(0, 1)[:n].cpu(), inferred=inferred_video.float().flatten(0, 1)[:n].cpu(),
                           motion=motion.flatten(0, 1)[:n].cpu(), energy=energy[:n].cpu(), attention=last4[:n].float().cpu(),
                           indices=idx.flatten()[:n].cpu())

    if attention is not None:
        attention.remove()
    data = {k: (v if k == 'episode' else torch.cat(v).numpy()) for k, v in records.items()}
    data['codebook_size'] = K
    inter = {k: torch.cat(v).numpy() for k, v in inter.items() if v}
    if inter:
        inter['K'] = K
    attn = {l: {k: torch.cat(v).numpy() for k, v in d.items()} for l, d in attn.items()}
    return data, inter, attn, visuals


# ----------------------------------------------------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------------------------------------------------

def probes(data, num_folds):
    folds = episode_folds(np.array(data['episode']), num_folds)
    K = data['codebook_size']
    inputs = {'z (action continue)': data['z'], 'code (one-hot)': np.eye(K)[data['code']],
              'f_t+1 - f_t (référence)': data['feat_diff']}
    shift = data['shift']
    targets = {'dbend': data['dbend'], 'dpressure': data['dpressure'],
               'camera': np.concatenate([shift, np.linalg.norm(shift, axis=1, keepdims=True),
                                         data['pixel_change'][:, None]], 1)}
    results = {}
    for name, X in inputs.items():
        res = {}
        bend_r2 = r2(targets['dbend'], ridge_oof(X, targets['dbend'], folds)).reshape(2, 5)
        res['dbend_r2_par_doigt'] = dict(zip(FINGERS, bend_r2.mean(0).round(4).tolist()))
        res['dbend_r2'] = float(bend_r2.mean())
        pressure_r2 = r2(targets['dpressure'], ridge_oof(X, targets['dpressure'], folds)).reshape(2, 6)
        res['dpressure_r2_par_region'] = dict(zip(REGIONS, pressure_r2.mean(0).round(4).tolist()))
        res['dpressure_r2'] = float(pressure_r2.mean())
        camera_r2 = r2(targets['camera'], ridge_oof(X, targets['camera'], folds))
        res['camera_r2'] = dict(zip(['dy', 'dx', '|shift|', '|Δpixel|'], camera_r2.round(4).tolist()))
        for event in ('onset', 'offset'):
            labels = data[event]
            scores = ridge_oof(X, labels.astype(np.float64), folds)
            aucs = np.array([auroc(labels[:, j], scores[:, j]) for j in range(labels.shape[1])]).reshape(2, 6)
            res[f'contact_{event}_auroc_par_region'] = dict(zip(REGIONS, np.nanmean(aucs, 0).round(4).tolist()))
            res[f'contact_{event}_auroc'] = float(np.nanmean(aucs))
        res['ratio_doigts_sur_camera'] = res['dbend_r2'] / max(float(camera_r2[2]), 1e-3)
        results[name] = res
    return results


def code_dictionary(data, out_dir):
    codes = data['code']
    K = data['codebook_size']
    dbend = data['dbend'].reshape(-1, 2, 5).mean(1)                                   # moyenne des deux gants
    shift = np.linalg.norm(data['shift'], axis=1)
    finger_onset = data['onset'].reshape(-1, 2, 6)[..., :5].any((1, 2))
    finger_offset = data['offset'].reshape(-1, 2, 6)[..., :5].any((1, 2))
    rows = []
    for k in range(K):
        sel = codes == k
        row = dict(code=k, n=int(sel.sum()), freq=float(sel.mean()))
        if sel.any():
            row.update({f'dbend_{f}': float(dbend[sel, i].mean()) for i, f in enumerate(FINGERS)})
            row.update(abs_dbend=float(np.abs(dbend[sel]).mean()), onset_doigts=float(finger_onset[sel].mean()),
                       offset_doigts=float(finger_offset[sel].mean()), onset_paume=float(data['onset'].reshape(-1, 2, 6)[sel, :, 5].any(1).mean()),
                       camera_shift=float(shift[sel].mean()), pixel_change=float(data['pixel_change'][sel].mean()))
        rows.append(row)
    with open(out_dir / 'code_dictionary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(max(rows, key=len)))
        writer.writeheader()
        writer.writerows(rows)

    # η² : part de la variance de chaque quantité expliquée par le code
    eta = dict(dbend_par_doigt=dict(zip(FINGERS, eta_squared(dbend, codes).round(4).tolist())),
               dbend_total=float(eta_squared(dbend.sum(1, keepdims=True), codes)[0]),
               camera_shift=float(eta_squared(shift[:, None], codes)[0]),
               pixel_change=float(eta_squared(data['pixel_change'][:, None], codes)[0]))
    # NMI entre le code et des événements discrets
    tau = 0.5 * dbend.std(0)
    events = {'flexion_totale (ferme/immobile/ouvre)': ternary(dbend.sum(1), 0.5 * dbend.sum(1).std()),
              'contact (rien/début/fin/les deux)': finger_onset.astype(int) + 2 * finger_offset.astype(int),
              'camera (quartiles)': np.digitize(shift, np.quantile(shift, [0.25, 0.5, 0.75]))}
    for i, finger in enumerate(FINGERS):
        events[f'flexion_{finger}'] = ternary(dbend[:, i], tau[i])
    nmis = {name: round(nmi(codes, labels), 4) for name, labels in events.items()}
    return rows, dict(eta_squared=eta, nmi=nmis, entropie_codes_bits=entropy(codes) / math.log(2),
                      codes_utilises=int(len(np.unique(codes))), codebook_size=K)


def interventions_summary(inter, data_dbend_std):
    if not inter:
        return {}
    K_rank = lambda r: dict(moyen=float(r.mean()), top1=float((r == 1).mean()), hasard=(inter['K'] + 1) / 2)
    tactile_std = inter['tactile_std'].mean(0)                                        # (2, 5)
    return dict(
        image_entropie_energie=float(inter['energy_entropy'].mean()),
        image_part_top10_patchs=float(inter['energy_top10'].mean()),
        image_corr_energie_mouvement=float(np.nanmean(inter['energy_motion_corr'])),
        controlabilite_visuelle=float(inter['visual_std'].mean() / max(inter['real_change'].mean(), 1e-8)),
        controlabilite_tactile=float(tactile_std.mean() / max(data_dbend_std.mean(), 1e-8)),
        controlabilite_tactile_par_doigt=dict(zip(FINGERS, (tactile_std.mean(0) / np.maximum(data_dbend_std.reshape(2, 5).mean(0), 1e-8)).round(4).tolist())),
        rang_code_infere_pixels=K_rank(inter['pixel_rank']),
        rang_code_infere_flexion=K_rank(inter['bend_rank']),
        erreur_flexion_code_infere=float(inter['bend_err_inferred'].mean()),
        erreur_flexion_meilleur_code=float(inter['bend_err_best'].mean()),
    )


def attention_summary(attn):
    return {f'couche_{l}': {k: float(np.nanmean(v)) for k, v in d.items()} for l, d in sorted(attn.items())}


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------

def figures(data, visuals, attn, out_dir, grid, seed=0):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figures] matplotlib absent : pas de figures")
        return

    # ---- projection 2D des actions continues
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(data['z']), size=min(5000, len(data['z'])), replace=False)
    z = data['z'][sel]
    try:
        from sklearn.manifold import TSNE
        emb, method = TSNE(n_components=2, init='pca', random_state=seed, perplexity=30).fit_transform(z), 't-SNE'
    except ImportError:
        zc = torch.from_numpy(z - z.mean(0))
        emb, method = (zc @ torch.pca_lowrank(zc, q=2)[2]).numpy(), 'PCA'
    dbend = data['dbend'][sel].reshape(-1, 2, 5).mean(1).sum(1)
    shift = np.linalg.norm(data['shift'][sel], axis=1)
    contact = (data['onset'][sel].any(1).astype(int) + 2 * data['offset'][sel].any(1).astype(int))
    panels = [('code', data['code'][sel], 'tab10'), ('Δflexion totale', dbend, 'coolwarm'),
              ('|mouvement caméra| (px)', shift, 'viridis'), ('contact : 0 rien, 1 début, 2 fin, 3 les deux', contact, 'tab10')]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (title, color, cmap) in zip(axes, panels):
        if title.startswith('Δ'):
            lim = np.quantile(np.abs(color), 0.98)
            s = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap=cmap, s=3, vmin=-lim, vmax=lim)
        else:
            s = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap=cmap, s=3)
        fig.colorbar(s, ax=ax)
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f'{method} des actions latentes continues')
    fig.savefig(out_dir / 'embedding.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    if not visuals:
        return
    h, w = grid
    to_img = lambda x: x.permute(1, 2, 0).clamp(0, 1).numpy()

    # ---- interventions et attention, une ligne par transition
    n = len(visuals['frames_t'])
    fig, axes = plt.subplots(n, 5, figsize=(15, 2.6 * n), squeeze=False)
    titles = ['frame t', 'frame t+1', 'reconstruite (code inféré)', '|Δ| réel', 'énergie des interventions', ]
    for i in range(n):
        size = visuals['frames_t'][i].shape[-2:]
        up = lambda m: F.interpolate(m.reshape(1, 1, h, w), size=size, mode='nearest')[0, 0].numpy()
        axes[i, 0].imshow(to_img(visuals['frames_t'][i]))
        axes[i, 1].imshow(to_img(visuals['frames_t1'][i]))
        axes[i, 2].imshow(to_img(visuals['inferred'][i]))
        axes[i, 2].set_ylabel(f"code {int(visuals['indices'][i])}")
        axes[i, 3].imshow(up(visuals['motion'][i]), cmap='magma')
        axes[i, 4].imshow(to_img(visuals['frames_t1'][i]))
        axes[i, 4].imshow(up(visuals['energy'][i]), cmap='magma', alpha=0.6)
        for j in range(5):
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(titles[j])
    fig.savefig(out_dir / 'interventions.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(n, 2, figsize=(6, 2.6 * n), squeeze=False)
    for i in range(n):
        size = visuals['frames_t1'][i].shape[-2:]
        up = F.interpolate(visuals['attention'][i].reshape(1, 1, h, w), size=size, mode='nearest')[0, 0].numpy()
        axes[i, 0].imshow(to_img(visuals['frames_t1'][i]))
        axes[i, 1].imshow(to_img(visuals['frames_t1'][i]))
        axes[i, 1].imshow(up, cmap='magma', alpha=0.6)
        for j in range(2):
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
    axes[0, 0].set_title('frame t')
    axes[0, 1].set_title('attention du readout (4 dernières couches)')
    fig.savefig(out_dir / 'readout_attention.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

    # ---- la même transition décodée avec chacun des codes
    codes = visuals['codes']
    fig, axes = plt.subplots(1, len(codes) + 1, figsize=(2.4 * (len(codes) + 1), 2.6))
    axes[0].imshow(to_img(visuals['frames_t'][0]))
    axes[0].set_title('frame t')
    for k, img in enumerate(codes):
        axes[k + 1].imshow(to_img(img))
        axes[k + 1].set_title(f'code {k}' + (' (inféré)' if k == int(visuals['indices'][0]) else ''))
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.savefig(out_dir / 'decoded_per_code.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

    # ---- attention par couche
    if attn:
        layers = sorted(attn)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        ax[0].plot(layers, [np.nanmean(attn[l]['entropy']) for l in layers], marker='o')
        ax[0].set_title('entropie normalisée de l\'attention du readout')
        ax[1].plot(layers, [np.nanmean(attn[l]['motion_corr']) for l in layers], marker='o')
        ax[1].set_title('corrélation attention / mouvement réel')
        for a in ax:
            a.set_xlabel('couche')
        fig.savefig(out_dir / 'attention_layers.png', dpi=110, bbox_inches='tight')
        plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------

def print_summary(summary):
    p = summary['sondes']
    print("\n================ SONDES LINÉAIRES (R² hors-fold, par épisode) ================")
    print(f"{'représentation':28s} {'Δflexion':>9s} {'Δpression':>10s} {'|caméra|':>9s} {'début c.':>9s} {'fin c.':>8s} {'doigts/cam':>11s}")
    for name, r in p.items():
        print(f"{name:28s} {r['dbend_r2']:9.3f} {r['dpressure_r2']:10.3f} {r['camera_r2']['|shift|']:9.3f} "
              f"{r['contact_onset_auroc']:9.3f} {r['contact_offset_auroc']:8.3f} {r['ratio_doigts_sur_camera']:11.2f}")
    for name, r in p.items():
        print(f"  Δflexion par doigt, {name}: {r['dbend_r2_par_doigt']}")
    d = summary['dictionnaire']
    print("\n================ CODES ================")
    print(f"codes utilisés : {d['codes_utilises']}/{d['codebook_size']}, entropie {d['entropie_codes_bits']:.2f} bits")
    print(f"η² : {d['eta_squared']}")
    print(f"NMI : {d['nmi']}")
    if summary['interventions']:
        print("\n================ INTERVENTIONS ================")
        for k, v in summary['interventions'].items():
            print(f"{k:36s} {v}")
    if summary['attention']:
        print("\n================ ATTENTION DU READOUT ================")
        for k, v in summary['attention'].items():
            print(f"{k:10s} " + "  ".join(f"{n}={x:.3f}" for n, x in v.items()))


def main():
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=None, help='model_config.yaml (par défaut : à côté du checkpoint)')
    parser.add_argument('--data-path', default=None, help='remplace data.data_module.data_path')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test_seen', 'test_unseen'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--max-batches', type=int, default=0, help='0 : tout le split')
    parser.add_argument('--intervention-batches', type=int, default=8, help='batches décodés avec chacun des K codes')
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--num-visuals', type=int, default=8)
    parser.add_argument('--no-bf16', dest='bf16', action='store_false')
    parser.add_argument('--out', default=None, help='dossier de sortie (par défaut eval/<nom du checkpoint>)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.checkpoint, args.config, device)
    loader, camera = build_loader(config, args)
    patch_size = config.data.data_module.patch_size
    out_dir = Path(args.out or Path('eval') / Path(args.checkpoint).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, inter, attn, visuals = collect(model, loader, camera, args, device, patch_size)
    print(f"{len(data['code'])} transitions, {len(set(data['episode']))} épisodes")

    _, dict_summary = code_dictionary(data, out_dir)
    summary = dict(checkpoint=str(args.checkpoint), split=args.split, tactile_target=bool(model.tactile_target),
                   transitions=len(data['code']), episodes=len(set(data['episode'])),
                   copie={'dbend_abs_moyen': float(np.abs(data['dbend']).mean()),
                          'dbend_std_par_doigt': dict(zip(FINGERS, data['dbend'].reshape(-1, 2, 5).std(0).mean(0).round(4).tolist()))},
                   sondes=probes(data, args.num_folds), dictionnaire=dict_summary,
                   interventions=interventions_summary(inter, data['dbend'].std(0)), attention=attention_summary(attn))
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    grid = tuple(s // patch_size for s in visuals['frames_t'].shape[-2:]) if visuals else None
    figures(data, visuals, attn, out_dir, grid)
    print_summary(summary)
    print(f"\nrésultats dans {out_dir}/ : summary.json, code_dictionary.csv, embedding.png, interventions.png, "
          f"readout_attention.png, decoded_per_code.png, attention_layers.png")


if __name__ == '__main__':
    main()
