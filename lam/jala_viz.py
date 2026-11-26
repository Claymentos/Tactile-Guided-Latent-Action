"""Visualisations du test de cycle-consistance de JALA (lam.jala.JALA.cycle_consistency) sur EgoDex.

Une frame de vidéo par transition t, une ligne par paire de clips (A, B), quatre panneaux :
  1. A_t : squelette des mains, flèches du mouvement réel (vert), décodé de z_A (bleu), décodé de z_cycle (rouge) ;
     un bon cycle a des flèches rouges proches des bleues, et toutes deux proches des vertes
  2. B_t : mouvement réel de B (vert) et mouvement de A transféré sur les mains de B (bleu, décodé de z_A)
  3. B_t+1 réel, features DINO projetées en RGB par PCA
  4. B'_t+1 = ForwardProbe(B_t, z_A), même PCA : ce que « faire l'action de A dans la scène B » donne en features
Les flèches sont agrandies (arrow_scale) : à 30 fps et δ = 4 frames, la main ne bouge que de quelques pixels.
Les mouvements sont dans le repère caméra de la première frame de chaque clip : le transfert A -> B suppose ces
repères comparables (vidéos égocentrées, tête à peu près dans la même pose).
"""

import cv2
import numpy as np
import torch

INTRINSICS = (736.6339, 736.6339, 960.0, 540.0)          # fx, fy, cx, cy d'EgoDex en 1920 x 1080
SOURCE_SIZE = (1080, 1920)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
FINGER_BASES = (1, 5, 9, 13, 17)                       # pouce, index, majeur, annulaire, auriculaire (4 points chacun)
ARROW_POINTS = (0, 4, 8, 12, 16, 20)                   # poignet et bouts des doigts, par main
COLORS = dict(real=(40, 200, 40), original=(40, 120, 255), cycle=(240, 50, 50), skeleton=(235, 235, 235))


def project(points, cam0_to_camt, image_size=(224, 384), crop=7):
    """Points 3D (..., 3) dans le repère caméra de la frame 0 -> pixels (..., 2) dans l'image t (redimensionnée à
    image_size[0] de haut et recadrée de `crop` pixels de chaque côté, comme EgoDexDataset.image_transform), et masque
    de validité (devant la caméra et dans l'image). Même convention que lam.datasets.egodex.dataset.get_pixels_from_kp.
    cam0_to_camt : (4, 4), ou (..., 4, 4) avec les mêmes dimensions de tête que points sans celle des points."""
    rotation, translation = cam0_to_camt[..., :3, :3].unsqueeze(-3), cam0_to_camt[..., :3, 3].unsqueeze(-2)
    p = (rotation @ points.unsqueeze(-1)).squeeze(-1) + translation
    fx, fy, cx, cy = INTRINSICS
    z = p[..., 2]
    u = fx * p[..., 0] / z.clamp(min=1e-6) + cx
    v = fy * p[..., 1] / z.clamp(min=1e-6) + cy
    scale = image_size[0] / SOURCE_SIZE[0]
    u, v = u * scale - crop, v * scale
    valid = (z > 1e-3) & (u >= 0) & (u < image_size[1]) & (v >= 0) & (v < image_size[0])
    return torch.stack([u, v], -1), valid


def to_uint8(frames):
    """Frames normalisées ImageNet (..., 3, H, W) -> uint8 (..., H, W, 3)."""
    mean, std = IMAGENET_MEAN.to(frames.device)[:, None, None], IMAGENET_STD.to(frames.device)[:, None, None]
    x = (frames.float() * std + mean).clamp(0, 1)
    return (x.movedim(-3, -1) * 255).round().to(torch.uint8).cpu().numpy()


def _point(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_skeleton(image, pixels, valid):
    for hand in range(2):
        o = 21 * hand
        for base in FINGER_BASES:
            chain = [o] + [o + base + j for j in range(4)]
            for a, b in zip(chain[:-1], chain[1:]):
                if valid[a] and valid[b]:
                    cv2.line(image, _point(pixels[a]), _point(pixels[b]), COLORS['skeleton'], 1, cv2.LINE_AA)
    for i in range(42):
        if valid[i]:
            cv2.circle(image, _point(pixels[i]), 2, COLORS['skeleton'], -1, cv2.LINE_AA)


def draw_arrows(image, start, end, valid, color):
    for hand in range(2):
        for k in ARROW_POINTS:
            i = 21 * hand + k
            if valid[i]:
                cv2.arrowedLine(image, _point(start[i]), _point(end[i]), color, 2, cv2.LINE_AA, tipLength=0.3)


def label(image, lines):
    y = 14
    for text, color in lines:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (2, y - h - 3), (6 + w, y + 3), (0, 0, 0), -1)
        cv2.putText(image, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += h + 7
    return image


def hand_panel(frame, points, cam0_to_camt, motions, arrow_scale, title):
    """frame uint8 (H, W, 3) ; points (42, 3) mètres, repère caméra 0 ; motions : liste (déplacement (42, 3) cm,
    couleur, légende) dessinés en flèches depuis les points."""
    image = np.ascontiguousarray(frame.copy())
    pixels, valid = project(points, cam0_to_camt, frame.shape[:2])
    draw_skeleton(image, pixels, valid)
    for motion, color, _ in motions:
        end, end_valid = project(points + arrow_scale * motion / 100.0, cam0_to_camt, frame.shape[:2])
        draw_arrows(image, pixels, end, valid & end_valid, color)
    return label(image, [(title, (255, 255, 255))] + [(name, color) for _, color, name in motions])


def pca_rgb(features, grid, size):
    """features (N, P, F) -> N images uint8 (size, 3), même projection PCA (3 composantes) et même échelle pour les N."""
    x = features.float().flatten(0, 1)
    x = x - x.mean(0)
    with torch.autocast(x.device.type, enabled=False):          # QR non implémentée en bf16 (validation bf16-mixed)
        _, _, v = torch.pca_lowrank(x, q=3, center=False)
        y = x @ v[:, :3]
    low, high = torch.quantile(y, 0.01, dim=0), torch.quantile(y, 0.99, dim=0)
    y = ((y - low) / (high - low).clamp(min=1e-6)).clamp(0, 1)
    y = (y.unflatten(0, features.shape[:2]).unflatten(1, grid) * 255).round().to(torch.uint8).cpu().numpy()
    return [cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST) for img in y]


@torch.no_grad()
def cycle_video(images_a, images_b, actions_a, actions_b, cam_a, cam_b, cycle, grid, num_samples=4, arrow_scale=3.0):
    """Assemble la vidéo du cycle. images_* (B, T, 3, H, W) normalisées ; actions_* (B, T-1, 88, 3) ; cam_* (B, T, 4, 4)
    metadata['cam0_to_camt'] ; cycle : sortie de JALA.cycle_consistency(images_a, images_b, actions_a).
    Renvoie un tableau uint8 (T-1, 3, num_samples * H, 4 * W) pour wandb.Video."""
    from lam.jala import NUM_HAND_KEYPOINTS, hand_motion_target
    n = min(num_samples, images_a.shape[0])
    S = actions_a.shape[1]
    frames_a, frames_b = to_uint8(images_a[:n]), to_uint8(images_b[:n])                      # (n, T, H, W, 3)
    size = frames_a.shape[2:4]
    points_a = actions_a[:n, :, :NUM_HAND_KEYPOINTS].float().cpu()                           # (n, S, 42, 3) mètres
    points_b = actions_b[:n, :, :NUM_HAND_KEYPOINTS].float().cpu()
    real_b = hand_motion_target(actions_b[:n].float()).cpu()
    get = lambda key: cycle[key][:n].float().cpu()
    real_a, from_za, from_zc = get('hands_target'), get('hands_original'), get('hands_cycle')
    cam_a, cam_b = cam_a[:n].float().cpu(), cam_b[:n].float().cpu()
    scale = f"(fleches x{arrow_scale:g})"   # cv2.putText : ASCII seulement

    video = []
    for t in range(S):
        rows = []
        for i in range(n):
            pca = pca_rgb(torch.stack([cycle['features_b'][i, 1:][t], cycle['transferred'][i, t]]), grid, size)
            err = lambda m: f"{float((m[i, t] - real_a[i, t]).abs().mean()):.2f} cm"
            panels = [
                hand_panel(frames_a[i, t], points_a[i, t], cam_a[i, t],
                           [(real_a[i, t], COLORS['real'], 'A reel'),
                            (from_za[i, t], COLORS['original'], f'z_A  {err(from_za)}'),
                            (from_zc[i, t], COLORS['cycle'], f'z_cycle  {err(from_zc)}')], arrow_scale, f'A  t={t} {scale}'),
                hand_panel(frames_b[i, t], points_b[i, t], cam_b[i, t],
                           [(real_b[i, t], COLORS['real'], 'B reel'),
                            (from_za[i, t], COLORS['original'], 'action de A sur B')], arrow_scale, f'B  t={t}'),
                label(np.ascontiguousarray(pca[0]), [('B t+1 reel (PCA DINO)', (255, 255, 255))]),
                label(np.ascontiguousarray(pca[1]), [("B' t+1 = B_t + z_A", (255, 255, 255))]),
            ]
            rows.append(np.concatenate(panels, axis=1))
        video.append(np.concatenate(rows, axis=0))
    return np.stack(video).transpose(0, 3, 1, 2)                                             # (S, 3, H, W)


def heatmap_overlay(frame, attention, grid, alpha=0.55):
    """frame uint8 (H, W, 3), attention (P,) -> carte de chaleur (normalisée par son max) superposée à la frame."""
    a = attention.float().reshape(grid).cpu().numpy()
    a = a / max(a.max(), 1e-8)
    a = cv2.resize(a, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    color = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]          # BGR -> RGB
    weight = (alpha * a[..., None]) ** 0.7
    return np.ascontiguousarray((frame * (1 - weight) + color * weight).astype(np.uint8))


def _mass(attention, mask):
    return float((attention * mask).sum())


@torch.no_grad()
def finger_attention_video(images, actions, cameras, attention, hands_probe, grid, num_samples=4, arrow_scale=3.0):
    """Où regardent les actions latentes ? Une frame de vidéo par transition t, une ligne par clip, quatre panneaux :
      1. attention moyenne des K queries de LAP sur (v_t, v_t+1), en carte de chaleur, avec le squelette des mains ;
         légende : part de l'attention sur les doigts, les bouts des doigts, le poignet, et le lift (masse / surface)
      2. attention des queries de la tête gauche ; 3. de la tête droite (spécialisation par main)
      4. articulation des doigts (mouvement relatif au poignet) : réelle (vert) et décodée de z par la sonde (bleu)
    images (B, T, 3, H, W) ; actions (B, T-1, 88, 3) ; cameras (B, T, 4, 4) ; attention (B, T-1, K, P) ;
    hands_probe (B, T-1, 42, 3) cm. Renvoie uint8 (T-1, 3, n * H, 4 * W)."""
    from lam.jala import FINGERS, NUM_HAND_KEYPOINTS, WRIST, hand_motion_target, keypoint_patch_masks
    n = min(num_samples, images.shape[0])
    S, K = attention.shape[1], attention.shape[2]
    frames = to_uint8(images[:n])
    H, W = frames.shape[2:4]
    patch = H // grid[0]
    points = actions[:n, :, :NUM_HAND_KEYPOINTS].float().cpu()
    cameras, attention = cameras[:n, :S].float().cpu(), attention[:n].float().cpu()
    real, probe = hand_motion_target(actions[:n].float()).cpu(), hands_probe[:n].float().cpu()
    wrist_of = torch.tensor([WRIST[0] if i < 21 else WRIST[1] for i in range(42)])
    relative = lambda m: m - m[..., wrist_of, :]                                   # mouvement relatif au poignet
    pixels, valid = project(points, cameras)                                       # (n, S, 42, 2)
    masks = keypoint_patch_masks(pixels, valid, grid, patch)

    video = []
    for t in range(S):
        rows = []
        for i in range(n):
            frame, att = frames[i, t], attention[i, t]                             # (K, P)
            mean = att.mean(0)
            m = {k: v[i, t] for k, v in masks.items()}
            stats = []
            for region in ('fingers', 'fingertips', 'wrist'):
                area = float(m[region].mean())
                mass = _mass(mean, m[region])
                stats.append((f"{region} {mass:.2f} (x{mass / area:.1f})" if area > 0 else f"{region} -", (255, 255, 255)))
            panels = []
            for title, a in ((f'attention LAP t={t}', mean), ('queries main gauche', att[:K // 2].mean(0)),
                             ('queries main droite', att[K // 2:].mean(0))):
                image = heatmap_overlay(frame, a, grid)
                draw_skeleton(image, pixels[i, t], valid[i, t])
                extra = stats if title.startswith('attention') else [
                    (f"G {_mass(a, m['left']):.2f}  D {_mass(a, m['right']):.2f}", (255, 255, 255))]
                panels.append(label(image, [(title, (255, 255, 0))] + extra))
            fingers = list(FINGERS)
            real_rel, probe_rel = relative(real[i, t]), relative(probe[i, t])
            err = float((probe_rel[fingers] - real_rel[fingers]).abs().mean())
            zero = float(real_rel[fingers].abs().mean())
            image = np.ascontiguousarray(frame.copy())
            draw_skeleton(image, pixels[i, t], valid[i, t])
            for motion, color in ((real_rel, COLORS['real']), (probe_rel, COLORS['original'])):
                end, end_valid = project(points[i, t] + arrow_scale * motion / 100.0, cameras[i, t], (H, W))
                ok = valid[i, t] & end_valid
                for k in fingers:
                    if ok[k]:
                        cv2.arrowedLine(image, _point(pixels[i, t, k]), _point(end[k]), color, 1, cv2.LINE_AA, tipLength=0.35)
            panels.append(label(image, [(f'doigts / poignet (x{arrow_scale:g})', (255, 255, 0)),
                                        ('reel', COLORS['real']),
                                        (f'sonde z  {err:.2f} cm (fige {zero:.2f})', COLORS['original'])]))
            rows.append(np.concatenate(panels, axis=1))
        video.append(np.concatenate(rows, axis=0))
    return np.stack(video).transpose(0, 3, 1, 2)
