"""A contact representation shared by the datasets: values on the 778 vertices of MANO, drawn on a MANO UV map.

Every dataset expresses its touch on the vertices of the MANO hand: ARCTIC the hand-object contacts computed on the
fitted meshes, EgoTouch its glove taxels placed on the hand. The left and right MANO hands share their vertex indexing
(one is the mirror image of the other), so a vertex is the same point of the hand for both, like EgoTouch's canonical
glove layout.

The UV map lays out the surface of the MANO hand at rest (flat, fingers together) in two panels side by side, drawn as
in the palm view of EgoTouch's maps (fingers up, pinky on the left, thumb on the right):
    left panel   the palm side
    right panel  the back of the hand, mirrored so that both panels share the pinky-thumb axis
Each texel is a barycentric combination of the 3 vertices of the triangle drawn there. Values on the vertices are
interpolated over the triangles, so the map is smooth. Freely available MANO UV maps (HTML, Handy) are under license;
this one is computed from MANO itself (build_mano_uv). Any UV map of MANO would fit: only `texel_vertices` and
`texel_weights` define it.

The assets (template, faces, UV lookup) are built once with `python lam/datasets/build_mano_uv.py`, which needs the
MANO model (smplx + chumpy). Using them only needs numpy and torch.
"""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

MANO_NUM_VERTICES = 778
HANDS = ('left', 'right')

# MANO's template, in meters: fingers along -x (the wrist is at +x), thumb towards +z, the palm faces -y
_FINGER_AXIS, _THICKNESS_AXIS, _WIDTH_AXIS = 0, 1, 2


def build_mano_uv(template: np.ndarray, faces: np.ndarray, size: int = 128) -> dict:
    """Rasterizes the MANO hand at rest into the two panels of the UV map.

    arguments: template (778, 3) the right MANO hand at rest (flat_hand_mean), faces (F, 3), size the height in texels
    returns: {'texel_vertices' (H, W, 3) int64, 'texel_weights' (H, W, 3) float32, 'texel_mask' (H, W) bool,
              'face_palm' (F,) bool, 'vertex_uv' (778, 2, 2) float32 the (column, row) of each vertex in each panel}
    A texel outside the hand has weights 0. 777 of the 778 vertices show up at size 128 (746 at 64): the others are
    hidden in folds, between the fingers.
    """
    template = np.asarray(template, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    length = np.ptp(template[:, _FINGER_AXIS])
    width = np.ptp(template[:, _WIDTH_AXIS])
    height, panel_width = size, int(round(size * width / length))

    normals = np.cross(template[faces[:, 1]] - template[faces[:, 0]], template[faces[:, 2]] - template[faces[:, 0]])
    face_palm = normals[:, _THICKNESS_AXIS] < 0
    row = (template[:, _FINGER_AXIS] - template[:, _FINGER_AXIS].min()) / length * (height - 2) + 1  # 1 texel margin
    z = template[:, _WIDTH_AXIS]

    texel_vertices = np.zeros((height, 2 * panel_width, 3), dtype=np.int64)
    texel_weights = np.zeros((height, 2 * panel_width, 3), dtype=np.float32)
    vertex_uv = np.zeros((MANO_NUM_VERTICES, 2, 2), dtype=np.float32)
    for panel, (selected, side) in enumerate([(face_palm, 1.0), (~face_palm, -1.0)]):
        # the back is mirrored: the pinky stays on the left
        column = (z - z.min() if side > 0 else z.max() - z) / width * (panel_width - 2) + 1
        depth = side * template[:, _THICKNESS_AXIS]  # the viewer of the palm stands at -y, the one of the back at +y
        vertex_uv[:, panel] = np.stack([column + panel * panel_width, row], axis=-1)
        z_buffer = np.full((height, panel_width), np.inf)
        for face in np.flatnonzero(selected):
            corners = faces[face]
            x, y = column[corners], row[corners]
            det = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(det) < 1e-12:
                continue
            x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())), panel_width - 1)
            y0, y1 = max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())), height - 1)
            ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            px, py = xs + 0.5, ys + 0.5  # texel centers
            w0 = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / det
            w1 = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / det
            w2 = 1 - w0 - w1
            weights = np.stack([w0, w1, w2], axis=-1)
            inside = (weights >= -1e-9).all(-1)
            d = weights @ depth[corners]
            closer = inside & (d < z_buffer[ys, xs])
            ys, xs = ys[closer], xs[closer]
            z_buffer[ys, xs] = d[closer]
            texel_vertices[ys, xs + panel * panel_width] = corners
            texel_weights[ys, xs + panel * panel_width] = np.clip(weights[closer], 0, 1)

    return dict(texel_vertices=texel_vertices, texel_weights=texel_weights, texel_mask=texel_weights.sum(-1) > 0,
                face_palm=face_palm, vertex_uv=vertex_uv)


def load_mano_assets(path) -> dict:
    """The arrays written by lam/datasets/build_mano_uv.py: template, faces and the UV lookup of build_mano_uv."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: build it with `python lam/datasets/build_mano_uv.py --out {path}`")
    with np.load(path) as assets:
        return {key: assets[key] for key in assets.files}


class ManoUV(nn.Module):
    """Draws values on the MANO vertices on the UV map: (..., 778) -> (..., H, W). Differentiable, runs on any device.

    `texel_mask` (H, W) is True on the hand. `coverage(vertex_mask)` gives the texels a dataset observes, e.g. the ones
    under EgoTouch's taxels.
    """

    def __init__(self, assets):
        super().__init__()
        assets = load_mano_assets(assets) if isinstance(assets, (str, Path)) else assets
        texel_vertices = torch.as_tensor(assets['texel_vertices'], dtype=torch.long)
        self.height, self.width = texel_vertices.shape[:2]
        self.register_buffer('texel_vertices', texel_vertices.flatten(0, 1), persistent=False)  # (H*W, 3)
        self.register_buffer('texel_weights', torch.as_tensor(assets['texel_weights']).flatten(0, 1), persistent=False)
        self.register_buffer('texel_mask', torch.as_tensor(assets['texel_mask']), persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        texels = values[..., self.texel_vertices]  # (..., H*W, 3)
        texels = (texels * self.texel_weights.to(texels.dtype)).sum(-1)
        return texels.unflatten(-1, (self.height, self.width))

    def coverage(self, vertex_mask: Tensor) -> Tensor:
        """(..., 778) bool -> (..., H, W) bool: the texels whose 3 vertices are all observed."""
        observed = vertex_mask[..., self.texel_vertices].all(-1).unflatten(-1, (self.height, self.width))
        return observed & self.texel_mask


# ----------------------------------------------------------------------------------------------------------------------
# EgoTouch's gloves on MANO
#
# As EgoTac does for the OpenTouch glove (data_processing/common/tactile_to_mano_mapper_patch.py), each region of taxels
# is a regular grid laid on a patch of the MANO hand, and each vertex of the patch takes the bilinear interpolation of
# the 4 nearest taxels. EgoTac's patches are planar quadrilaterals between 4 vertices; on the curved fingers they only
# catch a few vertices, so the patches are defined here on the anatomy of MANO, on its palm side:
#     fingers  the vertices skinned to the middle and distal phalanges (thumb: proximal and distal), rows from the
#              fingertip to the joint at the base of the middle phalanx, columns across the finger
#     palm     the vertices skinned to the wrist and to the thumb's metacarpal (thenar eminence), rows from the fingers
#              to the wrist
# Columns go from the pinky side to the thumb side, as in EgoTouch's canonical layout (lam.tactile). EgoTouch does not
# document where its pads lie on the hand: these extents are an assumption, set by EGOTOUCH_REGIONS.
# ----------------------------------------------------------------------------------------------------------------------

# in lam.tactile.PRESSURE_CELLS order: (MANO joints whose bones the region covers, fingertip vertex, joint at the base)
# MANO's joints: 0 wrist, index 1-3, middle 4-6, pinky 7-9, ring 10-12, thumb 13-15 (from the palm to the tip)
EGOTOUCH_REGIONS = {
    'pinky': ((8, 9), 672, 8),
    'ring': ((11, 12), 555, 11),
    'middle': ((5, 6), 443, 5),
    'index': ((2, 3), 320, 2),
    'thumb': ((14, 15), 744, 14),
    'palm': ((0, 13), None, None),
}
PALM_FACING = 0.5  # a vertex is on the palm side if its normal is within 60 degrees of the palm's normal (-y)


def vertex_normals(template: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(template)
    face_normals = np.cross(template[faces[:, 1]] - template[faces[:, 0]], template[faces[:, 2]] - template[faces[:, 0]])
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)  # weighted by the area of the faces
    return normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)


def egotouch_vertex_weights(assets: dict) -> np.ndarray:
    """(778, 151) weights: the value of a vertex is weights @ taxels, the taxels in lam.tactile.ALL_PRESSURE_CELLS order.
    Each row sums to 1 on the covered vertices, to 0 elsewhere. assets: load_mano_assets' arrays.
    """
    from lam.tactile import GRID_SIZE, PRESSURE_CELLS

    template = np.asarray(assets['template'], dtype=np.float64)
    joints = np.asarray(assets['joints'], dtype=np.float64)
    palm_side = vertex_normals(template, assets['faces'])[:, _THICKNESS_AXIS] < -PALM_FACING
    bone = np.asarray(assets['lbs_weights']).argmax(1)
    plane = template[:, [_FINGER_AXIS, _WIDTH_AXIS]]  # the hand at rest is flat in this plane

    weights = np.zeros((MANO_NUM_VERTICES, sum(len(cells) for cells in PRESSURE_CELLS)), dtype=np.float64)
    offset = 0
    for (bones, tip, base), cells in zip(EGOTOUCH_REGIONS.values(), PRESSURE_CELLS):
        rows = sorted({cell // GRID_SIZE for cell in cells})
        columns = sorted({cell % GRID_SIZE for cell in cells})
        position = {cell: offset + i for i, cell in enumerate(cells)}  # the palm's grid misses cells: none there
        offset += len(cells)

        covered = np.flatnonzero(palm_side & np.isin(bone, bones))
        if tip is None:  # the palm: from the fingers (-x) to the wrist
            axis = np.array([-1.0, 0.0])
            along = -(plane[covered] @ axis)
        else:  # a finger: from its tip to the joint at its base
            axis = plane[tip] - joints[base, [_FINGER_AXIS, _WIDTH_AXIS]]
            along = (plane[tip] - plane[covered]) @ axis / (axis @ axis)
        across = plane[covered] @ np.array([axis[1], -axis[0]])  # towards the thumb side
        v = (along - along.min()) / np.ptp(along) if tip is None else np.clip(along, 0, 1)
        u = (across - across.min()) / np.ptp(across)

        for vertex, gx, gy in zip(covered, u * (len(columns) - 1), v * (len(rows) - 1)):
            x0, y0 = int(gx), int(gy)
            x1, y1 = min(x0 + 1, len(columns) - 1), min(y0 + 1, len(rows) - 1)
            dx, dy = gx - x0, gy - y0
            for (gr, gc), w in (((y0, x0), (1 - dx) * (1 - dy)), ((y0, x1), dx * (1 - dy)),
                                ((y1, x0), (1 - dx) * dy), ((y1, x1), dx * dy)):
                cell = rows[gr] * GRID_SIZE + columns[gc]
                if w > 0 and cell in position:
                    weights[vertex, position[cell]] += w

    total = weights.sum(1, keepdims=True)
    return (weights / np.where(total > 0, total, 1)).astype(np.float32)
