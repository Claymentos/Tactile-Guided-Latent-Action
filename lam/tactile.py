"""EgoTouch's glove maps: their layout, and the modules that turn them into tokens and tokens back into them.

The 21x21 map of a glove is laid out as the left glove seen from the palm (the right glove is its mirror image). Of
its 217 cells on the glove, 151 are taxels (pressure) and 15 are bend sensors, one per finger column, measuring how
much the fingers are flexed even without any contact.
"""

import torch
from einops import rearrange
from torch import Tensor, nn

GRID_SIZE = 21
FINGER_COLUMNS = {'pinky': (1, 3), 'ring': (5, 7), 'middle': (9, 11), 'index': (13, 15), 'thumb': (17, 19)}
PAD_ROWS = {'finger': (2, 5), 'thumb': (6, 9)}
BEND_ROW = {'finger': 7, 'thumb': 11}  # a bend strip repeats one sensor per column over its rows: its first row is enough
PALM_ROWS = (13, 17)
PALM_COLUMNS = (1, 19)
PALM_FIRST_ROW_LAST_COLUMN = 15
CONTACT_THRESHOLD = 0.1  # the dataset's contact threshold


def _cells(rows, columns):
    return [row * GRID_SIZE + column for row in rows for column in range(columns[0], columns[1] + 1)]


def _kind(finger):
    return 'thumb' if finger == 'thumb' else 'finger'


_palm_cells = [cell for cell in _cells(range(PALM_ROWS[0], PALM_ROWS[1] + 1), PALM_COLUMNS)
               if cell // GRID_SIZE > PALM_ROWS[0] or cell % GRID_SIZE <= PALM_FIRST_ROW_LAST_COLUMN]
# flattened-grid indices of each hand region: the five fingers, then the palm for the pressure cells
PRESSURE_CELLS = [_cells(range(PAD_ROWS[_kind(f)][0], PAD_ROWS[_kind(f)][1] + 1), c) for f, c in FINGER_COLUMNS.items()] + [_palm_cells]
BEND_CELLS = [_cells([BEND_ROW[_kind(f)]], c) for f, c in FINGER_COLUMNS.items()]
ALL_PRESSURE_CELLS = sum(PRESSURE_CELLS, [])
ALL_BEND_CELLS = sum(BEND_CELLS, [])
# the map draws each bend sensor as a strip, repeating it over these rows of its column
BEND_STRIP_ROWS = {'finger': (7, 11), 'thumb': (11, 12)}
_strips = [(sensor, row * GRID_SIZE + cell % GRID_SIZE)
           for sensor, (finger, cell) in enumerate((f, cell) for f, cells in zip(FINGER_COLUMNS, BEND_CELLS) for cell in cells)
           for row in range(BEND_STRIP_ROWS[_kind(finger)][0], BEND_STRIP_ROWS[_kind(finger)][1] + 1)]
BEND_STRIP_SENSORS = [sensor for sensor, _ in _strips]
BEND_STRIP_CELLS = [cell for _, cell in _strips]
NUM_REGIONS = len(PRESSURE_CELLS)
NUM_GLOVES = 2
NUM_TACTILE_TOKENS = NUM_GLOVES * NUM_REGIONS


def canonical_cells(tactile: Tensor) -> Tensor:
    """(..., 2, 21, 21) gloves (left, right) -> (..., 2, 441), the right glove mirrored onto the left one's layout."""
    left, right = tactile.unbind(dim=-3)
    return torch.stack([left, right.flip(-1)], dim=-3).flatten(-2)


def cells_to_maps(pressure: Tensor, bend: Tensor) -> Tensor:
    """Inverse of canonical_cells for the cells on the glove: (..., 2, 151) taxels and (..., 2, 15) bend sensors ->
    (..., 2, 21, 21) maps as the dataset draws them, 0 off the glove."""
    grid = pressure.new_zeros(*pressure.shape[:-1], GRID_SIZE * GRID_SIZE)
    grid[..., ALL_PRESSURE_CELLS] = pressure
    grid[..., BEND_STRIP_CELLS] = bend[..., BEND_STRIP_SENSORS]
    left, right = grid.unflatten(-1, (GRID_SIZE, GRID_SIZE)).unbind(dim=-3)
    return torch.stack([left, right.flip(-1)], dim=-3)


def zero_init(layer: nn.Linear) -> None:
    """For heads predicting changes: the model starts from 'nothing changes'. Otherwise it first has to learn to copy its
    input, its loss stays above that of the copy for long, and the latents collapse because they bring nothing."""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


def f1_score(prediction: Tensor, target: Tensor) -> Tensor:
    true_positives = (prediction & target).sum()
    return 2 * true_positives / (prediction.sum() + target.sum()).clamp(min=1)


class TactileEncoder(nn.Module):
    """One token per hand region and per glove. Both gloves share the weights, through the canonical layout."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.pressure_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(len(cells), model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
            for cells in PRESSURE_CELLS
        ])
        self.bend_proj = nn.ModuleList([nn.Linear(len(cells), model_dim) for cells in BEND_CELLS])
        self.region_embed = nn.Parameter(torch.randn(1, 1, NUM_REGIONS, model_dim) * 0.02)
        self.glove_embed = nn.Parameter(torch.randn(1, NUM_GLOVES, 1, model_dim) * 0.02)

    def forward(self, tactile: Tensor) -> Tensor:
        """(B, 2, 21, 21) -> (B, NUM_TACTILE_TOKENS, D)"""
        cells = canonical_cells(tactile)
        tokens = [proj(cells[..., region]) for proj, region in zip(self.pressure_proj, PRESSURE_CELLS)]
        for i, (proj, region) in enumerate(zip(self.bend_proj, BEND_CELLS)):  # the palm has no bend sensor
            tokens[i] = tokens[i] + proj(cells[..., region])
        tokens = torch.stack(tokens, dim=2) + self.region_embed + self.glove_embed
        return rearrange(tokens, 'b g r d -> b (g r) d')


class TactileDecoder(nn.Module):
    """From glove tokens to the change of each cell and to the contact logits of the taxels."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.pressure_head = nn.ModuleList([nn.Linear(model_dim, 2 * len(cells)) for cells in PRESSURE_CELLS])
        self.bend_head = nn.ModuleList([nn.Linear(model_dim, len(cells)) for cells in BEND_CELLS])
        for head in (*self.pressure_head, *self.bend_head):
            zero_init(head)

    def forward(self, tokens: Tensor):
        """(B, NUM_TACTILE_TOKENS, D) -> pressure change, contact logits (B, 2, 151) and bend change (B, 2, 15)"""
        tokens = rearrange(tokens, 'b (g r) d -> b g r d', g=NUM_GLOVES)
        pressure = [head(tokens[:, :, i]).chunk(2, dim=-1) for i, head in enumerate(self.pressure_head)]
        delta_pressure = torch.cat([delta for delta, _ in pressure], dim=-1)
        contact_logits = torch.cat([logits for _, logits in pressure], dim=-1)
        delta_bend = torch.cat([head(tokens[:, :, i]) for i, head in enumerate(self.bend_head)], dim=-1)
        return delta_pressure, contact_logits, delta_bend
