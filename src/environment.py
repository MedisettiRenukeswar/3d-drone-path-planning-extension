"""
src/environment.py
------------------
3D voxel occupancy grid with density-controlled obstacle generation.

Key design decisions
--------------------
* Obstacle density is a parameter, not a side-effect.  The generator
  adds structural obstacles first (cylinders, boxes, walls) then
  fills remaining voxels one-by-one until the target density is
  reached within ±0.5 %.  This means every "low / medium / high"
  condition produces genuinely different environments.

* Start and goal cells (and their immediate neighbours) are always
  cleared after generation, so planners always have valid endpoints.

* All generators are deterministic given (family, grid_size, density,
  seed) — reproducibility is guaranteed.

Seven environment families
--------------------------
  sparse_forest   – thin vertical cylinders, low occlusion
  dense_forest    – wider cylinders, high occlusion
  canyon          – two parallel walls with navigable gaps
  urban_corridor  – rectangular building blocks
  random_clutter  – mixed boxes and cylinders
  narrow_passage  – two walls with a single narrow gap
  maze            – grid-aligned wall segments

Author: Medisetti Renukeswar
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

GRID_SIZE = (25, 25, 12)   # (X, Y, Z) in voxels — used throughout project
START     = (1, 1, 1)
GOAL      = (23, 23, 10)

DENSITY_LOW    = 0.06   # 6 % obstacle density
DENSITY_MEDIUM = 0.14   # 14 %
DENSITY_HIGH   = 0.22   # 22 %

ENV_FAMILIES = [
    "sparse_forest",
    "dense_forest",
    "canyon",
    "urban_corridor",
    "random_clutter",
    "narrow_passage",
    "maze",
]


# ---------------------------------------------------------------------------
# Grid class
# ---------------------------------------------------------------------------

class OccupancyGrid3D:
    """
    3D voxel occupancy grid.

    Attributes
    ----------
    grid : np.ndarray, shape (x_size, y_size, z_size), dtype uint8
        0 = free, 1 = obstacle.
    """

    def __init__(
        self,
        x_size: int = 25,
        y_size: int = 25,
        z_size: int = 12,
        resolution: float = 1.0,
    ) -> None:
        self.x_size     = x_size
        self.y_size     = y_size
        self.z_size     = z_size
        self.resolution = resolution
        self.grid       = np.zeros((x_size, y_size, z_size), dtype=np.uint8)
        self._dist_map: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Obstacle placement
    # ------------------------------------------------------------------

    def add_obstacle(self, x: int, y: int, z: int) -> None:
        """Mark a single voxel as obstacle."""
        if self._in_bounds(x, y, z):
            self.grid[x, y, z] = 1
            self._dist_map = None

    def add_box(
        self,
        x0: int, y0: int, z0: int,
        x1: int, y1: int, z1: int,
    ) -> None:
        """Fill an axis-aligned box [x0:x1, y0:y1, z0:z1] with obstacles."""
        x0, x1 = max(0, x0), min(self.x_size, x1)
        y0, y1 = max(0, y0), min(self.y_size, y1)
        z0, z1 = max(0, z0), min(self.z_size, z1)
        self.grid[x0:x1, y0:y1, z0:z1] = 1
        self._dist_map = None

    # kept for backwards compatibility with original project
    def add_box_obstacle(
        self,
        x_min: int, y_min: int, z_min: int,
        x_max: int, y_max: int, z_max: int,
    ) -> None:
        self.add_box(x_min, y_min, z_min, x_max, y_max, z_max)

    def add_cylinder(
        self,
        cx: int, cy: int,
        radius: int,
        z0: int, z1: int,
    ) -> None:
        """Vertical cylinder obstacle."""
        for x in range(self.x_size):
            for y in range(self.y_size):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.grid[x, y, max(0, z0) : min(self.z_size, z1)] = 1
        self._dist_map = None

    def add_random_obstacles(
        self,
        num_boxes: int = 8,
        num_cylinders: int = 4,
        seed: int = 42,
    ) -> None:
        """Backwards-compatible random obstacle generator (fixed counts)."""
        rng = np.random.default_rng(seed)
        for _ in range(num_boxes):
            x = int(rng.integers(2, self.x_size - 4))
            y = int(rng.integers(2, self.y_size - 4))
            z = int(rng.integers(0, self.z_size - 2))
            w = int(rng.integers(1, 4))
            d = int(rng.integers(1, 4))
            h = int(rng.integers(1, 4))
            self.add_box(x, y, z, x + w, y + d, z + h)
        for _ in range(num_cylinders):
            cx = int(rng.integers(3, self.x_size - 3))
            cy = int(rng.integers(3, self.y_size - 3))
            r  = int(rng.integers(1, 3))
            zt = int(rng.integers(3, self.z_size))
            self.add_cylinder(cx, cy, r, 0, zt)

    def inject_dynamic_obstacle(
        self,
        x: int, y: int, z: int,
        size: int = 2,
    ) -> Tuple[int, int, int, int, int, int]:
        self.add_box(x, y, z, x + size, y + size, z + size)
        return (x, y, z, x + size, y + size, z + size)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_free(self, x: int, y: int, z: int) -> bool:
        return self._in_bounds(x, y, z) and self.grid[x, y, z] == 0

    def is_obstacle(self, x: int, y: int, z: int) -> bool:
        return self._in_bounds(x, y, z) and self.grid[x, y, z] == 1

    def _in_bounds(self, x: int, y: int, z: int) -> bool:
        return (
            0 <= x < self.x_size
            and 0 <= y < self.y_size
            and 0 <= z < self.z_size
        )

    def get_neighbors(
        self,
        x: int, y: int, z: int,
        connectivity: int = 26,
    ) -> List[Tuple[Tuple[int, int, int], float]]:
        """Return (neighbour_pos, edge_cost) for free neighbours."""
        if connectivity == 6:
            deltas = [
                (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
            ]
        else:
            deltas = [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == dy == dz == 0)
            ]
        out = []
        for dx, dy, dz in deltas:
            nx, ny, nz = x + dx, y + dy, z + dz
            if self.is_free(nx, ny, nz):
                cost = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                out.append(((nx, ny, nz), cost))
        return out

    def path_is_clear(self, path: List[Tuple[int, int, int]]) -> bool:
        return all(self.is_free(x, y, z) for x, y, z in path)

    # ------------------------------------------------------------------
    # Distance transform & clearance
    # ------------------------------------------------------------------

    def get_distance_map(self) -> np.ndarray:
        """
        Euclidean distance-to-nearest-obstacle for every voxel (cached).

        Returns float32 array with same shape as grid.
        Obstacle voxels have distance 0.
        """
        if self._dist_map is None:
            free_mask = (self.grid == 0).astype(float)
            self._dist_map = distance_transform_edt(free_mask).astype(np.float32)
        return self._dist_map

    def path_clearance(self, path: List[Tuple[int, int, int]]) -> float:
        """Minimum distance-to-obstacle along a path (voxels)."""
        dm = self.get_distance_map()
        if not path:
            return 0.0
        return float(
            min(
                dm[x, y, z]
                for x, y, z in path
                if self._in_bounds(x, y, z)
            )
        )

    def mean_clearance(self, path: List[Tuple[int, int, int]]) -> float:
        """Mean distance-to-obstacle along a path (voxels)."""
        dm = self.get_distance_map()
        vals = [
            dm[x, y, z]
            for x, y, z in path
            if self._in_bounds(x, y, z)
        ]
        return float(np.mean(vals)) if vals else 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        np.save(filepath, self.grid)

    def load(self, filepath: str) -> None:
        self.grid = np.load(filepath)
        self.x_size, self.y_size, self.z_size = self.grid.shape
        self._dist_map = None

    def save_config(self, filepath: str) -> None:
        config = {
            "x_size": self.x_size,
            "y_size": self.y_size,
            "z_size": self.z_size,
            "resolution": self.resolution,
            "obstacle_count": int(np.sum(self.grid)),
        }
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def obstacle_density(self) -> float:
        total = self.x_size * self.y_size * self.z_size
        return float(np.sum(self.grid)) / total

    def __repr__(self) -> str:
        return (
            f"OccupancyGrid3D("
            f"{self.x_size}×{self.y_size}×{self.z_size}, "
            f"density={self.obstacle_density():.2%})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clear_endpoints(
    grid: OccupancyGrid3D,
    start: Tuple[int, int, int],
    goal:  Tuple[int, int, int],
    radius: int = 1,
) -> None:
    """Guarantee start and goal and their radius-1 neighbourhood are free."""
    for pt in (start, goal):
        px, py, pz = pt
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if grid._in_bounds(px + dx, py + dy, pz + dz):
                        grid.grid[px + dx, py + dy, pz + dz] = 0
    grid._dist_map = None


def _fill_to_density(
    grid:      OccupancyGrid3D,
    target:    float,
    rng:       np.random.Generator,
    start:     Tuple[int, int, int],
    goal:      Tuple[int, int, int],
    tolerance: float = 0.005,
    max_iters: int   = 30_000,
) -> None:
    """
    Iteratively mark single free voxels as obstacles until
    grid.obstacle_density() ≥ target − tolerance.

    Never touches the 2-voxel neighbourhood of start or goal.
    """
    X, Y, Z = grid.x_size, grid.y_size, grid.z_size
    sx, sy, sz = start
    gx, gy, gz = goal
    iters = 0
    while grid.obstacle_density() < target - tolerance and iters < max_iters:
        x = int(rng.integers(0, X))
        y = int(rng.integers(0, Y))
        z = int(rng.integers(0, Z))
        if abs(x - sx) <= 2 and abs(y - sy) <= 2 and abs(z - sz) <= 2:
            iters += 1
            continue
        if abs(x - gx) <= 2 and abs(y - gy) <= 2 and abs(z - gz) <= 2:
            iters += 1
            continue
        if grid.grid[x, y, z] == 0:
            grid.grid[x, y, z] = 1
        iters += 1


# ---------------------------------------------------------------------------
# Environment families
# ---------------------------------------------------------------------------

def _make_sparse_forest(
    gs: Tuple[int,int,int], density: float, seed: int,
    s: Tuple, g: Tuple,
) -> OccupancyGrid3D:
    grid = OccupancyGrid3D(*gs)
    rng  = np.random.default_rng(seed)
    X, Y, Z = gs
    n_trees = max(3, int(X * Y * density * 0.35 / (Z * np.pi)))
    for _ in range(n_trees):
        cx = int(rng.integers(2, X - 2))
        cy = int(rng.integers(2, Y - 2))
        grid.add_cylinder(cx, cy, 1, 0, Z)
    _fill_to_density(grid, density, rng, s, g)
    _clear_endpoints(grid, s, g)
    return grid


def _make_dense_forest(
    gs: Tuple[int,int,int], density: float, seed: int,
    s: Tuple, g: Tuple,
) -> OccupancyGrid3D:
    grid = OccupancyGrid3D(*gs)
    rng  = np.random.default_rng(seed)
    X, Y, Z = gs
    n_trees = max(2, int(X * Y * density * 0.25 / (Z * np.pi)))
    for _ in range(n_trees):
        if grid.obstacle_density() >= density * 0.85:
            break
        cx = int(rng.integers(2, X - 2))
        cy = int(rng.integers(2, Y - 2))
        r  = 1 if density <= 0.08 else int(rng.integers(1, 3))
        grid.add_cylinder(cx, cy, r, 0, Z)
    _fill_to_density(grid, density, rng, s, g)
    _clear_endpoints(grid, s, g)
    return grid


def _make_canyon(
    gs: Tuple[int,int,int], density: float, seed: int,
    s: Tuple, g: Tuple,
) -> OccupancyGrid3D:
    grid = OccupancyGrid3D(*gs)
    rng  = np.random.default_rng(seed)
    X, Y, Z = gs
    w1    = Y // 3
    gap_x = int(rng.integers(X // 4, 3 * X // 4))
    gap_w = max(3, X // 6)
    for x in range(X):
        if grid.obstacle_density() < density * 0.7 and not (gap_x <= x < gap_x + gap_w):
            grid.add_box(x, w1, 0, x + 1, w1 + 2, Z)
    w2     = 2 * Y // 3
    gap_x2 = int(rng.integers(X // 4, 3 * X // 4))
    for x in range(X):
        if grid.obstacle_density() < density * 0.9 and not (gap_x2 <= x < gap_x2 + gap_w):
            grid.add_box(x, w2, 0, x + 1, w2 + 2, Z)
    _fill_to_density(grid, density, rng, s, g)
    _clear_endpoints(grid, s, g)
    return grid


def _make_urban_corridor(
    gs: Tuple[int,int,int], density: float, seed: int,
    s: Tuple, g: Tuple,
) -> OccupancyGrid3D:
    grid = OccupancyGrid3D(*gs)
    rng  = np.random.default_rng(seed)
    X, Y, Z = gs
    bw  = max(3, X // 6)
    bd  = max(3, Y // 6)
    stx = bw + max(2, X // 8)
    sty = bd + max(2, Y // 8)
    for bx in range(1, X - bw, stx):
        for by in range(1, Y - bd, sty):
            if grid.obstacle_density() >= density * 0.85:
                break
            h = int(rng.integers(Z // 2, Z))
            grid.add_box(bx, by, 0, bx + bw, by + bd, h)
    _fill_to_density(grid, density, rng, s, g)
    _clear_endpoints(grid, s, g)
    return grid


def _make_random_clutter(
    gs: Tuple[int,int,int], density: float, seed: int,
    s: Tuple, g: Tuple,
) -> OccupancyGrid3D:
    grid = OccupancyGrid3D(*gs)
    rng  = np.random.default_rng(seed)
    X, Y, Z = gs
    for _ in range(max(3, X // 8)):
        x = int(rng.integers(2, X - 4))
        y = int(rng.integers(2, Y - 4))
        z = int(rng.integers(0, Z - 2))
        w = int(rng.integers(1, max(2, X // 10)))
        d = int(rng.integers(1, max(2, Y // 10)))
        h = int(rng.integers(1, max(2, Z // 4)))
        grid.add_box(x, y, z, x + w, y + d, z + h)
    for _ in range(max(2, X // 12)):
        cx = int(rng.integers(3, X - 3))
        cy = int(rng.integers(3, Y - 3))
        r  = int(rng.integers(1, max(2, X // 12)))
        zt = int(rng.integers(3, Z))
        grid.add_cylinder(cx, cy, r, 0, zt)
    _fill_to_density(grid, density, rng, s, g)
    _clear_endpoints(grid, s, g)
    return grid


def _make_narrow_passage(
    gs: Tuple[int,int,int], density: float, seed: int,
    s: Tuple, g: Tuple,
) -> OccupancyGrid3D:
    grid = OccupancyGrid3D(*gs)
    rng  = np.random.default_rng(seed)
    X, Y, Z = gs
    wx  = X // 3
    gy1 = int(rng.integers(Y // 4, 3 * Y // 4))
    gw  = max(3, Y // 8)
    for y in range(Y):
        if grid.obstacle_density() < density * 0.85 and not (gy1 <= y < gy1 + gw):
            grid.add_box(wx, y, 0, wx + 2, y + 1, Z)
    wx2 = 2 * X // 3
    gy2 = int(rng.integers(Y // 4, 3 * Y // 4))
    for y in range(Y):
        if grid.obstacle_density() < density * 0.95 and not (gy2 <= y < gy2 + gw):
            grid.add_box(wx2, y, 0, wx2 + 2, y + 1, Z)
    _fill_to_density(grid, density, rng, s, g)
    _clear_endpoints(grid, s, g)
    return grid


def _make_maze(
    gs: Tuple[int,int,int], density: float, seed: int,
    s: Tuple, g: Tuple,
) -> OccupancyGrid3D:
    grid = OccupancyGrid3D(*gs)
    rng  = np.random.default_rng(seed)
    X, Y, Z = gs
    cell = max(5, X // 5)
    for xi in range(cell, X - cell, cell):
        if grid.obstacle_density() >= density * 0.80:
            break
        gap = int(rng.integers(0, Y - 2))
        gw  = max(3, Y // 6)
        for y in range(Y):
            if not (gap <= y < gap + gw):
                grid.add_box(xi, y, 0, xi + 1, y + 1, Z - 2)
    _fill_to_density(grid, density, rng, s, g)
    _clear_endpoints(grid, s, g)
    return grid


_BUILDERS = {
    "sparse_forest":  _make_sparse_forest,
    "dense_forest":   _make_dense_forest,
    "canyon":         _make_canyon,
    "urban_corridor": _make_urban_corridor,
    "random_clutter": _make_random_clutter,
    "narrow_passage": _make_narrow_passage,
    "maze":           _make_maze,
}


def make_environment(
    family:    str,
    grid_size: Tuple[int, int, int] = GRID_SIZE,
    seed:      int = 0,
    start:     Tuple[int, int, int] = START,
    goal:      Tuple[int, int, int] = GOAL,
    density:   float = DENSITY_MEDIUM,
) -> OccupancyGrid3D:
    """
    Build a 3D occupancy grid for the requested environment family.

    Parameters
    ----------
    family    : one of ENV_FAMILIES
    grid_size : (X, Y, Z) voxel dimensions
    seed      : environment RNG seed (independent of planner seed)
    start     : start voxel — guaranteed free after construction
    goal      : goal voxel  — guaranteed free after construction
    density   : target obstacle density in [0, 1]

    Returns
    -------
    OccupancyGrid3D with density within ±1.5 % of target
    """
    if family not in _BUILDERS:
        raise ValueError(
            f"Unknown family {family!r}. "
            f"Choose from: {ENV_FAMILIES}"
        )
    grid = _BUILDERS[family](grid_size, density, seed, start, goal)
    # Final safety pass
    _clear_endpoints(grid, start, goal, radius=1)
    return grid
