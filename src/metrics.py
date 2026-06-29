"""
src/metrics.py
--------------
Path quality metrics for publication-grade evaluation.

All metrics operate on a raw list of (x, y, z) waypoint tuples.
The grid object is passed only when distance-based metrics are needed.

Metrics
-------
path_length       Euclidean sum of segment lengths (m)
clearance         Minimum distance to nearest obstacle (voxels)
mean_clearance    Mean distance to nearest obstacle (voxels)
smoothness        Mean turning angle at interior waypoints (rad)
mean_curvature    Mean Menger curvature κ (1/m)
max_curvature     Maximum Menger curvature (1/m)
jerk              Mean 3rd-order finite-difference magnitude
energy_proxy      Kinematic energy proxy (J) — see docstring
dynamic_feasible  All κ ≤ KAPPA_MAX

Physical constants
------------------
DRONE_MASS   0.5 kg  (small quadrotor)
DRONE_VMAX   3.0 m/s (cruise speed)
KAPPA_MAX    0.8 1/m  (max physically flyable curvature)

Author: Medisetti Renukeswar
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

DRONE_MASS = 0.5    # kg
DRONE_VMAX = 3.0    # m/s
KAPPA_MAX  = 0.8    # 1/m


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def path_length(path: List[Tuple]) -> float:
    """Euclidean sum of all segment lengths (m)."""
    if len(path) < 2:
        return 0.0
    pts = np.array(path, float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def clearance(path: List[Tuple], grid) -> float:
    """Minimum distance-to-nearest-obstacle along path (voxels)."""
    dm = grid.get_distance_map()
    if not path:
        return 0.0
    return float(min(
        dm[x, y, z] for x, y, z in path if grid._in_bounds(x, y, z)
    ))


def mean_clearance(path: List[Tuple], grid) -> float:
    """Mean distance-to-nearest-obstacle along path (voxels)."""
    dm = grid.get_distance_map()
    vals = [dm[x, y, z] for x, y, z in path if grid._in_bounds(x, y, z)]
    return float(np.mean(vals)) if vals else 0.0


def smoothness(path: List[Tuple]) -> float:
    """
    Mean turning angle at interior waypoints (radians).

    0 rad = perfectly straight,  π rad = 180° reversal.
    Lower is smoother.
    """
    if len(path) < 3:
        return 0.0
    pts = np.array(path, float)
    angles = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angles.append(np.arccos(cos_a))
    return float(np.mean(angles)) if angles else 0.0


def menger_curvature_profile(path: List[Tuple]) -> np.ndarray:
    """
    Menger curvature at each interior waypoint.

        κ_i = 2 · area(△p_{i-1}, p_i, p_{i+1})
              / (|ab| · |bc| · |ca|)

    Returns 1-D array of length max(0, len(path) - 2).
    """
    if len(path) < 3:
        return np.array([])
    pts = np.array(path, float)
    kappas = []
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        area2 = float(np.linalg.norm(np.cross(b - a, c - a)))
        denom = (np.linalg.norm(b - a) *
                 np.linalg.norm(c - b) *
                 np.linalg.norm(a - c))
        kappas.append(area2 / denom if denom > 1e-12 else 0.0)
    return np.array(kappas)


def jerk(path: List[Tuple]) -> float:
    """
    Mean 3rd-order finite-difference magnitude (path jerk proxy).

    This is not physical jerk (d³p/dt³) but a dimensionless
    geometric proxy for trajectory smoothness change.
    """
    if len(path) < 4:
        return 0.0
    pts = np.array(path, float)
    j3  = np.diff(pts, n=3, axis=0)
    return float(np.mean(np.linalg.norm(j3, axis=1)))


def energy_proxy(path: List[Tuple]) -> float:
    """
    Kinematic energy proxy (Joules).

    Models the energy cost of flying a path at constant speed DRONE_VMAX.

        E = ½ · m · v² · n_wps          (traversal KE contribution)
          + m · Σ |Δv_i| · v            (manoeuvre cost at each turn)

    where Δv_i is the direction change at waypoint i, scaled by cruise
    speed to convert to a velocity-change magnitude.

    This proxy is strictly proportional to path length when the path is
    straight, and increases with curvature — capturing the physical
    cost of decelerating and accelerating at corners.
    """
    if len(path) < 2:
        return 0.0
    pts  = np.array(path, float)
    segs = np.diff(pts, axis=0)
    dists = np.linalg.norm(segs, axis=1)
    dirs  = segs / (dists[:, None] + 1e-12)   # unit direction vectors

    # Direction changes at interior waypoints
    delta_v = 0.0
    for i in range(len(dirs) - 1):
        delta_v += float(np.linalg.norm(dirs[i + 1] - dirs[i])) * DRONE_VMAX

    traversal  = 0.5 * DRONE_MASS * DRONE_VMAX ** 2 * len(path)
    manoeuvre  = DRONE_MASS * delta_v
    return float(traversal + manoeuvre)


def is_dynamically_feasible(path: List[Tuple]) -> bool:
    """Return True iff all Menger curvatures ≤ KAPPA_MAX."""
    kappas = menger_curvature_profile(path)
    return bool(len(kappas) == 0 or np.all(kappas <= KAPPA_MAX))


# ---------------------------------------------------------------------------
# Convenience: compute everything in one call
# ---------------------------------------------------------------------------

def compute_all(
    path:      List[Tuple],
    grid,
    algorithm: str = "",
) -> Dict:
    """
    Compute every metric and return a flat dict suitable for CSV export.

    Parameters
    ----------
    path      : list of (x, y, z) tuples from a planner
    grid      : OccupancyGrid3D (used for clearance)
    algorithm : label to embed in the dict

    Returns
    -------
    dict with keys matching the CSV_COLUMNS defined in benchmark/runner.py
    """
    if not path:
        return {
            "algorithm":        algorithm,
            "found":            False,
            "path_length":      0.0,
            "clearance":        0.0,
            "mean_clearance":   0.0,
            "smoothness":       0.0,
            "mean_curvature":   0.0,
            "max_curvature":    0.0,
            "jerk":             0.0,
            "energy_proxy":     0.0,
            "dynamic_feasible": False,
            "n_waypoints":      0,
        }

    kappas = menger_curvature_profile(path)
    return {
        "algorithm":        algorithm,
        "found":            True,
        "path_length":      round(path_length(path), 4),
        "clearance":        round(clearance(path, grid), 4),
        "mean_clearance":   round(mean_clearance(path, grid), 4),
        "smoothness":       round(smoothness(path), 4),
        "mean_curvature":   round(float(np.mean(kappas)) if len(kappas) else 0.0, 6),
        "max_curvature":    round(float(np.max(kappas))  if len(kappas) else 0.0, 6),
        "jerk":             round(jerk(path), 6),
        "energy_proxy":     round(energy_proxy(path), 4),
        "dynamic_feasible": is_dynamically_feasible(path),
        "n_waypoints":      len(path),
    }
