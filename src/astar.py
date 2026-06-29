"""
src/astar.py
------------
A* search on a 3D voxel occupancy grid.

Algorithm
---------
Priority queue ordered by f(n) = g(n) + w·h(n).

g(n)  true cost from start to n (Euclidean sum of edge lengths).
h(n)  Euclidean straight-line distance n → goal.
      Admissible and consistent in 26-connected 3D grids, so
      w = 1 guarantees an optimal path.

Edge costs are the Euclidean distance between voxel centres, so
diagonal moves are correctly penalised (√2 for face-diagonal,
√3 for body-diagonal).

Reference
---------
Hart, Nilsson & Raphael (1968). A formal basis for the heuristic
determination of minimum cost paths. IEEE Trans. Syst. Sci. Cybern.

Author: Medisetti Renukeswar
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment import OccupancyGrid3D


@dataclass(order=True)
class _Node:
    f:      float
    g:      float  = field(compare=False)
    pos:    tuple  = field(compare=False)
    parent: object = field(default=None, compare=False)


class AStarPlanner:
    """
    A* path planner for OccupancyGrid3D.

    Parameters
    ----------
    grid   : OccupancyGrid3D
    weight : heuristic inflation factor.  weight=1 → optimal A*.
             weight>1 → weighted A* (faster, possibly sub-optimal).
    """

    def __init__(self, grid: OccupancyGrid3D, weight: float = 1.0) -> None:
        self.grid   = grid
        self.weight = weight

    def plan(
        self,
        start: Tuple[int, int, int],
        goal:  Tuple[int, int, int],
    ) -> Dict:
        """
        Find the shortest collision-free path from start to goal.

        Returns
        -------
        dict with keys:
            path            list of (x,y,z) tuples; empty on failure
            found           bool
            path_length     float  Euclidean length in metres
            nodes_explored  int
            time_ms         float  wall-clock planning time
            algorithm       str    "A*"
        """
        t0    = time.perf_counter()
        start = tuple(int(v) for v in start)
        goal  = tuple(int(v) for v in goal)

        if not self.grid.is_free(*start):
            return self._fail("start in obstacle", t0)
        if not self.grid.is_free(*goal):
            return self._fail("goal in obstacle", t0)

        open_list:  List[_Node]        = []
        closed_set: set                = set()
        g_score:    Dict[tuple, float] = {start: 0.0}
        came_from:  Dict[tuple, tuple] = {}
        nodes_exp   = 0

        heapq.heappush(
            open_list,
            _Node(f=self.weight * self._h(start, goal), g=0.0, pos=start),
        )

        while open_list:
            cur = heapq.heappop(open_list)
            if cur.pos in closed_set:
                continue
            closed_set.add(cur.pos)
            nodes_exp += 1

            if cur.pos == goal:
                path    = self._reconstruct(came_from, goal)
                elapsed = (time.perf_counter() - t0) * 1000.0
                return {
                    "path":           path,
                    "found":          True,
                    "path_length":    self._path_len(path),
                    "nodes_explored": nodes_exp,
                    "time_ms":        elapsed,
                    "algorithm":      "A*",
                }

            for nb, step_cost in self.grid.get_neighbors(*cur.pos):
                if nb in closed_set:
                    continue
                tg = g_score[cur.pos] + step_cost
                if tg < g_score.get(nb, float("inf")):
                    g_score[nb]   = tg
                    came_from[nb] = cur.pos
                    f = tg + self.weight * self._h(nb, goal)
                    heapq.heappush(open_list, _Node(f=f, g=tg, pos=nb))

        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._fail("goal unreachable", t0, nodes_exp, elapsed)

    # ------------------------------------------------------------------

    @staticmethod
    def _h(a: tuple, b: tuple) -> float:
        return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))

    def _reconstruct(
        self, came_from: Dict[tuple, tuple], current: tuple
    ) -> List[tuple]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _path_len(self, path: List[tuple]) -> float:
        if len(path) < 2:
            return 0.0
        pts = np.array(path, dtype=float)
        return round(
            float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            * self.grid.resolution,
            4,
        )

    def _fail(
        self,
        reason:   str,
        t0:       float,
        nodes:    int            = 0,
        elapsed:  Optional[float] = None,
    ) -> Dict:
        if elapsed is None:
            elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "path":           [],
            "found":          False,
            "path_length":    0.0,
            "nodes_explored": nodes,
            "time_ms":        elapsed,
            "algorithm":      "A*",
        }


# ---------------------------------------------------------------------------
# Path post-processing
# ---------------------------------------------------------------------------

def smooth_path(
    path: List[tuple],
    grid: OccupancyGrid3D,
    samples: int = 20,
) -> List[tuple]:
    """
    Line-of-sight shortcutting.  Removes intermediate waypoints when a
    direct segment is collision-free.  Preserves start and goal.
    """
    if len(path) < 3:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _los(path[i], path[j], grid, samples):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


def _los(
    p1: tuple,
    p2: tuple,
    grid: OccupancyGrid3D,
    samples: int = 20,
) -> bool:
    a = np.array(p1, dtype=float)
    b = np.array(p2, dtype=float)
    for t in np.linspace(0, 1, samples):
        pt = a + t * (b - a)
        xi, yi, zi = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
        if not grid.is_free(xi, yi, zi):
            return False
    return True
