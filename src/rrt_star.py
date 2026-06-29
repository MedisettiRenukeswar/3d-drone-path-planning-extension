"""
src/rrt_star.py
---------------
RRT* with the theoretically correct shrinking rewire radius.

The rewire radius follows Karaman & Frazzoli (2011) Theorem 38:

    r_n = min( γ·(log n / n)^{1/d},  η )

where:
    d = 3  (workspace dimension)
    η = step_size × 5  (upper bound — prevents O(n²) neighbour queries)
    γ = 2·(1 + 1/d)^{1/d}·(μ(X_free)/ζ_d)^{1/d}

    ζ_d = 4π/3  (volume of unit 3-ball)
    μ(X_free)   = free-space volume in voxels

This radius shrinks as n grows, which is essential for asymptotic
optimality.  Using a fixed radius (as in many implementations) is
incorrect; it violates the convergence guarantee.

Convergence log
---------------
Every time the best-known path cost improves, the tuple
(iteration, best_cost) is appended to result["convergence_log"].
This is used to produce convergence-curve figures.

Reference
---------
Karaman & Frazzoli (2011). Sampling-based algorithms for optimal
motion planning. IJRR 30(7):846-894.

Author: Medisetti Renukeswar
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment import OccupancyGrid3D


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class RRTNode:
    pos:      tuple
    parent:   Optional["RRTNode"] = field(default=None, repr=False)
    cost:     float               = 0.0
    children: List                = field(default_factory=list, repr=False)

    def __hash__(self)         -> int:  return hash(self.pos)
    def __eq__(self, other)    -> bool: return self.pos == other.pos


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class RRTStarPlanner:
    """
    RRT* planner for OccupancyGrid3D.

    Parameters
    ----------
    grid             : OccupancyGrid3D
    max_iter         : sampling budget
    step_size        : max extension distance per iteration (voxels)
    goal_sample_rate : probability of directly sampling the goal
    seed             : RNG seed — must be independent of env seed
    """

    def __init__(
        self,
        grid:             OccupancyGrid3D,
        max_iter:         int            = 2000,
        step_size:        float          = 1.5,
        goal_sample_rate: float          = 0.10,
        seed:             Optional[int]  = None,
    ) -> None:
        self.grid             = grid
        self.max_iter         = max_iter
        self.step_size        = step_size
        self.goal_sample_rate = goal_sample_rate
        self.rng              = np.random.default_rng(seed)
        self.nodes:   List[RRTNode] = []
        self._gamma:  float         = self._compute_gamma()

    def plan(
        self,
        start: Tuple[int, int, int],
        goal:  Tuple[int, int, int],
    ) -> Dict:
        """
        Run RRT* and return the best path found within max_iter.

        Returns
        -------
        dict with keys:
            path, found, path_length, nodes_explored, time_ms,
            algorithm, convergence_log
        """
        t0    = time.perf_counter()
        start = tuple(int(v) for v in start)
        goal  = tuple(int(v) for v in goal)

        if not self.grid.is_free(*start):
            return self._fail("start in obstacle", t0)
        if not self.grid.is_free(*goal):
            return self._fail("goal in obstacle", t0)

        self._gamma = self._compute_gamma()
        self.nodes  = [RRTNode(pos=start, cost=0.0)]
        goal_node:  Optional[RRTNode]          = None
        best_cost   = float("inf")
        conv_log:   List[Tuple[int, float]]    = []

        for it in range(self.max_iter):
            q_rand    = self._sample(goal)
            q_near    = self._nearest(q_rand)
            q_new_pos = self._steer(q_near.pos, q_rand)

            if not self._collision_free(q_near.pos, q_new_pos):
                continue

            r        = self._rewire_radius(len(self.nodes) + 1)
            nbrs     = self._near(q_new_pos, r)
            q_new    = self._choose_parent(q_new_pos, nbrs, q_near)

            self.nodes.append(q_new)
            if q_new.parent:
                q_new.parent.children.append(q_new)
            self._rewire(q_new, nbrs)

            if self._dist(q_new.pos, goal) <= self.step_size:
                if self._collision_free(q_new.pos, goal):
                    gc = q_new.cost + self._dist(q_new.pos, goal)
                    if gc < best_cost:
                        best_cost = gc
                        goal_node = RRTNode(pos=goal, parent=q_new, cost=gc)
                        conv_log.append((it, best_cost))

        elapsed = (time.perf_counter() - t0) * 1000.0
        if goal_node is None:
            return self._fail("goal not reached", t0, len(self.nodes), elapsed)

        path = self._extract(goal_node)
        return {
            "path":             path,
            "found":            True,
            "path_length":      self._path_len(path),
            "nodes_explored":   len(self.nodes),
            "time_ms":          elapsed,
            "algorithm":        "RRT*",
            "convergence_log":  conv_log,
        }

    # ------------------------------------------------------------------
    # Karaman-Frazzoli radius
    # ------------------------------------------------------------------

    def _compute_gamma(self) -> float:
        d     = 3
        zeta  = (4.0 / 3.0) * np.pi          # volume of unit 3-ball
        mu    = (1.0 - self.grid.obstacle_density()) * (
            self.grid.x_size * self.grid.y_size * self.grid.z_size
        )
        return 2.0 * (1 + 1 / d) ** (1 / d) * (mu / zeta) ** (1 / d)

    def _rewire_radius(self, n: int) -> float:
        d = 3
        r = self._gamma * (np.log(max(n, 2)) / max(n, 2)) ** (1 / d)
        return min(r, self.step_size * 5.0)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _sample(self, goal: tuple) -> tuple:
        if self.rng.random() < self.goal_sample_rate:
            return goal
        return (
            int(self.rng.integers(0, self.grid.x_size)),
            int(self.rng.integers(0, self.grid.y_size)),
            int(self.rng.integers(0, self.grid.z_size)),
        )

    def _nearest(self, q: tuple) -> RRTNode:
        return min(self.nodes, key=lambda n: self._dist(n.pos, q))

    def _steer(self, frm: tuple, to: tuple) -> tuple:
        diff = np.array(to, float) - np.array(frm, float)
        dist = float(np.linalg.norm(diff))
        if dist <= self.step_size:
            raw = np.array(to, float)
        else:
            raw = np.array(frm, float) + (diff / dist) * self.step_size
        x = int(np.clip(round(raw[0]), 0, self.grid.x_size - 1))
        y = int(np.clip(round(raw[1]), 0, self.grid.y_size - 1))
        z = int(np.clip(round(raw[2]), 0, self.grid.z_size - 1))
        return (x, y, z)

    def _collision_free(self, p1: tuple, p2: tuple, samples: int = 15) -> bool:
        a = np.array(p1, float)
        b = np.array(p2, float)
        for t in np.linspace(0, 1, samples):
            pt = a + t * (b - a)
            xi = int(round(pt[0]))
            yi = int(round(pt[1]))
            zi = int(round(pt[2]))
            if not self.grid.is_free(xi, yi, zi):
                return False
        return True

    def _near(self, pos: tuple, r: float) -> List[RRTNode]:
        return [n for n in self.nodes if self._dist(n.pos, pos) <= r]

    def _choose_parent(
        self,
        pos:      tuple,
        nbrs:     List[RRTNode],
        fallback: RRTNode,
    ) -> RRTNode:
        best_p = fallback
        best_c = fallback.cost + self._dist(fallback.pos, pos)
        for n in nbrs:
            c = n.cost + self._dist(n.pos, pos)
            if c < best_c and self._collision_free(n.pos, pos):
                best_c, best_p = c, n
        return RRTNode(pos=pos, parent=best_p, cost=best_c)

    def _rewire(self, q_new: RRTNode, nbrs: List[RRTNode]) -> None:
        for n in nbrs:
            if n is q_new.parent:
                continue
            nc = q_new.cost + self._dist(q_new.pos, n.pos)
            if nc < n.cost and self._collision_free(q_new.pos, n.pos):
                if n.parent and n in n.parent.children:
                    n.parent.children.remove(n)
                n.parent = q_new
                n.cost   = nc
                q_new.children.append(n)
                self._propagate(n)

    def _propagate(self, node: RRTNode) -> None:
        for c in node.children:
            c.cost = node.cost + self._dist(node.pos, c.pos)
            self._propagate(c)

    def _extract(self, node: RRTNode) -> List[tuple]:
        path: List[tuple] = []
        cur = node
        while cur is not None:
            path.append(cur.pos)
            cur = cur.parent
        return path[::-1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dist(self, a: tuple, b: tuple) -> float:
        return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))

    def _path_len(self, path: List[tuple]) -> float:
        if len(path) < 2:
            return 0.0
        pts = np.array(path, float)
        return round(
            float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            * self.grid.resolution,
            4,
        )

    def _fail(
        self,
        reason:  str,
        t0:      float,
        nodes:   int            = 0,
        elapsed: Optional[float] = None,
    ) -> Dict:
        if elapsed is None:
            elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "path":            [],
            "found":           False,
            "path_length":     0.0,
            "nodes_explored":  nodes,
            "time_ms":         elapsed,
            "algorithm":       "RRT*",
            "convergence_log": [],
        }
