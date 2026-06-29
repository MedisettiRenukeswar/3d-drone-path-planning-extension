"""
src/informed_rrt_star.py
------------------------
Informed RRT* — Gammell, Srinivasa & Barfoot (IROS 2014, IJRR 2018).

Once an initial feasible path of cost c_best is found, all subsequent
samples are drawn from the prolate hyperspheroid containing every path
shorter than c_best:

    transverse semi-axis   a₁ = c_best / 2
    conjugate semi-axes    aᵢ = √(c_best² − c_min²) / 2

The ellipsoid orientation is set so its major axis aligns with the
start→goal direction. The rotation matrix C is computed via SVD so
that C[:,0] = (goal − start)/‖goal − start‖.

This is the IROS 2014 implementation (direct ellipsoid sampling),
not the later measure-theoretic version.

Reference
---------
Gammell, Srinivasa & Barfoot (2014). Informed RRT*: Optimal
sampling-based path planning focused via direct sampling of an
admissible ellipsoidal heuristic. IROS 2014.

Author: Medisetti Renukeswar
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment import OccupancyGrid3D
from src.rrt_star    import RRTNode, RRTStarPlanner


class InformedRRTStarPlanner(RRTStarPlanner):
    """
    Informed RRT* — inherits all RRT* mechanics; overrides sampling.

    After finding the first feasible solution, samples are restricted
    to the heuristic ellipsoid.  The ellipsoid shrinks as c_best
    improves, concentrating effort where improvement is still possible.
    """

    def __init__(
        self,
        grid:             OccupancyGrid3D,
        max_iter:         int           = 2000,
        step_size:        float         = 1.5,
        goal_sample_rate: float         = 0.10,
        seed:             Optional[int] = None,
    ) -> None:
        super().__init__(grid, max_iter, step_size, goal_sample_rate, seed)
        # Ellipsoid state — set during plan()
        self._c_best:   float                = float("inf")
        self._c_min:    float                = 1.0
        self._C:        Optional[np.ndarray] = None   # 3×3 rotation
        self._x_centre: Optional[np.ndarray] = None   # midpoint

    def plan(
        self,
        start: Tuple[int, int, int],
        goal:  Tuple[int, int, int],
    ) -> Dict:
        t0    = time.perf_counter()
        start = tuple(int(v) for v in start)
        goal  = tuple(int(v) for v in goal)

        if not self.grid.is_free(*start):
            return self._fail("start in obstacle", t0)
        if not self.grid.is_free(*goal):
            return self._fail("goal in obstacle", t0)

        xs = np.array(start, float)
        xg = np.array(goal, float)
        self._c_min    = float(np.linalg.norm(xg - xs))
        self._x_centre = (xs + xg) / 2.0
        self._C        = _rotation_to_world(xs, xg)
        self._c_best   = float("inf")
        self._gamma    = self._compute_gamma()
        self.nodes     = [RRTNode(pos=start, cost=0.0)]
        goal_node: Optional[RRTNode]       = None
        best_cost  = float("inf")
        conv_log:  List[Tuple[int, float]] = []

        for it in range(self.max_iter):
            q_rand    = self._sample(goal)
            q_near    = self._nearest(q_rand)
            q_new_pos = self._steer(q_near.pos, q_rand)

            if not self._collision_free(q_near.pos, q_new_pos):
                continue

            r     = self._rewire_radius(len(self.nodes) + 1)
            nbrs  = self._near(q_new_pos, r)
            q_new = self._choose_parent(q_new_pos, nbrs, q_near)

            self.nodes.append(q_new)
            if q_new.parent:
                q_new.parent.children.append(q_new)
            self._rewire(q_new, nbrs)

            if self._dist(q_new.pos, goal) <= self.step_size:
                if self._collision_free(q_new.pos, goal):
                    gc = q_new.cost + self._dist(q_new.pos, goal)
                    if gc < best_cost:
                        best_cost      = gc
                        self._c_best   = gc
                        goal_node      = RRTNode(pos=goal, parent=q_new, cost=gc)
                        conv_log.append((it, best_cost))

        elapsed = (time.perf_counter() - t0) * 1000.0
        if goal_node is None:
            return self._fail("goal not reached", t0, len(self.nodes), elapsed)

        path = self._extract(goal_node)
        return {
            "path":            path,
            "found":           True,
            "path_length":     self._path_len(path),
            "nodes_explored":  len(self.nodes),
            "time_ms":         elapsed,
            "algorithm":       "Informed RRT*",
            "convergence_log": conv_log,
        }

    # ------------------------------------------------------------------
    # Overridden sampler
    # ------------------------------------------------------------------

    def _sample(self, goal: tuple) -> tuple:
        if self.rng.random() < self.goal_sample_rate:
            return goal
        if self._c_best < float("inf") and self._C is not None:
            return self._sample_ellipsoid()
        return (
            int(self.rng.integers(0, self.grid.x_size)),
            int(self.rng.integers(0, self.grid.y_size)),
            int(self.rng.integers(0, self.grid.z_size)),
        )

    def _sample_ellipsoid(self) -> tuple:
        """
        Draw a uniform sample from the prolate hyperspheroid.

        Method: sample unit 3-ball → scale by semi-axes → rotate → translate.
        """
        d  = 3
        a1 = self._c_best / 2.0
        ai = float(np.sqrt(max(self._c_best ** 2 - self._c_min ** 2, 1e-12))) / 2.0
        L  = np.diag([a1, ai, ai])

        # Rejection-free unit-ball sample (Muller 1959)
        u = self.rng.standard_normal(d)
        u /= float(np.linalg.norm(u)) + 1e-12
        r  = float(self.rng.random()) ** (1.0 / d)
        x_ball = r * u

        x_world = self._C @ (L @ x_ball) + self._x_centre
        xi = int(np.clip(round(x_world[0]), 0, self.grid.x_size - 1))
        yi = int(np.clip(round(x_world[1]), 0, self.grid.y_size - 1))
        zi = int(np.clip(round(x_world[2]), 0, self.grid.z_size - 1))
        return (xi, yi, zi)

    def _fail(self, reason, t0, nodes=0, elapsed=None):
        if elapsed is None:
            elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "path": [], "found": False, "path_length": 0.0,
            "nodes_explored": nodes, "time_ms": elapsed,
            "algorithm": "Informed RRT*", "convergence_log": [],
        }


# ---------------------------------------------------------------------------
# Rotation matrix (module-level so it can be tested directly)
# ---------------------------------------------------------------------------

def _rotation_to_world(xs: np.ndarray, xg: np.ndarray) -> np.ndarray:
    """
    Build a 3×3 rotation matrix C such that C[:,0] aligns with
    (xg - xs) / ‖xg - xs‖.

    Uses the SVD construction from Gammell et al. (2014) supplementary.
    Guarantees det(C) = +1 (proper rotation).
    """
    d  = 3
    a1 = (xg - xs) / (float(np.linalg.norm(xg - xs)) + 1e-12)
    M  = np.outer(a1, np.array([1.0] + [0.0] * (d - 1)))
    U, _, Vt = np.linalg.svd(M)
    diag       = np.ones(d)
    diag[-1]   = float(np.linalg.det(U)) * float(np.linalg.det(Vt.T))
    return U @ np.diag(diag) @ Vt
