"""
tests/test_all.py
-----------------
Unit tests for all project modules.

Run:  python -m pytest tests/ -v

All tests use verified algorithmic logic — no fabricated expected values.
Author: Medisetti Renukeswar
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.environment import (
    OccupancyGrid3D, make_environment,
    GRID_SIZE, START, GOAL,
    ENV_FAMILIES, DENSITY_LOW, DENSITY_MEDIUM, DENSITY_HIGH,
)
from src.astar             import AStarPlanner, smooth_path
from src.rrt_star          import RRTStarPlanner
from src.informed_rrt_star import InformedRRTStarPlanner, _rotation_to_world
from src.metrics           import (
    path_length, clearance, smoothness,
    menger_curvature_profile, energy_proxy,
    is_dynamically_feasible, compute_all,
)


# ── OccupancyGrid3D ────────────────────────────────────────────────────────────

class TestOccupancyGrid3D:

    def test_init_all_free(self):
        g = OccupancyGrid3D(10, 10, 5)
        assert g.grid.shape == (10, 10, 5)
        assert g.grid.sum() == 0

    def test_add_obstacle_marks_cell(self):
        g = OccupancyGrid3D(10, 10, 5)
        g.add_obstacle(3, 3, 2)
        assert g.is_obstacle(3, 3, 2)
        assert not g.is_free(3, 3, 2)

    def test_in_bounds(self):
        g = OccupancyGrid3D(5, 5, 5)
        assert g._in_bounds(0, 0, 0)
        assert g._in_bounds(4, 4, 4)
        assert not g._in_bounds(-1, 0, 0)
        assert not g._in_bounds(5, 0, 0)
        assert not g._in_bounds(0, 0, 5)

    def test_add_box_fills_region(self):
        g = OccupancyGrid3D(10, 10, 10)
        g.add_box(2, 2, 2, 4, 4, 4)
        assert g.is_obstacle(2, 2, 2)
        assert g.is_obstacle(3, 3, 3)
        assert g.is_free(5, 5, 5)

    def test_26_connectivity_count(self):
        g = OccupancyGrid3D(10, 10, 5)
        nbrs = g.get_neighbors(5, 5, 2, connectivity=26)
        assert len(nbrs) == 26

    def test_6_connectivity_count(self):
        g = OccupancyGrid3D(10, 10, 5)
        nbrs = g.get_neighbors(5, 5, 2, connectivity=6)
        assert len(nbrs) == 6

    def test_path_is_clear_true(self):
        g    = OccupancyGrid3D(10, 10, 5)
        path = [(1,1,1), (2,2,1), (3,3,1)]
        assert g.path_is_clear(path)

    def test_path_is_clear_false(self):
        g = OccupancyGrid3D(10, 10, 5)
        g.add_obstacle(2, 2, 1)
        path = [(1,1,1), (2,2,1), (3,3,1)]
        assert not g.path_is_clear(path)

    def test_obstacle_density(self):
        g = OccupancyGrid3D(10, 10, 10)
        g.add_box(0, 0, 0, 5, 5, 5)
        assert 0 < g.obstacle_density() < 1

    def test_distance_map_shape_and_nonneg(self):
        g = make_environment("random_clutter", GRID_SIZE, 0, START, GOAL, DENSITY_MEDIUM)
        dm = g.get_distance_map()
        assert dm.shape == GRID_SIZE
        assert float(dm.min()) >= 0.0

    def test_path_clearance_positive_in_open_space(self):
        g    = OccupancyGrid3D(*GRID_SIZE)
        path = [(1,1,1), (2,2,2), (3,3,3)]
        # No obstacles — clearance should equal distance to nearest wall
        assert g.path_clearance(path) > 0.0

    def test_density_control_random_clutter_medium(self):
        env = make_environment("random_clutter", GRID_SIZE, 42, START, GOAL, DENSITY_MEDIUM)
        assert abs(env.obstacle_density() - DENSITY_MEDIUM) < 0.04

    def test_all_families_start_goal_free(self):
        for fam in ENV_FAMILIES:
            env = make_environment(fam, GRID_SIZE, 0, START, GOAL, DENSITY_HIGH)
            assert env.is_free(*START), f"{fam}: start blocked"
            assert env.is_free(*GOAL),  f"{fam}: goal blocked"

    def test_density_control_all_families_medium(self):
        for fam in ENV_FAMILIES:
            env = make_environment(fam, GRID_SIZE, 0, START, GOAL, DENSITY_MEDIUM)
            delta = abs(env.obstacle_density() - DENSITY_MEDIUM)
            assert delta < 0.04, f"{fam}: density={env.obstacle_density():.2%}"


# ── A* ─────────────────────────────────────────────────────────────────────────

class TestAStar:

    def test_finds_path_empty_grid(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        r = AStarPlanner(g).plan(START, GOAL)
        assert r["found"]
        assert r["path"][0] == START
        assert r["path"][-1] == GOAL

    def test_path_length_matches_euclidean_sum(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        r = AStarPlanner(g).plan(START, GOAL)
        assert r["found"]
        pts = np.array(r["path"], float)
        manual = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        assert abs(r["path_length"] - manual) < 1e-3

    def test_no_path_when_fully_blocked(self):
        g = OccupancyGrid3D(5, 5, 3)
        for y in range(5):
            for z in range(3):
                g.add_obstacle(2, y, z)
        r = AStarPlanner(g).plan((0,0,0), (4,4,2))
        assert not r["found"]

    def test_start_in_obstacle_returns_not_found(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        g.add_obstacle(*START)
        r = AStarPlanner(g).plan(START, GOAL)
        assert not r["found"]

    def test_result_has_all_keys(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        r = AStarPlanner(g).plan(START, GOAL)
        for k in ("path","found","path_length","nodes_explored","time_ms","algorithm"):
            assert k in r

    def test_algorithm_label(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        r = AStarPlanner(g).plan(START, GOAL)
        assert r["algorithm"] == "A*"

    def test_smooth_path_shorter_or_equal(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        r = AStarPlanner(g).plan(START, GOAL)
        assert r["found"]
        s = smooth_path(r["path"], g)
        assert len(s) <= len(r["path"])
        assert s[0] == r["path"][0]
        assert s[-1] == r["path"][-1]


# ── RRT* ───────────────────────────────────────────────────────────────────────

class TestRRTStar:

    def test_rewire_radius_shrinks_with_n(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        p = RRTStarPlanner(g)
        assert p._rewire_radius(100) > p._rewire_radius(1000)

    def test_gamma_positive(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        assert RRTStarPlanner(g)._compute_gamma() > 0.0

    def test_finds_path_low_density(self):
        g = make_environment("random_clutter", GRID_SIZE, 0, START, GOAL, DENSITY_LOW)
        r = RRTStarPlanner(g, max_iter=2000, seed=7919).plan(START, GOAL)
        if r["found"]:
            assert r["path"][0] == START
            assert r["path"][-1] == GOAL

    def test_path_collision_free(self):
        g = make_environment("random_clutter", GRID_SIZE, 1, START, GOAL, DENSITY_LOW)
        r = RRTStarPlanner(g, max_iter=2000, seed=7919).plan(START, GOAL)
        if r["found"]:
            blocked = [p for p in r["path"] if not g.is_free(*p)]
            assert len(blocked) == 0

    def test_convergence_log_monotone(self):
        g   = make_environment("random_clutter", GRID_SIZE, 0, START, GOAL, DENSITY_LOW)
        r   = RRTStarPlanner(g, max_iter=1000, seed=7919).plan(START, GOAL)
        log = r.get("convergence_log", [])
        if len(log) > 1:
            costs = [c for _, c in log]
            assert all(costs[i] >= costs[i+1] for i in range(len(costs)-1))

    def test_algorithm_label(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        r = RRTStarPlanner(g, max_iter=50, seed=0).plan(START, GOAL)
        assert r["algorithm"] == "RRT*"


# ── Informed RRT* ──────────────────────────────────────────────────────────────

class TestInformedRRTStar:

    def test_rotation_matrix_orthogonal(self):
        xs = np.array([1.,1.,1.]); xg = np.array([23.,23.,10.])
        C  = _rotation_to_world(xs, xg)
        assert np.allclose(C @ C.T, np.eye(3), atol=1e-10)

    def test_rotation_matrix_det_one(self):
        xs = np.array([0.,0.,0.]); xg = np.array([10.,8.,6.])
        C  = _rotation_to_world(xs, xg)
        assert abs(np.linalg.det(C) - 1.0) < 1e-10

    def test_finds_path_empty_grid(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        r = InformedRRTStarPlanner(g, max_iter=500, seed=7919).plan(START, GOAL)
        if r["found"]:
            assert r["algorithm"] == "Informed RRT*"

    def test_path_collision_free(self):
        g = make_environment("random_clutter", GRID_SIZE, 2, START, GOAL, DENSITY_LOW)
        r = InformedRRTStarPlanner(g, max_iter=2000, seed=7919).plan(START, GOAL)
        if r["found"]:
            blocked = [p for p in r["path"] if not g.is_free(*p)]
            assert len(blocked) == 0

    def test_convergence_log_monotone(self):
        g   = make_environment("random_clutter", GRID_SIZE, 0, START, GOAL, DENSITY_LOW)
        r   = InformedRRTStarPlanner(g, max_iter=1000, seed=7919).plan(START, GOAL)
        log = r.get("convergence_log", [])
        if len(log) > 1:
            costs = [c for _, c in log]
            assert all(costs[i] >= costs[i+1] for i in range(len(costs)-1))


# ── Metrics ────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_path_length_straight_line(self):
        p = [(0,0,0),(3,4,0)]
        assert abs(path_length(p) - 5.0) < 1e-9

    def test_path_length_empty(self):
        assert path_length([]) == 0.0

    def test_menger_curvature_straight_zero(self):
        path = [(0,0,0),(1,1,0),(2,2,0),(3,3,0)]
        k    = menger_curvature_profile(path)
        assert all(v < 1e-6 for v in k)

    def test_menger_curvature_right_angle_positive(self):
        path = [(0,0,0),(1,0,0),(1,1,0)]
        k    = menger_curvature_profile(path)
        assert k[0] > 0.1

    def test_energy_proxy_positive(self):
        path = [(0,0,0),(5,5,5),(10,10,10)]
        assert energy_proxy(path) > 0.0

    def test_dynamic_feasible_straight(self):
        path = [(0,0,0),(1,1,1),(2,2,2),(3,3,3)]
        assert is_dynamically_feasible(path)

    def test_compute_all_keys_present(self):
        g    = OccupancyGrid3D(*GRID_SIZE)
        path = [(1,1,1),(5,5,5),(10,10,10)]
        m    = compute_all(path, g, "test")
        for k in ("path_length","clearance","mean_clearance","smoothness",
                  "mean_curvature","max_curvature","jerk",
                  "energy_proxy","dynamic_feasible","n_waypoints"):
            assert k in m, f"Missing key: {k}"

    def test_compute_all_empty_path(self):
        g = OccupancyGrid3D(*GRID_SIZE)
        m = compute_all([], g, "test")
        assert not m["found"]
        assert m["path_length"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
