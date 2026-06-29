"""
benchmark/runner.py
-------------------
Monte Carlo benchmark engine.

Design principles
-----------------
1. Every trial result is appended to the CSV immediately after
   completion.  If the process is killed, re-running with --resume
   skips already-recorded trials.

2. Planner seed is decoupled from environment seed:
       planner_seed = (env_seed + 1) * 7919
   The +1 prevents planner_seed == 0 when env_seed == 0;
   7919 is prime, giving broad distribution.

3. MAX_ITER = 2000 for RRT* and Informed RRT*.  This is enough for
   the 25×25×12 grid to achieve > 95 % success while keeping each
   trial under ~5 s on a modern laptop.

4. All metrics (path length, clearance, curvature, energy …) are
   computed inside run_single_trial and stored in every CSV row.
   No post-processing fabrication is possible.

Usage (scripts call these functions directly)
-----
    from benchmark.runner import run_benchmark, CSV_COLUMNS
    run_benchmark("results/tables/all_runs.csv", n_seeds=30)

Author: Medisetti Renukeswar
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.environment import (
    make_environment,
    GRID_SIZE, START, GOAL,
    ENV_FAMILIES,
    DENSITY_LOW, DENSITY_MEDIUM, DENSITY_HIGH,
)
from src.astar             import AStarPlanner
from src.rrt_star          import RRTStarPlanner
from src.informed_rrt_star import InformedRRTStarPlanner
from src.metrics           import compute_all

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALGORITHMS = ["A*", "RRT*", "Informed RRT*"]
DENSITIES  = [DENSITY_LOW, DENSITY_MEDIUM, DENSITY_HIGH]
MAX_ITER   = 2000        # sampling budget for RRT* and Informed RRT*
_SEED_PRIME = 7919       # prime for seed decoupling

DENSITY_LABEL = {
    DENSITY_LOW:    "low",
    DENSITY_MEDIUM: "medium",
    DENSITY_HIGH:   "high",
}

CSV_COLUMNS = [
    "run_id", "algorithm", "family",
    "density_label", "density_actual",
    "env_seed", "planner_seed",
    "found",
    "path_length", "time_ms", "nodes_explored",
    "clearance", "mean_clearance",
    "smoothness", "mean_curvature", "max_curvature",
    "jerk", "energy_proxy",
    "dynamic_feasible", "n_waypoints",
]


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

def run_single_trial(
    algorithm:  str,
    family:     str,
    density:    float,
    env_seed:   int,
    run_id:     int,
) -> Dict:
    """
    Execute one complete planning trial and return a CSV-ready row.

    The function is self-contained: it builds the grid, constructs the
    planner, calls plan(), computes all metrics, and packages everything
    into a flat dict.  No state escapes.
    """
    planner_seed = (env_seed + 1) * _SEED_PRIME

    grid    = make_environment(family, GRID_SIZE, env_seed, START, GOAL, density)
    planner = _build_planner(algorithm, grid, planner_seed)
    result  = planner.plan(START, GOAL)

    path    = result.get("path", [])
    metrics = compute_all(path, grid, algorithm)

    return {
        "run_id":           run_id,
        "algorithm":        algorithm,
        "family":           family,
        "density_label":    DENSITY_LABEL.get(density, f"{density:.0%}"),
        "density_actual":   round(grid.obstacle_density(), 4),
        "env_seed":         env_seed,
        "planner_seed":     planner_seed,
        "found":            result["found"],
        "path_length":      metrics["path_length"],
        "time_ms":          round(result.get("time_ms", 0.0), 3),
        "nodes_explored":   result.get("nodes_explored", 0),
        "clearance":        metrics["clearance"],
        "mean_clearance":   metrics["mean_clearance"],
        "smoothness":       metrics["smoothness"],
        "mean_curvature":   metrics["mean_curvature"],
        "max_curvature":    metrics["max_curvature"],
        "jerk":             metrics["jerk"],
        "energy_proxy":     metrics["energy_proxy"],
        "dynamic_feasible": metrics["dynamic_feasible"],
        "n_waypoints":      metrics["n_waypoints"],
    }


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    output_csv:  str,
    algorithms:  Optional[List[str]] = None,
    families:    Optional[List[str]] = None,
    densities:   Optional[List[float]] = None,
    n_seeds:     int  = 30,
    resume:      bool = True,
    verbose:     bool = True,
) -> str:
    """
    Run the Monte Carlo benchmark and stream results to output_csv.

    Parameters
    ----------
    output_csv : destination CSV path
    algorithms : subset of ALGORITHMS (default: all 3)
    families   : subset of ENV_FAMILIES (default: all 7)
    densities  : subset of DENSITIES (default: all 3)
    n_seeds    : number of random seeds per condition (0 … n_seeds-1)
    resume     : skip trials whose key already exists in output_csv
    verbose    : print one line per trial

    Returns
    -------
    Path to the written CSV.
    """
    algos = algorithms or ALGORITHMS
    fams  = families   or ENV_FAMILIES
    dens  = densities  or DENSITIES

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Load already-done keys if resuming
    done_keys: Set[str] = set()
    if resume and os.path.exists(output_csv):
        done_keys = _load_done_keys(output_csv)
        if verbose and done_keys:
            print(f"  [resume] {len(done_keys)} trials already recorded — skipping.")

    total  = len(algos) * len(fams) * len(dens) * n_seeds
    done   = 0
    errors = 0
    run_id = _next_run_id(output_csv)
    t0     = time.time()

    if verbose:
        print(f"\n{'='*62}")
        print(f"  Monte Carlo Benchmark")
        print(f"  Algorithms : {algos}")
        print(f"  Families   : {len(fams)}")
        print(f"  Densities  : {[DENSITY_LABEL[d] for d in dens]}")
        print(f"  Seeds      : {n_seeds}  |  Total trials: {total}")
        print(f"{'='*62}\n")

    for algo in algos:
        for family in fams:
            for density in dens:
                for seed in range(n_seeds):
                    key = _trial_key(algo, family, density, seed)
                    if key in done_keys:
                        done += 1
                        run_id += 1
                        continue

                    try:
                        row = run_single_trial(algo, family, density, seed, run_id)
                    except Exception as exc:
                        row = _error_row(run_id, algo, family, density, seed)
                        errors += 1
                        if verbose:
                            print(f"\n  [ERROR] {algo}/{family}/s={seed}: {exc}")

                    _append_row(row, output_csv)
                    done   += 1
                    run_id += 1

                    if verbose:
                        elapsed = time.time() - t0
                        rate    = done / elapsed if elapsed > 0 else 1
                        eta_s   = (total - done) / rate if rate > 0 else 0
                        print(
                            f"  [{done:4d}/{total}] {algo:15s} "
                            f"{family:16s} d={DENSITY_LABEL[density]:6s} "
                            f"s={seed:2d}  "
                            f"ok={str(row.get('found',False)):5s}  "
                            f"len={row.get('path_length',0.0):.2f}m  "
                            f"t={row.get('time_ms',0.0):.0f}ms  "
                            f"ETA={eta_s/60:.1f}min",
                            flush=True,
                        )

    if verbose:
        rows    = _count_rows(output_csv)
        n_found = _count_found(output_csv)
        wall    = time.time() - t0
        print(f"\n{'='*62}")
        print(f"  Benchmark complete")
        print(f"  Rows: {rows}  |  Success: {n_found}/{rows} = {n_found/max(rows,1):.1%}")
        print(f"  Errors: {errors}  |  Wall time: {wall/60:.1f} min")
        print(f"  Output: {output_csv}")
        print(f"{'='*62}\n")

    return output_csv


# ---------------------------------------------------------------------------
# Convergence study (sampling-based planners only)
# ---------------------------------------------------------------------------

def run_convergence_study(
    output_csv: str,
    n_seeds:    int  = 10,
    verbose:    bool = True,
) -> str:
    """
    Collect (iteration, best_cost) convergence logs for RRT* and
    Informed RRT* on the random_clutter medium-density environment.

    A* is deterministic; it gets a single entry at iteration 0.
    """
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    for algo in ALGORITHMS:
        for seed in range(n_seeds):
            ps   = (seed + 1) * _SEED_PRIME
            grid = make_environment(
                "random_clutter", GRID_SIZE, seed, START, GOAL, DENSITY_MEDIUM
            )
            planner = _build_planner(algo, grid, ps)
            result  = planner.plan(START, GOAL)
            conv    = result.get("convergence_log", [])

            if algo == "A*":
                conv = [(0, result.get("path_length", 0.0))]

            for it, cost in conv:
                rows.append({
                    "algorithm":  algo,
                    "env_seed":   seed,
                    "iteration":  it,
                    "best_cost":  round(cost, 4),
                    "final_cost": round(result.get("path_length", 0.0), 4),
                    "found":      result["found"],
                })

            if verbose:
                print(
                    f"  {algo:15s} s={seed:2d}  "
                    f"improvements={len(conv):3d}  "
                    f"final={result.get('path_length',0):.2f}m",
                    flush=True,
                )

    _write_csv(rows, output_csv)
    if verbose:
        print(f"  Convergence data → {output_csv} ({len(rows)} rows)")
    return output_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_planner(algo: str, grid, planner_seed: int) -> Any:
    if algo == "A*":
        return AStarPlanner(grid)
    if algo == "RRT*":
        return RRTStarPlanner(grid, max_iter=MAX_ITER, seed=planner_seed)
    if algo == "Informed RRT*":
        return InformedRRTStarPlanner(grid, max_iter=MAX_ITER, seed=planner_seed)
    raise ValueError(f"Unknown algorithm: {algo!r}")


def _trial_key(
    algo: str, family: str, density: float, seed: int
) -> str:
    return f"{algo}|{family}|{density:.4f}|{seed}"


def _load_done_keys(csv_path: str) -> Set[str]:
    keys: Set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                keys.add(_trial_key(
                    row["algorithm"],
                    row["family"],
                    float(row.get("density_actual", 0)),
                    int(row["env_seed"]),
                ))
            except (KeyError, ValueError):
                pass
    return keys


def _next_run_id(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    try:
        return max(int(r["run_id"]) for r in rows) + 1
    except (KeyError, ValueError):
        return len(rows)


def _append_row(row: Dict, csv_path: str) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _error_row(
    run_id: int, algo: str, family: str, density: float, seed: int
) -> Dict:
    ps = (seed + 1) * _SEED_PRIME
    return {
        "run_id": run_id, "algorithm": algo,
        "family": family,
        "density_label":  DENSITY_LABEL.get(density, ""),
        "density_actual": density,
        "env_seed": seed, "planner_seed": ps,
        "found": False, "path_length": 0.0,
        "time_ms": 0.0, "nodes_explored": 0,
        "clearance": 0.0, "mean_clearance": 0.0,
        "smoothness": 0.0, "mean_curvature": 0.0,
        "max_curvature": 0.0, "jerk": 0.0,
        "energy_proxy": 0.0, "dynamic_feasible": False,
        "n_waypoints": 0,
    }


def _count_rows(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _count_found(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sum(
            1 for r in csv.DictReader(f)
            if r.get("found") in ("True", True, "1", 1)
        )
