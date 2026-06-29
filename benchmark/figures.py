"""
benchmark/figures.py
--------------------
Publication-quality figures at 300 DPI.

Colour palette
--------------
Wong (2011) eight-colour palette — fully distinguishable under the most
common forms of colour blindness (protanopia, deuteranopia).

Every figure is generated from actual benchmark CSVs or stats JSON.
No values are hard-coded.

Author: Medisetti Renukeswar
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive; works on every OS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Colour palette (Wong 2011)
# ---------------------------------------------------------------------------

WONG = {
    "A*":            "#0072B2",   # blue
    "RRT*":          "#E69F00",   # orange
    "Informed RRT*": "#009E73",   # green
}
MARKERS = {"A*": "o", "RRT*": "s", "Informed RRT*": "^"}
ALGOS   = ["A*", "RRT*", "Informed RRT*"]
DPI     = 300


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path}")


def _load_rows(csv_path: str) -> List[Dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _vals(
    rows:       List[Dict],
    algo:       str,
    metric:     str,
    found_only: bool = True,
) -> List[float]:
    out = []
    for r in rows:
        if r.get("algorithm") != algo:
            continue
        if found_only and r.get("found") not in ("True", True, "1", 1):
            continue
        try:
            v = float(r[metric])
            if np.isfinite(v):
                out.append(v)
        except (ValueError, KeyError):
            pass
    return out


def _legend_patches() -> List[mpatches.Patch]:
    return [mpatches.Patch(color=WONG[a], label=a) for a in ALGOS]


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def fig_boxplot(
    rows:       List[Dict],
    metric:     str,
    ylabel:     str,
    title:      str,
    output_path: str,
    log_scale:  bool = False,
) -> None:
    """Box plot of `metric` for all three algorithms."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [_vals(rows, a, metric) for a in ALGOS]
    bp   = ax.boxplot(
        data,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"markersize": 3, "alpha": 0.5},
    )
    for patch, algo in zip(bp["boxes"], ALGOS):
        patch.set_facecolor(WONG[algo])
        patch.set_alpha(0.72)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(ALGOS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    if log_scale:
        ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path)


def fig_success_rates(stats_data: Dict, output_path: str) -> None:
    """Grouped bar chart with Wilson 95 % CI."""
    sr    = stats_data.get("success_rates", {})
    algos = [a for a in ALGOS if a in sr]
    rates = [sr[a]["rate"] * 100 for a in algos]
    los   = [sr[a]["wilson_lo"] * 100 for a in algos]
    his   = [sr[a]["wilson_hi"] * 100 for a in algos]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(algos))
    ax.bar(x, rates, color=[WONG[a] for a in algos],
           alpha=0.8, edgecolor="black", linewidth=0.7)
    for i, (rate, lo, hi) in enumerate(zip(rates, los, his)):
        ax.errorbar(
            i, rate,
            yerr=[[max(0.0, rate - lo)], [max(0.0, hi - rate)]],
            fmt="none", color="black", capsize=5, linewidth=1.2,
        )
        ax.text(i, min(hi + 0.8, 103), f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Planning Success Rate — 95 % Wilson CI",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path)


def fig_pareto(rows: List[Dict], output_path: str) -> None:
    """Pareto front: path length vs planning time (log-x)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in ALGOS:
        pl = _vals(rows, algo, "path_length")
        tm = _vals(rows, algo, "time_ms")
        if not pl or not tm:
            continue
        mx, my = np.mean(tm),  np.mean(pl)
        sx, sy = np.std(tm, ddof=1), np.std(pl, ddof=1)
        ax.errorbar(
            mx, my, xerr=sx, yerr=sy,
            fmt=MARKERS[algo], color=WONG[algo],
            markersize=10, capsize=3, linewidth=1.4,
            label=algo,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean Planning Time (ms) — log scale", fontsize=12)
    ax.set_ylabel("Mean Path Length (m)", fontsize=12)
    ax.set_title("Pareto Front: Path Quality vs Speed  (mean ± SD)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path)


def fig_env_heatmap(
    rows:        List[Dict],
    families:    List[str],
    output_path: str,
    metric:      str = "path_length",
    metric_label: str = "Mean Path Length (m)",
) -> None:
    """Heatmap: mean metric per algorithm × environment family."""
    mat = np.full((len(ALGOS), len(families)), np.nan)
    for i, algo in enumerate(ALGOS):
        for j, fam in enumerate(families):
            vals = [
                float(r[metric])
                for r in rows
                if r.get("algorithm") == algo
                and r.get("family") == fam
                and r.get("found") in ("True", True, "1", 1)
                and r.get(metric)
            ]
            if vals:
                mat[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(13, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=30, vmax=52)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(
        [f.replace("_", " ").title() for f in families],
        rotation=28, ha="right", fontsize=10,
    )
    ax.set_yticks(range(len(ALGOS)))
    ax.set_yticklabels(ALGOS, fontsize=11)
    plt.colorbar(im, ax=ax, label=metric_label, fraction=0.025, pad=0.02)
    ax.set_title(f"Multi-Environment Heatmap — {metric_label}",
                 fontsize=12, fontweight="bold", pad=8)
    for i in range(len(ALGOS)):
        for j in range(len(families)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}",
                        ha="center", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)


def fig_density_bars(
    rows:        List[Dict],
    metric:      str,
    ylabel:      str,
    title:       str,
    output_path: str,
) -> None:
    """Grouped bar chart: metric vs density label for each algorithm."""
    density_labels = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x     = np.arange(len(density_labels))
    width = 0.25

    for k, algo in enumerate(ALGOS):
        means, errs = [], []
        for dl in density_labels:
            vals = [
                float(r[metric])
                for r in rows
                if r.get("algorithm") == algo
                and r.get("density_label") == dl
                and r.get("found") in ("True", True, "1", 1)
                and r.get(metric)
            ]
            means.append(np.mean(vals) if vals else 0.0)
            errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        ax.bar(x + k * width, means, width,
               color=WONG[algo], alpha=0.8,
               edgecolor="black", linewidth=0.6, label=algo)
        ax.errorbar(x + k * width, means, yerr=errs,
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [f"Density\n{dl.title()}" for dl in density_labels],
        fontsize=11,
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path)


def fig_convergence(
    conv_csv:    str,
    output_path: str,
    max_iter:    int = 2000,
) -> None:
    """Cost vs iteration curves with IQR bands (RRT* and Informed RRT*)."""
    with open(conv_csv, newline="", encoding="utf-8") as f:
        conv_rows = list(csv.DictReader(f))

    from collections import defaultdict
    by_algo_seed: Dict = defaultdict(lambda: defaultdict(list))
    for r in conv_rows:
        try:
            by_algo_seed[r["algorithm"]][int(r["env_seed"])].append(
                (int(r["iteration"]), float(r["best_cost"]))
            )
        except (ValueError, KeyError):
            pass

    iters = np.arange(0, max_iter + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algo in ["RRT*", "Informed RRT*"]:
        logs = list(by_algo_seed[algo].values())
        if not logs:
            continue
        curves = []
        for log in logs:
            if not log:
                continue
            curve   = np.full(max_iter + 1, np.nan)
            last    = float("inf")
            li      = 0
            for i in range(max_iter + 1):
                while li < len(log) and log[li][0] <= i:
                    last = log[li][1]
                    li  += 1
                if last < float("inf"):
                    curve[i] = last
            curves.append(curve)

        if not curves:
            continue
        mat   = np.array(curves)
        med   = np.nanmedian(mat, axis=0)
        q25   = np.nanpercentile(mat, 25, axis=0)
        q75   = np.nanpercentile(mat, 75, axis=0)
        valid = ~np.isnan(med)
        ax.plot(iters[valid], med[valid],
                color=WONG[algo], label=algo, linewidth=2)
        ax.fill_between(iters[valid], q25[valid], q75[valid],
                        color=WONG[algo], alpha=0.20)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Path Cost (m)", fontsize=12)
    ax.set_title("Convergence: Cost vs Iteration  (median ± IQR over seeds)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path)


def fig_3d_path(
    paths:       Dict[str, List[Tuple]],
    grid,
    start:       Tuple,
    goal:        Tuple,
    title:       str,
    output_path: str,
) -> None:
    """3D path visualisation for multiple algorithms on the same environment."""
    n     = len(paths)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig   = plt.figure(figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    obs = np.argwhere(grid.grid == 1)

    for k, (algo, path) in enumerate(paths.items()):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        if len(obs):
            ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2],
                       c="#E69F00", s=4, alpha=0.18, marker="s")
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            zs = [p[2] for p in path]
            ax.plot(xs, ys, zs, color=WONG.get(algo, "#000000"),
                    linewidth=2.0, label=algo)
        ax.scatter(*start, c="#009E73", s=100, marker="^", zorder=10)
        ax.scatter(*goal,  c="#CC79A7", s=100, marker="*", zorder=10)
        ax.set_title(algo, fontsize=10)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Master: generate every figure from a completed benchmark run
# ---------------------------------------------------------------------------

def generate_all(
    csv_path:    str,
    stats_path:  str,
    conv_csv:    Optional[str],
    figures_dir: str,
    families:    Optional[List[str]] = None,
) -> None:
    """
    Generate all standard figures from benchmark output files.

    Parameters
    ----------
    csv_path    : path to all_runs.csv
    stats_path  : path to stats_results.json
    conv_csv    : path to convergence CSV (may be None)
    figures_dir : output directory
    families    : ENV_FAMILIES order (default: standard order)
    """
    from src.environment import ENV_FAMILIES
    fams = families or ENV_FAMILIES

    rows = _load_rows(csv_path)
    with open(stats_path, encoding="utf-8") as f:
        stats_data = json.load(f)

    os.makedirs(figures_dir, exist_ok=True)

    # Box plots
    fig_boxplot(rows, "path_length", "Path Length (m)",
                "Path Length Distribution  (all environments & densities)",
                os.path.join(figures_dir, "path_length.png"))

    fig_boxplot(rows, "time_ms", "Planning Time (ms)",
                "Planning Time Distribution  (log scale)",
                os.path.join(figures_dir, "planning_time.png"),
                log_scale=True)

    fig_boxplot(rows, "energy_proxy", "Kinematic Energy Proxy (J)",
                "Energy Cost Distribution",
                os.path.join(figures_dir, "energy.png"))

    fig_boxplot(rows, "mean_curvature", "Mean Menger Curvature (1/m)",
                "Path Curvature Distribution  (lower = smoother)",
                os.path.join(figures_dir, "curvature.png"))

    fig_boxplot(rows, "clearance", "Min Obstacle Clearance (voxels)",
                "Obstacle Clearance Distribution",
                os.path.join(figures_dir, "clearance.png"))

    fig_boxplot(rows, "smoothness", "Turning Angle (rad/waypoint)",
                "Path Smoothness Distribution  (lower = smoother)",
                os.path.join(figures_dir, "smoothness.png"))

    # Specialised figures
    fig_success_rates(stats_data, os.path.join(figures_dir, "success_rates.png"))
    fig_pareto(rows, os.path.join(figures_dir, "pareto.png"))
    fig_env_heatmap(rows, fams, os.path.join(figures_dir, "env_heatmap.png"))

    fig_density_bars(rows, "path_length", "Mean Path Length (m)",
                     "Path Length vs Obstacle Density",
                     os.path.join(figures_dir, "density_path.png"))

    fig_density_bars(rows, "time_ms", "Mean Planning Time (ms)",
                     "Planning Time vs Obstacle Density",
                     os.path.join(figures_dir, "density_time.png"))

    if conv_csv and os.path.exists(conv_csv):
        fig_convergence(conv_csv, os.path.join(figures_dir, "convergence.png"))

    print(f"\n  All figures saved to {figures_dir}/")
