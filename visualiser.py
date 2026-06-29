"""
visualiser.py
=============
Real-time 3D drone path planning visualiser.

Runs entirely inside matplotlib — no Gazebo, no ROS, no GPU required.
Works on Windows 10 / 11 with Python 3.9+.

What it shows
-------------
  LEFT panel   – 3D environment with obstacles, planned path, drone position
  RIGHT top    – Live metrics dashboard (path length, time, clearance, energy)
  RIGHT bottom – Control panel (algorithm, environment, density, seed, speed)

Controls (click the buttons in the window)
-------------------------------------------
  [Plan]      – run the selected planner and show the path
  [Fly]       – animate the drone flying the planned path
  [Stop]      – halt a running animation
  [New Env]   – generate a new random environment (new seed)
  [Compare]   – run all three planners and compare side-by-side
  Algorithm   – dropdown: A* / RRT* / Informed RRT*
  Environment – dropdown: all 7 families
  Density     – dropdown: Low 6% / Medium 14% / High 22%

Usage
-----
  python visualiser.py

Author: Medisetti Renukeswar
"""
from __future__ import annotations

import sys
import os
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple

# ── ensure src/ is importable ─────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # works on Windows 10; falls back gracefully
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — side-effect import

from src.environment import (
    make_environment, OccupancyGrid3D,
    GRID_SIZE, START, GOAL,
    ENV_FAMILIES, DENSITY_LOW, DENSITY_MEDIUM, DENSITY_HIGH,
)
from src.astar             import AStarPlanner
from src.rrt_star          import RRTStarPlanner
from src.informed_rrt_star import InformedRRTStarPlanner
from src.metrics           import (
    compute_all, path_length as _pl, clearance as _clr,
    mean_clearance as _mclr, energy_proxy as _ek,
    smoothness as _sm, menger_curvature_profile as _kap,
)

# ---------------------------------------------------------------------------
# Colour palette (Wong 2011, colourblind-safe)
# ---------------------------------------------------------------------------
C = {
    "A*":            "#0072B2",
    "RRT*":          "#E69F00",
    "Informed RRT*": "#009E73",
    "obstacle":      "#CC3311",
    "drone":         "#F0E442",
    "start":         "#44AA99",
    "goal":          "#AA4499",
    "trail":         "#BBBBBB",
    "bg":            "#FAFAFA",
    "panel":         "#F0F0F0",
}
MARKER_ALGO = {"A*": "o", "RRT*": "s", "Informed RRT*": "^"}

DENSITY_MAP = {
    "Low 6%":    DENSITY_LOW,
    "Medium 14%": DENSITY_MEDIUM,
    "High 22%":  DENSITY_HIGH,
}

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.env_seed    = 42
        self.algo        = "A*"
        self.family      = "random_clutter"
        self.density     = DENSITY_MEDIUM

        self.grid:  Optional[OccupancyGrid3D]     = None
        self.path:  Optional[List[Tuple[int,int,int]]] = None
        self.result: Optional[Dict]                = None

        self.flying   = False
        self.planning = False
        self._stop    = False

        # For compare mode
        self.compare_results: Dict[str, Dict] = {}

    def rebuild_grid(self):
        self.grid = make_environment(
            self.family, GRID_SIZE, self.env_seed,
            START, GOAL, self.density,
        )
        self.path   = None
        self.result = None
        self.compare_results = {}


# ---------------------------------------------------------------------------
# Main visualiser class
# ---------------------------------------------------------------------------
class DroneVisualiser:

    # ── setup ──────────────────────────────────────────────────────────────
    def __init__(self):
        self.state = AppState()
        self.state.rebuild_grid()

        # Animation
        self._anim_timer = None
        self._drone_pos  = np.array(START, float)
        self._trail: List[Tuple] = []
        self._fly_idx = 0

        self._build_figure()
        self._draw_scene()
        self._update_metrics()

    def _build_figure(self):
        self.fig = plt.figure(
            figsize=(16, 8),
            facecolor=C["bg"],
        )
        self.fig.suptitle(
            "CB-IRRT★  3D Drone Path Planning — Real-Time Visualiser",
            fontsize=14, fontweight="bold", y=0.98,
        )

        gs = gridspec.GridSpec(
            2, 3,
            left=0.01, right=0.99,
            top=0.93, bottom=0.02,
            wspace=0.05, hspace=0.35,
            width_ratios=[2.2, 1.0, 0.85],
        )

        # ── 3D scene ──────────────────────────────────────────────────────
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax3d.set_facecolor(C["bg"])

        # ── Metrics panel ─────────────────────────────────────────────────
        self.ax_met = self.fig.add_subplot(gs[0, 1])
        self.ax_met.set_facecolor(C["panel"])
        self.ax_met.axis("off")

        # ── Status bar ────────────────────────────────────────────────────
        self.ax_status = self.fig.add_subplot(gs[1, 1])
        self.ax_status.set_facecolor(C["panel"])
        self.ax_status.axis("off")

        # ── Control panel ─────────────────────────────────────────────────
        self.ax_ctrl = self.fig.add_subplot(gs[:, 2])
        self.ax_ctrl.set_facecolor(C["panel"])
        self.ax_ctrl.axis("off")

        self._build_controls()

        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _build_controls(self):
        """Create all interactive widgets inside ax_ctrl."""
        # We use axes-fraction coordinates inside ax_ctrl
        # Buttons and radio buttons are placed via figure axes
        fig = self.fig

        def ctrl_ax(left, bottom, width, height):
            """Convert ax_ctrl-relative coords to figure coords."""
            p   = self.ax_ctrl.get_position()
            return fig.add_axes([
                p.x0 + left   * p.width,
                p.y0 + bottom * p.height,
                width  * p.width,
                height * p.height,
            ])

        # ── Algorithm selector ────────────────────────────────────────────
        self.ax_algo = ctrl_ax(0.05, 0.77, 0.90, 0.20)
        self.ax_algo.set_title("Algorithm", fontsize=9, pad=2)
        self.radio_algo = RadioButtons(
            self.ax_algo,
            ["A*", "RRT*", "Informed RRT*"],
            active=0,
        )
        self.radio_algo.on_clicked(self._on_algo_change)
        for lbl in self.radio_algo.labels:
            lbl.set_fontsize(9)

        # ── Environment selector ──────────────────────────────────────────
        self.ax_env = ctrl_ax(0.05, 0.49, 0.90, 0.26)
        self.ax_env.set_title("Environment", fontsize=9, pad=2)
        self.radio_env = RadioButtons(
            self.ax_env,
            [f.replace("_", " ").title() for f in ENV_FAMILIES],
            active=ENV_FAMILIES.index("random_clutter"),
        )
        self.radio_env.on_clicked(self._on_env_change)
        for lbl in self.radio_env.labels:
            lbl.set_fontsize(8)

        # ── Density selector ──────────────────────────────────────────────
        self.ax_dens = ctrl_ax(0.05, 0.33, 0.90, 0.14)
        self.ax_dens.set_title("Density", fontsize=9, pad=2)
        self.radio_dens = RadioButtons(
            self.ax_dens,
            list(DENSITY_MAP.keys()),
            active=1,
        )
        self.radio_dens.on_clicked(self._on_dens_change)
        for lbl in self.radio_dens.labels:
            lbl.set_fontsize(8)

        # ── Action buttons ────────────────────────────────────────────────
        bw, bh, gap = 0.42, 0.07, 0.03

        self.ax_btn_plan = ctrl_ax(0.05,  0.235, bw, bh)
        self.ax_btn_fly  = ctrl_ax(0.53,  0.235, bw, bh)
        self.ax_btn_stop = ctrl_ax(0.05,  0.14,  bw, bh)
        self.ax_btn_new  = ctrl_ax(0.53,  0.14,  bw, bh)
        self.ax_btn_cmp  = ctrl_ax(0.05,  0.045, 0.90, bh)

        self.btn_plan = Button(self.ax_btn_plan, "▶  Plan",  color="#AED6F1", hovercolor="#5DADE2")
        self.btn_fly  = Button(self.ax_btn_fly,  "✈  Fly",   color="#A9DFBF", hovercolor="#27AE60")
        self.btn_stop = Button(self.ax_btn_stop, "■  Stop",  color="#F1948A", hovercolor="#E74C3C")
        self.btn_new  = Button(self.ax_btn_new,  "⟳  New Env", color="#F9E79F", hovercolor="#F39C12")
        self.btn_cmp  = Button(self.ax_btn_cmp,  "⚖  Compare All 3",  color="#D7BDE2", hovercolor="#8E44AD")

        for btn in (self.btn_plan, self.btn_fly, self.btn_stop,
                    self.btn_new, self.btn_cmp):
            btn.label.set_fontsize(9)

        self.btn_plan.on_clicked(self._on_plan)
        self.btn_fly .on_clicked(self._on_fly)
        self.btn_stop.on_clicked(self._on_stop)
        self.btn_new .on_clicked(self._on_new_env)
        self.btn_cmp .on_clicked(self._on_compare)

    # ── 3D scene rendering ─────────────────────────────────────────────────
    def _draw_scene(self, drone_pos=None):
        ax = self.ax3d
        ax.cla()

        g = self.state.grid
        X, Y, Z = g.x_size, g.y_size, g.z_size

        # Axis config
        ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.set_zlim(0, Z)
        ax.set_xlabel("X (m)", fontsize=8, labelpad=2)
        ax.set_ylabel("Y (m)", fontsize=8, labelpad=2)
        ax.set_zlabel("Z / Altitude", fontsize=8, labelpad=2)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.tick_params(labelsize=7)

        # Obstacles
        obs = np.argwhere(g.grid == 1)
        if len(obs):
            ax.scatter(
                obs[:, 0], obs[:, 1], obs[:, 2],
                c=C["obstacle"], s=8, alpha=0.22,
                marker="s", depthshade=True,
            )

        # Planned path(s)
        if self.state.compare_results:
            for algo, res in self.state.compare_results.items():
                p = res.get("path", [])
                if p:
                    xs = [v[0] for v in p]
                    ys = [v[1] for v in p]
                    zs = [v[2] for v in p]
                    ax.plot(xs, ys, zs,
                            color=C[algo], linewidth=1.8,
                            label=f"{algo}  {res['path_length']:.1f}m",
                            alpha=0.85)
            ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

        elif self.state.path:
            p = self.state.path
            xs = [v[0] for v in p]; ys = [v[1] for v in p]; zs = [v[2] for v in p]
            ax.plot(xs, ys, zs,
                    color=C[self.state.algo], linewidth=2.2,
                    label=f"{self.state.algo}", alpha=0.9)

        # Trail
        if self._trail:
            tx = [v[0] for v in self._trail]
            ty = [v[1] for v in self._trail]
            tz = [v[2] for v in self._trail]
            ax.plot(tx, ty, tz, color=C["trail"], linewidth=1.2,
                    alpha=0.6, linestyle="--")

        # Drone marker
        dp = drone_pos if drone_pos is not None else np.array(START, float)
        ax.scatter(*dp, c=C["drone"], s=160, marker="D",
                   zorder=15, edgecolors="black", linewidths=0.8)

        # Start / goal
        ax.scatter(*START, c=C["start"], s=160, marker="^",
                   zorder=15, edgecolors="black", linewidths=0.8)
        ax.scatter(*GOAL,  c=C["goal"],  s=160, marker="*",
                   zorder=15, edgecolors="black", linewidths=0.8)

        # Title with state info
        d_label = {DENSITY_LOW:"6%", DENSITY_MEDIUM:"14%", DENSITY_HIGH:"22%"}
        fam_str = self.state.family.replace("_", " ").title()
        ax.set_title(
            f"{fam_str}  |  Density {d_label[self.state.density]}  "
            f"|  Seed {self.state.env_seed}",
            fontsize=9, pad=4,
        )

        self.fig.canvas.draw_idle()

    def _update_metrics(self, extra_text: str = ""):
        ax = self.ax_met
        ax.cla(); ax.axis("off")
        ax.set_facecolor(C["panel"])

        lines = ["── Path Metrics ──────────────────"]
        if self.state.result and self.state.result.get("found"):
            r   = self.state.result
            path = r.get("path", [])
            g    = self.state.grid
            m    = compute_all(path, g, self.state.algo)
            lines += [
                f"Algorithm    : {self.state.algo}",
                f"Path length  : {m['path_length']:.2f} m",
                f"Planning time: {r['time_ms']:.0f} ms",
                f"Nodes expl.  : {r['nodes_explored']}",
                f"Waypoints    : {m['n_waypoints']}",
                f"Clearance    : {m['clearance']:.2f} vox (min)",
                f"Mean clear.  : {m['mean_clearance']:.2f} vox",
                f"Smoothness   : {m['smoothness']:.3f} rad/wp",
                f"Mean κ       : {m['mean_curvature']:.5f} 1/m",
                f"Energy proxy : {m['energy_proxy']:.1f} J",
                f"Dynamic feas.: {'✓ Yes' if m['dynamic_feasible'] else '✗ No'}",
            ]
        elif self.state.compare_results:
            lines += ["Algorithm        Len(m)   Time(ms)"]
            lines += ["-" * 34]
            for algo, res in self.state.compare_results.items():
                if res.get("found"):
                    lines.append(
                        f"{algo:16s} {res['path_length']:6.2f}  {res['time_ms']:8.0f}"
                    )
                else:
                    lines.append(f"{algo:16s}  —— not found ——")
        else:
            lines += ["No path planned yet.",
                      "Click ▶ Plan to run the planner."]

        if extra_text:
            lines += ["", extra_text]

        for i, line in enumerate(lines):
            ax.text(0.04, 0.97 - i * 0.077, line,
                    transform=ax.transAxes,
                    fontsize=8, family="monospace",
                    va="top", color="#1a1a1a")

        # Status bar
        self.ax_status.cla()
        self.ax_status.axis("off")
        self.ax_status.set_facecolor(C["panel"])
        status = extra_text or "Ready."
        self.ax_status.text(
            0.5, 0.5, status,
            transform=self.ax_status.transAxes,
            ha="center", va="center",
            fontsize=9, color="#333333",
            wrap=True,
        )

        self.fig.canvas.draw_idle()

    # ── Button callbacks ───────────────────────────────────────────────────
    def _on_algo_change(self, label: str):
        self.state.algo = label
        self.state.compare_results = {}
        self._update_metrics()

    def _on_env_change(self, label: str):
        fam_lookup = {
            f.replace("_", " ").title(): f
            for f in ENV_FAMILIES
        }
        self.state.family = fam_lookup.get(label, "random_clutter")
        self.state.rebuild_grid()
        self._trail = []
        self._drone_pos = np.array(START, float)
        self._draw_scene()
        self._update_metrics("New environment loaded.")

    def _on_dens_change(self, label: str):
        self.state.density = DENSITY_MAP.get(label, DENSITY_MEDIUM)
        self.state.rebuild_grid()
        self._trail = []
        self._drone_pos = np.array(START, float)
        self._draw_scene()
        self._update_metrics("Density changed — environment regenerated.")

    def _on_new_env(self, event):
        if self.state.flying:
            return
        self.state.env_seed += 1
        self.state.rebuild_grid()
        self._trail = []
        self._drone_pos = np.array(START, float)
        self._draw_scene()
        self._update_metrics(f"New environment — seed {self.state.env_seed}")

    def _on_plan(self, event):
        if self.state.planning or self.state.flying:
            return
        self.state.compare_results = {}
        self._trail = []
        self._drone_pos = np.array(START, float)
        self._update_metrics("Planning…  please wait.")
        # Run in a thread so the GUI stays responsive
        t = threading.Thread(target=self._run_plan, daemon=True)
        t.start()

    def _run_plan(self):
        self.state.planning = True
        self.state.path     = None
        self.state.result   = None
        try:
            g = self.state.grid
            planner = self._make_planner(self.state.algo, g)
            result  = planner.plan(START, GOAL)
            self.state.result = result
            self.state.path   = result.get("path", [])
            status = (
                f"✓ {self.state.algo} found path in {result['time_ms']:.0f} ms  "
                f"({result['path_length']:.2f} m)"
                if result["found"]
                else f"✗ {self.state.algo} failed to find a path."
            )
        except Exception as exc:
            status = f"Error: {exc}"
        finally:
            self.state.planning = False

        # Back on main thread for drawing
        self._schedule(lambda: (self._draw_scene(), self._update_metrics(status)))

    def _on_fly(self, event):
        if self.state.flying or not self.state.path:
            if not self.state.path:
                self._update_metrics("Plan a path first (click ▶ Plan).")
            return
        self.state._stop   = False
        self.state.flying  = True
        self._fly_idx      = 0
        self._trail        = []
        self._drone_pos    = np.array(START, float)
        self._fly_step()   # kick off animation loop

    def _fly_step(self):
        if self.state._stop or not self.state.flying:
            self.state.flying = False
            return
        path = self.state.path
        if self._fly_idx >= len(path):
            self.state.flying = False
            self._update_metrics("✈  Flight complete.")
            return

        wp                  = np.array(path[self._fly_idx], float)
        self._drone_pos     = wp
        self._trail.append(tuple(wp.astype(int)))
        self._fly_idx      += 1

        # Progress info
        pct    = self._fly_idx / len(path) * 100
        dist   = np.linalg.norm(wp - np.array(GOAL, float))
        status = (
            f"✈  Flying…  step {self._fly_idx}/{len(path)}  "
            f"({pct:.0f}%)  dist-to-goal={dist:.1f}m"
        )
        self._draw_scene(drone_pos=self._drone_pos)
        self._update_metrics(status)

        # Speed determined by slider-less approach: 80 ms per step
        self._anim_timer = self.fig.canvas.new_timer(interval=80)
        self._anim_timer.single_shot = True
        self._anim_timer.add_callback(self._fly_step)
        self._anim_timer.start()

    def _on_stop(self, event):
        self.state._stop  = True
        self.state.flying = False
        if self._anim_timer:
            try:
                self._anim_timer.stop()
            except Exception:
                pass
        self._update_metrics("■  Stopped.")

    def _on_compare(self, event):
        if self.state.planning or self.state.flying:
            return
        self._trail = []
        self._drone_pos = np.array(START, float)
        self._update_metrics("Comparing all 3 algorithms…  please wait.")
        t = threading.Thread(target=self._run_compare, daemon=True)
        t.start()

    def _run_compare(self):
        self.state.planning = True
        results: Dict[str, Dict] = {}
        g = self.state.grid
        for algo in ["A*", "RRT*", "Informed RRT*"]:
            try:
                planner = self._make_planner(algo, g)
                r = planner.plan(START, GOAL)
                results[algo] = r
            except Exception as exc:
                results[algo] = {"found": False, "path": [], "time_ms": 0,
                                 "path_length": 0, "nodes_explored": 0}
        self.state.compare_results = results
        self.state.path = None
        self.state.planning = False
        self._schedule(lambda: (
            self._draw_scene(),
            self._update_metrics("⚖  Comparison complete.")
        ))

    # ── Planner factory ────────────────────────────────────────────────────
    @staticmethod
    def _make_planner(algo: str, grid):
        if algo == "A*":
            return AStarPlanner(grid)
        if algo == "RRT*":
            return RRTStarPlanner(grid, max_iter=2000, seed=7919)
        if algo == "Informed RRT*":
            return InformedRRTStarPlanner(grid, max_iter=2000, seed=7919)
        raise ValueError(f"Unknown algorithm: {algo}")

    # ── Threading helper ───────────────────────────────────────────────────
    def _schedule(self, fn):
        """Call fn on the next GUI refresh (thread-safe)."""
        timer = self.fig.canvas.new_timer(interval=10)
        timer.single_shot = True
        timer.add_callback(fn)
        timer.start()

    def _on_close(self, event):
        self.state._stop  = True
        self.state.flying = False

    # ── Entry point ────────────────────────────────────────────────────────
    def run(self):
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  CB-IRRT*  Real-Time 3D Drone Path Planning Visualiser")
    print("  Author: Medisetti Renukeswar")
    print("=" * 60)
    print("\nStarting visualiser window…")
    print("Close the window to exit.\n")

    app = DroneVisualiser()
    app.run()
