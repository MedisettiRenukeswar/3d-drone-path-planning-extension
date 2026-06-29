"""
benchmark/stats.py
------------------
Statistical analysis for the Monte Carlo benchmark.

Tests used
----------
Kruskal-Wallis H     omnibus test (are any distributions different?)
Mann-Whitney U       pairwise, Bonferroni-corrected
  rank-biserial r    effect size from Mann-Whitney
Cohen's d            parametric effect size
Cliff's delta        non-parametric effect size
Bootstrap 95 % CI    for means (n_boot = 2000 resamples)
Wilson score CI      for success rates (proportions)
Spearman ρ           density vs planning time correlation

All pairwise comparisons are Bonferroni-corrected:
    α_corrected = 0.05 / n_comparisons

Metrics analysed
----------------
  path_length, time_ms, smoothness,
  energy_proxy, clearance, mean_curvature

Author: Medisetti Renukeswar
"""
from __future__ import annotations

import csv
import json
import os
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

def kruskal_wallis(groups: Dict[str, List[float]]) -> Dict:
    """Kruskal-Wallis H omnibus test across all groups."""
    arrs = [np.array(v) for v in groups.values()]
    if len(arrs) < 2 or any(len(a) == 0 for a in arrs):
        return {}
    H, p = stats.kruskal(*arrs)
    return {
        "H":          float(H),
        "p":          float(p),
        "significant": p < 0.05,
        "groups":     list(groups.keys()),
        "n_per_group": [len(a) for a in arrs],
    }


def mann_whitney_pairwise(
    groups: Dict[str, List[float]],
    alpha:  float = 0.05,
) -> List[Dict]:
    """
    Pairwise Mann-Whitney U with Bonferroni correction.

    Works correctly with unequal group sizes.
    Returns list sorted by p-value (smallest first).
    """
    pairs  = list(combinations(groups.keys(), 2))
    n_comp = len(pairs)
    a_corr = alpha / max(n_comp, 1)
    results: List[Dict] = []

    for a, b in pairs:
        va, vb = np.array(groups[a]), np.array(groups[b])
        if len(va) < 3 or len(vb) < 3:
            results.append({
                "comparison": f"{a} vs {b}",
                "U": float("nan"), "p": 1.0,
                "rank_biserial_r": 0.0,
                "cohens_d": 0.0, "cliffs_delta": 0.0,
                "significant": False, "alpha_corrected": a_corr,
            })
            continue
        try:
            U, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
            r    = float(1 - 2 * U / (len(va) * len(vb)))
            d    = cohens_d(list(va), list(vb))
            cd   = cliffs_delta(list(va), list(vb))
            results.append({
                "comparison":       f"{a} vs {b}",
                "U":                float(U),
                "p":                float(p),
                "rank_biserial_r":  round(r, 4),
                "cohens_d":         round(d, 4),
                "cliffs_delta":     round(cd, 4),
                "significant":      p < a_corr,
                "alpha_corrected":  a_corr,
            })
        except Exception:
            results.append({
                "comparison": f"{a} vs {b}",
                "U": float("nan"), "p": 1.0,
                "rank_biserial_r": 0.0,
                "cohens_d": 0.0, "cliffs_delta": 0.0,
                "significant": False, "alpha_corrected": a_corr,
            })

    return sorted(results, key=lambda x: x["p"])


def cohens_d(a: List[float], b: List[float]) -> float:
    """Pooled-SD Cohen's d effect size."""
    a_, b_ = np.array(a), np.array(b)
    n1, n2 = len(a_), len(b_)
    if n1 + n2 < 4:
        return 0.0
    var1 = float(np.var(a_, ddof=1)) if n1 > 1 else 0.0
    var2 = float(np.var(b_, ddof=1)) if n2 > 1 else 0.0
    s_pool = float(np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)))
    if s_pool < 1e-12:
        return 0.0
    return float((np.mean(a_) - np.mean(b_)) / s_pool)


def cliffs_delta(a: List[float], b: List[float]) -> float:
    """
    Cliff's delta — non-parametric effect size in [−1, 1].

    Interpretation: |δ| < 0.147 negligible, < 0.330 small,
                    < 0.474 medium, ≥ 0.474 large.
    """
    a_, b_ = np.array(a), np.array(b)
    n1, n2 = len(a_), len(b_)
    if n1 == 0 or n2 == 0:
        return 0.0
    dom_a = int(np.sum(a_[:, None] > b_[None, :]))
    dom_b = int(np.sum(a_[:, None] < b_[None, :]))
    return float((dom_a - dom_b) / (n1 * n2))


def bootstrap_ci(
    data:   List[float],
    n_boot: int   = 2000,
    ci:     float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    arr = np.array(data)
    if len(arr) < 2:
        v = float(arr[0]) if len(arr) == 1 else 0.0
        return (v, v)
    rng   = np.random.default_rng(0)   # fixed seed for reproducibility
    boots = [
        float(np.mean(arr[rng.integers(0, len(arr), len(arr))]))
        for _ in range(n_boot)
    ]
    lo = (1 - ci) / 2
    return (float(np.percentile(boots, 100 * lo)),
            float(np.percentile(boots, 100 * (1 - lo))))


def wilson_ci(
    successes: int,
    n:         int,
    z:         float = 1.96,
) -> Tuple[float, float]:
    """Wilson score 95 % CI for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p   = successes / n
    d   = 1 + z ** 2 / n
    c   = (p + z ** 2 / (2 * n)) / d
    m   = z * float(np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def summary_stats(
    values: List[float],
    n_boot: int = 2000,
) -> Dict:
    """Mean, SD, median, IQR, bootstrap 95 % CI."""
    arr = np.array(values)
    n   = len(arr)
    if n == 0:
        return {}
    ci_lo, ci_hi = bootstrap_ci(list(arr), n_boot=n_boot)
    return {
        "n":       n,
        "mean":    round(float(np.mean(arr)), 4),
        "std":     round(float(np.std(arr, ddof=1)) if n > 1 else 0.0, 4),
        "median":  round(float(np.median(arr)), 4),
        "iqr":     round(float(np.subtract(*np.percentile(arr, [75, 25]))), 4),
        "q25":     round(float(np.percentile(arr, 25)), 4),
        "q75":     round(float(np.percentile(arr, 75)), 4),
        "min":     round(float(np.min(arr)), 4),
        "max":     round(float(np.max(arr)), 4),
        "ci95_lo": round(ci_lo, 4),
        "ci95_hi": round(ci_hi, 4),
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def extract(
    rows:       List[Dict],
    metric:     str,
    found_only: bool = True,
) -> List[float]:
    out = []
    for r in rows:
        if found_only and r.get("found") not in (True, "True", "1", 1):
            continue
        try:
            v = float(r[metric])
            if np.isfinite(v):
                out.append(v)
        except (ValueError, KeyError):
            pass
    return out


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(
    csv_path:   str,
    output_dir: str,
    n_boot:     int = 2000,
) -> Dict:
    """
    Run complete statistical analysis and save results to JSON.

    Parameters
    ----------
    csv_path   : path to all_runs.csv
    output_dir : directory to write stats_results.json
    n_boot     : bootstrap resamples (2000 is adequate for publication)

    Returns
    -------
    Nested dict with all test results.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows     = load_csv(csv_path)
    by_algo  = group_by(rows, "algorithm")

    results: Dict = {
        "kruskal_wallis":   {},
        "mann_whitney":     {},
        "pairwise_effects": {},
        "summary":          {},
        "success_rates":    {},
        "spearman":         {},
    }

    metrics_to_test = [
        "path_length", "time_ms", "smoothness",
        "energy_proxy", "clearance", "mean_curvature",
    ]

    for metric in metrics_to_test:
        groups = {
            a: extract(recs, metric)
            for a, recs in by_algo.items()
            if extract(recs, metric)
        }
        if len(groups) < 2:
            continue

        results["kruskal_wallis"][metric] = kruskal_wallis(groups)
        results["mann_whitney"][metric]   = mann_whitney_pairwise(groups)
        results["summary"][metric]        = {
            a: summary_stats(v, n_boot=n_boot)
            for a, v in groups.items()
        }

        peff: Dict = {}
        for a, b in combinations(groups.keys(), 2):
            k = f"{a} vs {b}"
            peff[k] = {
                "cohens_d":     round(cohens_d(groups[a], groups[b]), 4),
                "cliffs_delta": round(cliffs_delta(groups[a], groups[b]), 4),
            }
        results["pairwise_effects"][metric] = peff

    # Success rates with Wilson CI
    for algo, recs in by_algo.items():
        n     = len(recs)
        n_ok  = sum(
            1 for r in recs
            if r.get("found") in (True, "True", "1", 1)
        )
        lo, hi = wilson_ci(n_ok, n)
        results["success_rates"][algo] = {
            "n":         n,
            "successes": n_ok,
            "rate":      round(n_ok / max(n, 1), 4),
            "wilson_lo": round(lo, 4),
            "wilson_hi": round(hi, 4),
        }

    # Spearman: density vs planning time
    for algo, recs in by_algo.items():
        dens  = [float(r.get("density_actual", 0)) for r in recs
                 if r.get("density_actual")]
        times = extract(recs, "time_ms", found_only=False)
        if len(dens) == len(times) and len(dens) > 3:
            rho, p = stats.spearmanr(dens, times)
            results["spearman"][algo] = {
                "rho":         round(float(rho), 4),
                "p":           float(p),
                "significant": p < 0.05,
            }

    out_path = os.path.join(output_dir, "stats_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    _print_summary(results)
    print(f"\n  Statistics saved → {out_path}")
    return results


def _print_summary(results: Dict) -> None:
    print("\n" + "=" * 68)
    print("  STATISTICAL ANALYSIS SUMMARY")
    print("=" * 68)

    for metric in ["path_length", "time_ms", "energy_proxy"]:
        kw = results["kruskal_wallis"].get(metric, {})
        if kw:
            print(
                f"\nKruskal-Wallis ({metric}):"
                f"  H={kw.get('H',0):.2f}  "
                f"p={kw.get('p',1):.2e}  "
                f"sig={kw.get('significant')}"
            )
        pws = results["mann_whitney"].get(metric, [])
        if pws:
            a_corr = pws[0].get("alpha_corrected", 0.05)
            print(f"Mann-Whitney pairwise  (α_corr={a_corr:.4f}):")
            for r in pws:
                s = "✓" if r.get("significant") else "✗"
                print(
                    f"  {s} {r['comparison']:35s}"
                    f"  p={r['p']:.2e}"
                    f"  r={r['rank_biserial_r']:+.3f}"
                    f"  d={r['cohens_d']:+.3f}"
                    f"  δ={r['cliffs_delta']:+.3f}"
                )

    print("\nSuccess rates  (Wilson 95 % CI):")
    for algo, sr in results.get("success_rates", {}).items():
        print(
            f"  {algo:22s}  {sr['rate']:.1%}"
            f"  [{sr['wilson_lo']:.2%}, {sr['wilson_hi']:.2%}]"
            f"  n={sr['n']}"
        )

    print("\nPath length  (mean ± SD, bootstrap 95 % CI):")
    for algo, st in results.get("summary", {}).get("path_length", {}).items():
        if st:
            print(
                f"  {algo:22s}  "
                f"{st['mean']:.3f} ± {st['std']:.3f} m  "
                f"[{st['ci95_lo']:.3f}, {st['ci95_hi']:.3f}]  "
                f"n={st['n']}"
            )
    print("=" * 68)
