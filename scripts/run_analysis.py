#!/usr/bin/env python
"""
scripts/run_analysis.py
-----------------------
Generate statistics and all publication figures from benchmark CSV.

Usage
-----
  python scripts/run_analysis.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.stats   import run_full_analysis
from benchmark.figures import generate_all

def main():
    csv_path  = "results/tables/all_runs.csv"
    conv_csv  = "results/tables/convergence.csv"
    stats_dir = "results/tables"
    figs_dir  = "results/figures"

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run benchmark first:")
        print("  python scripts/run_benchmark.py --seeds 5")
        sys.exit(1)

    print("\n[Analysis] Running statistical tests...")
    run_full_analysis(csv_path, stats_dir, n_boot=2000)

    print("\n[Analysis] Generating figures...")
    generate_all(
        csv_path    = csv_path,
        stats_path  = os.path.join(stats_dir, "stats_results.json"),
        conv_csv    = conv_csv if os.path.exists(conv_csv) else None,
        figures_dir = figs_dir,
    )
    print(f"\n[Analysis] Done.  Figures → {figs_dir}/")

if __name__ == "__main__":
    main()
