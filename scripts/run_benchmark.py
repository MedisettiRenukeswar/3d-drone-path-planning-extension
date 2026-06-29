#!/usr/bin/env python
"""
scripts/run_benchmark.py
------------------------
CLI wrapper for the Monte Carlo benchmark.

Usage
-----
  python scripts/run_benchmark.py                  # 30 seeds, all defaults
  python scripts/run_benchmark.py --seeds 5        # quick test (~15 min)
  python scripts/run_benchmark.py --seeds 30       # full benchmark (~2.5 hr)
  python scripts/run_benchmark.py --resume         # continue interrupted run
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.runner import run_benchmark, run_convergence_study

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",  type=int, default=30)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()

    os.makedirs("results/tables", exist_ok=True)

    print(f"\n[Benchmark] Running with {args.seeds} seeds, resume={args.resume}")
    run_benchmark(
        output_csv = "results/tables/all_runs.csv",
        n_seeds    = args.seeds,
        resume     = args.resume,
    )
    run_convergence_study(
        output_csv = "results/tables/convergence.csv",
        n_seeds    = min(args.seeds, 10),
    )
    print("\n[Benchmark] Done. Run: python scripts/run_analysis.py")

if __name__ == "__main__":
    main()
