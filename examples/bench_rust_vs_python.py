"""Compare benchmark results from Rust and Python PGBART runs.

Reads JSON result files produced by bench_runner.py (one per implementation),
computes speedup metrics, and outputs a combined report.

Usage:
    python examples/bench_rust_vs_python.py results/rust.json results/python.json
    python examples/bench_rust_vs_python.py results/rust.json results/python.json --json
"""

from __future__ import annotations

import argparse
import json
import sys


def compare(rust: dict, python: dict) -> dict:
    """Build a comparison from two result dicts."""
    speedup_mean = python["mean_ms"] / rust["mean_ms"] if rust["mean_ms"] > 0 else float("inf")
    speedup_median = (
        python["median_ms"] / rust["median_ms"] if rust["median_ms"] > 0 else float("inf")
    )

    return {
        "model": rust["model"],
        "n_trees": rust["n_trees"],
        "n_particles": rust["n_particles"],
        "n_steps": rust["n_steps"],
        "warmup": rust["warmup"],
        "rust": {k: rust[k] for k in _METRIC_KEYS},
        "python": {k: python[k] for k in _METRIC_KEYS},
        "speedup": {
            "mean": round(speedup_mean, 2),
            "median": round(speedup_median, 2),
        },
    }


_METRIC_KEYS = ("mean_ms", "std_ms", "median_ms", "p25_ms", "p75_ms", "min_ms", "max_ms")


def print_table(result: dict) -> None:
    """Pretty-print a comparison table."""
    header = (
        f"Model: {result['model']}  |  "
        f"trees={result['n_trees']}  particles={result['n_particles']}  "
        f"steps={result['n_steps']}  warmup={result['warmup']}"
    )
    print()
    print(header)
    print("=" * len(header))
    print(f"{'Metric':<12} {'Rust (ms)':>12} {'Python (ms)':>12} {'Speedup':>10}")
    print("-" * 48)

    r, p = result["rust"], result["python"]
    for key in _METRIC_KEYS:
        label = key.replace("_ms", "").replace("_", " ")
        rv, pv = r[key], p[key]
        sp = pv / rv if rv > 0 else float("inf")
        print(f"{label:<12} {rv:>12.3f} {pv:>12.3f} {sp:>9.2f}x")

    print("-" * 48)
    print(f"{'MEAN SPEEDUP':<12} {'':>12} {'':>12} {result['speedup']['mean']:>9.2f}x")
    print(f"{'MED. SPEEDUP':<12} {'':>12} {'':>12} {result['speedup']['median']:>9.2f}x")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare PGBART benchmark results")
    parser.add_argument("rust_json", help="Path to Rust results JSON file")
    parser.add_argument("python_json", help="Path to Python results JSON file")
    parser.add_argument("--json", action="store_true", help="Output combined JSON instead of table")
    parser.add_argument(
        "-o", "--output", default=None, help="Write combined JSON to this file"
    )
    args = parser.parse_args()

    with open(args.rust_json) as f:
        rust = json.load(f)
    with open(args.python_json) as f:
        python = json.load(f)

    result = compare(rust, python)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"Combined results written to {args.output}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_table(result)


if __name__ == "__main__":
    main()
