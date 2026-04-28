"""Benchmark runner for a single PGBART implementation.

Called by the Makefile inside the appropriate virtual environment.
Outputs a JSON blob with per-step timing data to stdout.

Usage:
    python examples/bench_runner.py --model coal --trees 50 --particles 10 --steps 20 --warmup 5
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pymc as pm
import pymc_bart as pmb
from pymc_bart.pgbart import PGBART


def build_coal_model(n_trees: int, n_particles: int) -> PGBART:
    coal = np.loadtxt(pm.get_data("coal.csv"))
    years = int(coal.max() - coal.min())
    bins = years // 4
    hist, x_edges = np.histogram(coal, bins=bins)
    x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
    x_data = x_centers[:, None]
    y_data = hist.astype("float64")

    with pm.Model():
        mu = pmb.BART("mu", X=x_data, Y=np.log(y_data),
                       alpha=0.95, beta=2.0, m=n_trees)
        pm.Poisson("y_pred", mu=pm.math.exp(mu), observed=y_data)
        step = PGBART([mu], num_particles=n_particles)
    return step


def build_propensity_model(n_trees: int, n_particles: int) -> PGBART:
    import pandas as pd

    nhefs_df = pd.read_csv(pm.get_data("nhefs.csv"))
    X = nhefs_df.astype("float64").copy()
    t = nhefs_df["trt"].astype("float64")
    X = X.drop(["trt", "outcome"], axis=1)

    with pm.Model():
        mu = pmb.BART("mu", X, t, m=n_trees)
        p = pm.Deterministic("p", pm.math.invprobit(mu))
        pm.Bernoulli("t_pred", p=p, observed=t)
        step = PGBART([mu], num_particles=n_particles)
    return step


BUILDERS = {
    "coal": build_coal_model,
    "propensity": build_propensity_model,
}


def main():
    parser = argparse.ArgumentParser(description="Run PGBART benchmark")
    parser.add_argument("--model", default="coal", choices=list(BUILDERS))
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--particles", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    step = BUILDERS[args.model](args.trees, args.particles)

    # Warmup (discard)
    for i in range(args.warmup):
        step.astep(i)

    # Timed steps
    times = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        step.astep(args.warmup + i)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_arr = np.array(times)

    result = {
        "model": args.model,
        "n_trees": args.trees,
        "n_particles": args.particles,
        "n_steps": args.steps,
        "warmup": args.warmup,
        "times": times,
        "mean_ms": float(times_arr.mean() * 1000),
        "std_ms": float(times_arr.std() * 1000),
        "median_ms": float(np.median(times_arr) * 1000),
        "min_ms": float(times_arr.min() * 1000),
        "max_ms": float(times_arr.max() * 1000),
        "p25_ms": float(np.percentile(times_arr, 25) * 1000),
        "p75_ms": float(np.percentile(times_arr, 75) * 1000),
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
