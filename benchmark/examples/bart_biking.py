import argparse

import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb


RANDOM_SEED = 8457


def main(args):
    bikes = pd.read_csv(pm.get_data("bikes.csv"))
    X = bikes[["hour", "temperature", "humidity", "workingday"]]
    Y = bikes["count"]

    with pm.Model() as model_bikes:
        alpha = pm.Exponential("alpha", 1.0)
        mu = pmb.BART("mu", X, np.log(Y), m=args.trees)
        y = pm.NegativeBinomial("y", mu=pm.math.exp(mu), alpha=alpha, observed=Y)
        idata_bikes = pm.sample(
            tune=args.tune,
            draws=args.draws,
            step=[
                pmb.PGBART([mu], batch=tuple(args.batch), num_particles=args.particles)
            ],
            random_seed=RANDOM_SEED,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=50, help="Number of trees")
    parser.add_argument("--particles", type=int, default=20, help="Number of particles")
    parser.add_argument("--tune", type=int, default=1000, help="Number of tuning steps")
    parser.add_argument("--draws", type=int, default=1000, help="Number of draws")
    parser.add_argument("--batch", nargs="+", default="1.0 1.0", type=float)
    args = parser.parse_args()
    main(args)
