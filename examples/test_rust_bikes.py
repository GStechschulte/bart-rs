from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import bart_rs as pmb

RANDOM_SEED = 8457

def main():

    num_trees = 50
    num_particles = 10

    bikes = pd.read_csv(pm.get_data("bikes.csv"))

    X = bikes[["hour", "temperature", "humidity", "workingday"]]
    X = bikes["hour"].values.reshape(-1, 1)
    Y = bikes["count"].values

    with pm.Model() as model_bikes:
        sigma = pm.HalfNormal("sigma", Y.std())
        mu = pmb.BART("mu", X, Y, m=num_trees)
        y = pm.Normal("y", mu, sigma, observed=Y)
        step = pmb.PGBART([mu], num_particles=num_particles, batch=(0.1, 0.1))
        # idata_bikes = pm.sample(random_seed=RANDOM_SEED)

    # y_hat = idata_bikes["posterior"]["mu"].mean(("chains", "draws")).to_numpy()
    sum_trees = step.astep(1)

    plt.scatter(X, Y, marker="x")
    plt.show(X, sum_trees)

if __name__ == "__main__":
    main()
