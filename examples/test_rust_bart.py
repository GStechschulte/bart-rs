import argparse

import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

import bart_rs as pmb

RANDOM_SEED = 8457
RNG = np.random.RandomState(RANDOM_SEED)

def main():

    np.random.seed(0)
    n = 1_000
    X = np.random.uniform(0, 10, n)
    Y = np.sin(X) + np.random.normal(0, 0.5, n)
    data = pd.DataFrame(data={'Feature': X.flatten(), 'Y': Y})

    num_trees = 100
    num_particles = 10

    with pm.Model() as model_coal:

        mu_ = pmb.BART(
            "mu",
            X=X[..., None],
            Y=Y,
            m=num_trees,
            split_rules=["ContinuousSplit"],
            alpha=0.95,
            beta=2.0
        )

        y = pm.Normal("y", mu_, sigma=1., observed=Y)
        idata = pm.sample(
            tune=300,
            draws=500,
            step=[pmb.PGBART([mu_], batch=(0.1, 0.99), num_particles=num_particles)],
            random_seed=42,
            )

    # y_hat = idata["posterior"]["mu"].mean(("chain", "draw"))
    # std_hat = idata["posterior"]["mu"].std(("chain", "draw"))
    # idx_sort = np.argsort(X.flatten())
    # plt.scatter(X.flatten()[idx_sort], Y[idx_sort])
    # plt.plot(X.flatten()[idx_sort], y_hat[idx_sort], color="black")
    # plt.fill_between(
    #     X.flatten()[idx_sort],
    #     y_hat[idx_sort] + std_hat[idx_sort] * 2,
    #     y_hat[idx_sort] - std_hat[idx_sort] * 2,
    #     color="grey",
    #     alpha=0.25
    # )
    # plt.show()

    # sum_trees = step.astep(1)
    # idx_sort = np.argsort(X.flatten())
    # plt.scatter(X.flatten()[idx_sort], Y[idx_sort])
    # plt.plot(X.flatten()[idx_sort], sum_trees[idx_sort], color="black")
    # plt.show()


if __name__ == "__main__":
    main()
