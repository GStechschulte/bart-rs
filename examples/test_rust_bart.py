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


def test_bikes():
    bikes = pd.read_csv(pm.get_data("bikes.csv"))

    features = ["hour", "temperature", "humidity", "workingday"]

    X = bikes[features]
    Y = bikes["count"]

    with pm.Model() as model_bikes:
        # α = pm.Exponential("α", 1)
        μ = pmb.BART("μ", X, np.log(Y), m=50)
        y = pm.NegativeBinomial("y", mu=pm.math.exp(μ), alpha=1., observed=Y)
        idata_bikes = pm.sample(compute_convergence_checks=False, random_seed=RANDOM_SEED)


def test_coal():
    coal = np.loadtxt("/Users/gabestechschulte/Documents/repos/BART/experiments/coal.csv")

    # discretize data
    years = int(coal.max() - coal.min())
    bins = years // 4
    hist, x_edges = np.histogram(coal, bins=bins)
    # compute the location of the centers of the discrete data
    x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
    # xdata needs to be 2D for BART
    X = x_centers[:, None]
    # express data as the rate number of disaster per year
    y = hist / 4

    num_trees = 5
    num_particles = 3

    with pm.Model() as model_coal:

        mu = pmb.BART(
            "mu",
            X=X,
            Y=y,
            m=num_trees,
            split_rules=["ContinuousSplit"],
            alpha=0.95,
            beta=2.0
        )

        y = pm.Normal("y", mu, sigma=1., observed=y)

        idata = pm.sample(
            tune=300,
            draws=500,
            step=[pmb.PGBART([mu], batch=(0.1, 0.9999), num_particles=num_particles)],
            random_seed=42,
            )

        # step = pmb.PGBART([mu], num_particles=10)

    y_hat = idata["posterior"]["mu"].mean(("chain", "draw")).to_numpy()
    std_hat = idata["posterior"]["mu"].std(("chain", "draw")).to_numpy()
    idx_sort = np.argsort(X.flatten())
    plt.scatter(X.flatten(), y.flatten())
    plt.plot(X.flatten()[idx_sort], y_hat[idx_sort], color="black")
    plt.fill_between(
        X.flatten()[idx_sort],
        y_hat[idx_sort] + std_hat[idx_sort] * 2,
        y_hat[idx_sort] - std_hat[idx_sort] * 2,
        color="grey",
        alpha=0.25
    )
    plt.show()

    # sum_trees = step.astep(1)
    # idx_sort = np.argsort(X.flatten())
    # plt.scatter(X.flatten()[idx_sort], Y[idx_sort])
    # plt.plot(X.flatten()[idx_sort], sum_trees[idx_sort], color="black")
    # plt.show()


if __name__ == "__main__":
    test_bikes()
