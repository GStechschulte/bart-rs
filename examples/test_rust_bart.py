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
        alpha = pm.Exponential("alpha", 1.)
        mu = pmb.BART("mu", X, np.log(Y), m=50)
        y = pm.NegativeBinomial("y", mu=pm.math.exp(mu), alpha=alpha, observed=Y)
        # step = pmb.PGBART([mu], batch=(0.1, 0.99), num_particles=10)
        idata_bikes = pm.sample(
            tune=1000,
            draws=1000,
            step=[pmb.PGBART([mu], batch=(0.1, 0.99), num_particles=10)],
            random_seed=RANDOM_SEED,
            )

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
    Y = hist / 4

    num_trees = 10
    num_particles = 5

    with pm.Model() as model_coal:

        mu = pmb.BART(
            "mu",
            X=X,
            Y=Y,
            m=num_trees,
            split_rules=["ContinuousSplit"],
            alpha=0.95,
            beta=2.0
        )

        sigma = pm.HalfNormal("sigma", 5.)

        y = pm.Normal("y", mu, sigma=sigma, observed=Y)

        idata = pm.sample(
            tune=300,
            draws=500,
            step=[pmb.PGBART([mu], batch=(0.1, 0.9), num_particles=num_particles)],
            random_seed=42,
            )

        # step = pmb.PGBART([mu], num_particles=10)

    # sum_trees = step.astep(1)

    _, ax = plt.subplots(nrows=1, ncols=1)

    rates = idata.posterior["mu"] / 4
    rate_mean = rates.mean(dim=["draw", "chain"]).to_numpy()
    ax.plot(x_centers, rate_mean, c="black", lw=3)
    # ax.plot(x_centers, y / 4, "k.", marker="x")
    az.plot_hdi(x_centers, rates, smooth=False)
    az.plot_hdi(x_centers, rates, hdi_prob=0.5, smooth=False, plot_kwargs={"alpha": 0})
    ax.plot(coal, np.zeros_like(coal) - 0.5, "k|")
    plt.ylim((0.0, 1.2))
    ax.set_xlabel("years")
    ax.set_ylabel("rate")
    ax.set_title("PyMC-BART: using shared step")
    plt.show()

    # sum_trees = step.astep(1)[0]
    # idx_sort = np.argsort(X.flatten())
    # plt.scatter(X.flatten()[idx_sort], Y[idx_sort])
    # plt.plot(X.flatten()[idx_sort], sum_trees[idx_sort], color="black")
    # plt.show()

if __name__ == "__main__":
    # test_bikes()
    test_coal()
