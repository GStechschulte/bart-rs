import argparse

import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

import bart_rs as pmb

warnings.simplefilter(action="ignore", category=FutureWarning)

RANDOM_SEED = 8457
RNG = np.random.RandomState(RANDOM_SEED)

def main():
    # coal = np.loadtxt(pm.get_data("coal.csv"))

    # # # discretize data
    # years = int(coal.max() - coal.min())
    # bins = years // 4
    # hist, x_edges = np.histogram(coal, bins=bins)
    # # compute the location of the centers of the discretized data
    # x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
    # # xdata needs to be 2D for BART
    # x_data = x_centers[:, None]
    # # express data as the rate number of disaster per year
    # y_data = hist / 4

    np.random.seed(0)
    n = 20_000
    X = np.random.uniform(0, 10, n)
    Y = np.sin(X) + np.random.normal(0, 0.5, n)
    data = pd.DataFrame(data={'Feature': X.flatten(), 'Y': Y})

    num_trees = 100
    num_particles = 10

    with pm.Model() as model_coal:
        mu_ = pmb.BART("mu", X=X[..., None], Y=Y, m=num_trees, alpha=0.95, beta=2.0)
        # mu = pm.Deterministic("μ", pm.math.exp(mu_))
        y = pm.Normal("y", mu_, sigma=1., observed=Y)
        # y_pred = pm.Poisson("y_pred", mu=mu, observed=y_data)

        step = pmb.PGBART([mu_], num_particles=num_particles, batch=(0.1, 0.1))

    sum_trees = step.astep(1)

    idx_sort = np.argsort(X.flatten())

    plt.scatter(X.flatten()[idx_sort], Y[idx_sort])
    plt.plot(X.flatten()[idx_sort], sum_trees[idx_sort], color="black")
    plt.show()

if __name__ == "__main__":
    main()
