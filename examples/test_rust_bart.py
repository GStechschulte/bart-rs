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
    coal = np.loadtxt(pm.get_data("coal.csv"))

    # # discretize data
    years = int(coal.max() - coal.min())
    bins = years // 4
    hist, x_edges = np.histogram(coal, bins=bins)
    # compute the location of the centers of the discretized data
    x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
    # xdata needs to be 2D for BART
    x_data = x_centers[:, None]
    # express data as the rate number of disaster per year
    y_data = hist / 4

    # np.random.seed(0)
    # n = 50
    # X = np.random.uniform(0, 10, n)
    # Y = np.sin(X) + np.random.normal(0, 0.5, n)
    # data = pd.DataFrame(data={'Feature': X.flatten(), 'Y': Y})

    with pm.Model() as model_coal:
        mu_ = pmb.BART("mu", X=x_data, Y=y_data, m=50, alpha=0.95, beta=1.5)
        # mu = pm.Deterministic("μ", pm.math.exp(mu_))
        y = pm.Normal("y", mu_, sigma=1., observed=y_data)
        # y_pred = pm.Poisson("y_pred", mu=mu, observed=y_data)

        step = pmb.PGBART([mu_], num_particles=10)

    for _ in range(100):
        sum_trees = step.astep(_)

    # sum_trees = step.astep(1)
    # sum_trees = step.astep(1)
    # sum_trees = step.astep(1)
    # sum_trees = step.astep(1)
    # sum_trees = step.astep(1)

    plt.scatter(x_data, y_data)
    plt.plot(x_data, sum_trees)
    plt.show()

if __name__ == "__main__":
    main()
