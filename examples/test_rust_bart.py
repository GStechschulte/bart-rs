import argparse

import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import y1

import bart_rs as pmb

RANDOM_SEED = 8457
RNG = np.random.RandomState(RANDOM_SEED)


def test_cohort_retention():
    pass


def test_propensity(args):
    nhefs_df = pd.read_csv(pm.get_data("nhefs.csv"))

    X = nhefs_df.astype("float64").copy()
    y = nhefs_df["outcome"].astype("float64")
    t = nhefs_df["trt"].astype("float64")
    X = X.drop(["trt", "outcome"], axis=1)

    def make_propensity_model(X, t, bart=True, probit=True, samples=1000, m=50):
        coords = {"coeffs": list(X.columns), "obs": range(len(X))}
        with pm.Model(coords=coords) as model_ps:
            X_data = pm.MutableData("X", X)
            t_data = pm.MutableData("t", t)
            if bart:
                mu = pmb.BART("mu", X, t, m=m)
                if probit:
                    p = pm.Deterministic("p", pm.math.invprobit(mu))
                else:
                    p = pm.Deterministic("p", pm.math.invlogit(mu))
            else:
                b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
                mu = pm.math.dot(X_data, b)
                p = pm.Deterministic("p", pm.math.invlogit(mu))

            t_pred = pm.Bernoulli("t_pred", p=p, observed=t_data, dims="obs")

            idata = pm.sample(
                tune=1000,
                draws=1000,
                step=[pmb.PGBART([mu], batch=tuple(args.batch), num_particles=10)],
                random_seed=42,
            )
            # idata = pm.sample_prior_predictive()
            # idata.extend(pm.sample(samples, random_seed=105, idata_kwargs={"log_likelihood": True}))
            # idata.extend(pm.sample_posterior_predictive(idata))

        return model_ps, idata

    m_ps_logit, idata_logit = make_propensity_model(X, t, bart=True, samples=1000)


def test_bikes(args):
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
            step=[pmb.PGBART([mu], batch=tuple(args.batch), num_particles=10)],
            random_seed=RANDOM_SEED,
            )

    # Plot convergence diagnostics
    ax = pmb.plot_convergence(idata_bikes, var_name="mu")
    plt.show()

    # Plot variable importance
    pmb.plot_pdp(mu, X=X, Y=Y, grid=(2, 2), func=np.exp, var_discrete=[3])
    plt.show()


def test_coal(args):
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

    num_trees = 20
    num_particles = 10

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
            tune=100,
            draws=200,
            chains=1,
            step=[pmb.PGBART([mu], batch=tuple(args.batch), num_particles=num_particles)],
            random_seed=42,
            )

        # step = pmb.PGBART([mu], batch=tuple(args.batch), num_particles=num_particles)

    _, ax = plt.subplots(nrows=1, ncols=1)
    rates = idata.posterior["mu"] / 4
    rate_mean = rates.mean(dim=["draw", "chain"]).to_numpy()
    ax.plot(x_centers, rate_mean, c="black", lw=3)
    ax.scatter(x_centers, Y / 4, marker="x", color="black")
    az.plot_hdi(x_centers, rates, smooth=False)
    az.plot_hdi(x_centers, rates, hdi_prob=0.5, smooth=False, plot_kwargs={"alpha": 0})
    ax.plot(coal, np.zeros_like(coal) - 0.5, "k|")
    plt.ylim((0.0, 1.2))
    ax.set_xlabel("years")
    ax.set_ylabel("rate")
    ax.set_title("bart-rs")
    plt.show()

    # sum_trees, stats = step.astep(1)

    # leaf_std = []
    # num_draws = 500
    # draws = np.zeros((num_draws, X.shape[0]))
    # rnge = range(0, num_draws)
    # for iter in rnge:
    #     sum_trees, stats = step.astep(iter)
    #     draws[iter, :] = sum_trees
    #     leaf_std.append(stats[0].get("leaf_std"))

    # mean_sum_trees = np.mean(draws, axis=0)
    # std_sum_trees = np.std(draws, axis=0)

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # idx_sort = np.argsort(X.flatten())
    # ax[0].scatter(X.flatten()[idx_sort], Y[idx_sort])
    # ax[0].plot(X.flatten()[idx_sort], mean_sum_trees[idx_sort], color="black")
    # ax[0].fill_between(
    #     x=X.flatten()[idx_sort],
    #     y1=mean_sum_trees[idx_sort] + std_sum_trees[idx_sort],
    #     y2=mean_sum_trees[idx_sort] - std_sum_trees[idx_sort],
    #     color="grey",
    #     alpha=0.25
    # )
    # ax[1].plot(rnge, leaf_std)
    # ax[0].set_title("bart-rs sum of trees predictions")
    # ax[1].set_title("Leaf standard deviation")
    # plt.show()

def main(args):

    if args.model == "coal":
        test_coal(args)
    elif args.model == "bikes":
        test_bikes(args)
    elif args.model == "propensity":
        test_propensity(args)
    else:
        raise TypeError("Invalid model argument passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test bart-rs with different models")
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch", nargs="+", type=float)
    args = parser.parse_args()
    main(args)
