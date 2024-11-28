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
            tune=200,
            draws=300,
            chains=4,
            step=[pmb.PGBART([mu], batch=tuple(args.batch), num_particles=num_particles)],
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
    ax.set_title("bart-rs: using shared step")
    plt.show()

    # sum_trees = step.astep(1)[0]
    # idx_sort = np.argsort(X.flatten())
    # plt.scatter(X.flatten()[idx_sort], Y[idx_sort])
    # plt.plot(X.flatten()[idx_sort], sum_trees[idx_sort], color="black")
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
