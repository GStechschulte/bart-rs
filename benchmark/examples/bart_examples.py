import argparse

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb

RANDOM_SEED = 8457


def test_propensity(args):
    nhefs_df = pd.read_csv(pm.get_data("nhefs.csv"))

    X = nhefs_df.astype("float64").copy()
    y = nhefs_df["outcome"].astype("float64")
    t = nhefs_df["trt"].astype("float64")
    X = X.drop(["trt", "outcome"], axis=1)

    coords = {"coeffs": list(X.columns), "obs": range(len(X))}
    with pm.Model(coords=coords) as model_ps:
        X_data = pm.MutableData("X", X)
        t_data = pm.MutableData("t", t)

        mu = pmb.BART("mu", X, t, m=args.trees)
        p = pm.Deterministic("p", pm.math.invprobit(mu))

        t_pred = pm.Bernoulli("t_pred", p=p, observed=t_data, dims="obs")

        idata = pm.sample(
            tune=args.tune,
            draws=args.draws,
            chains=4,
            step=[
                pmb.PGBART([mu], batch=tuple(args.batch), num_particles=args.particles)
            ],
            random_seed=RANDOM_SEED,
        )

    az.plot_forest(
        idata,
        var_names=["p"],
        coords={"p_dim_0": range(20)},
        figsize=(10, 13),
        combined=True,
        kind="ridgeplot",
        # model_names="BART",
        r_hat=True,
        ridgeplot_alpha=0.4,
    )
    plt.show()


def test_bikes(args):
    bikes = pd.read_csv(pm.get_data("bikes.csv"))
    X = bikes[["hour", "temperature", "humidity", "workingday"]]
    Y = bikes["count"]

    with pm.Model() as model_bikes:
        alpha = pm.Exponential("alpha", 1.0)
        mu = pmb.BART("mu", X, np.log(Y), m=args.trees)
        y = pm.NegativeBinomial("y", mu=pm.math.exp(mu), alpha=alpha, observed=Y)

        idata = pm.sample(
            chains=4,
            tune=args.tune,
            draws=args.draws,
            step=[
                pmb.PGBART([mu], batch=tuple(args.batch), num_particles=args.particles)
            ],
            random_seed=RANDOM_SEED,
        )

        posterior_predictive_oos_regression_train = pm.sample_posterior_predictive(
                trace=idata, random_seed=RANDOM_SEED
            )

        # step = pmb.PGBART([mu], batch=tuple(args.batch), num_particles=args.particles)

    print(idata["posterior"]["alpha"])
    pmb.plot_convergence(idata, var_name="mu")
    plt.show()

    az.plot_ppc(
        data=posterior_predictive_oos_regression_train, kind="cumulative", observed_rug=True
    )
    plt.show()

    # step.astep(1)
    # sum_trees, stats = step.astep(1)

    # for i in range(250):
    #     sum_trees, stats = step.astep(i)
    #     print(f"iter: {i}, time: {stats[0].get('time')}")


def test_coal(args):
    coal = np.loadtxt(pm.get_data("coal.csv"))

    # discretize data
    years = int(coal.max() - coal.min())
    bins = years // 4
    hist, x_edges = np.histogram(coal, bins=bins)
    # compute the location of the centers of the discretized data
    x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
    # xdata needs to be 2D for BART
    x_data = x_centers[:, None]
    # express data as the rate number of disaster per year
    y_data = hist

    with pm.Model() as model_coal:
        mu = pmb.BART(
            "mu",
            X=x_data,
            Y=np.log(y_data),
            m=args.trees
        )
        exp_mu = pm.Deterministic("exp_mu", pm.math.exp(mu))
        y_pred = pm.Poisson("y_pred", mu=exp_mu, observed=y_data)

        # idata = pm.sample(
        #     tune=args.tune,
        #     draws=args.draws,
        #     chains=4,
        #     step=[
        #         pmb.PGBART([mu], batch=tuple(args.batch), num_particles=args.particles)
        #     ],
        #     random_seed=RANDOM_SEED,
        # )
        step = pmb.PGBART([mu], batch=tuple(args.batch), num_particles=args.particles)

    # sum_trees, stats = step.astep(1)

    for i in range(500):
         sum_trees, stats = step.astep(i)
         print(f"iter: {i}, time: {stats[0].get('time')}")

    # _, ax = plt.subplots(figsize=(10, 6))
    # rates = idata.posterior["exp_mu"] / 4
    # rate_mean = rates.mean(dim=["draw", "chain"])
    # ax.plot(x_centers, rate_mean, "w", lw=3)
    # ax.plot(x_centers, y_data / 4, "k.")
    # az.plot_hdi(x_centers, rates, smooth=False)
    # az.plot_hdi(x_centers, rates, hdi_prob=0.5, smooth=False, plot_kwargs={"alpha": 0})
    # ax.plot(coal, np.zeros_like(coal) - 0.5, "k|")
    # ax.set_xlabel("years")
    # ax.set_ylabel("rate")
    # plt.show()

def test_asymmetric_laplace(args):
    bmi = pd.read_csv(pm.get_data("bmi.csv"))

    y = bmi.bmi.values
    X = bmi.age.values[:, None]
    y_stack = np.stack([bmi.bmi.values] * 3)
    quantiles = np.array([[0.1, 0.5, 0.9]]).T

    coords = {
        "quantiles": quantiles.flatten(),
        "n_obs": np.arange(X.shape[0])
    }

    with pm.Model(coords=coords) as model:
        # mu = pmb.BART("mu", X, y, m=50, dims=["quantiles", "n_obs"])
        mu = pmb.BART("mu", X, y, m=50)
        sigma = pm.HalfNormal("sigma", 5)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_stack)

        idata = pm.sample(
            tune=args.tune,
            draws=args.draws,
            chains=4,
            step=[
                pmb.PGBART([mu], batch=tuple(args.batch), num_particles=args.particles)
            ],
            random_seed=RANDOM_SEED,
        )


def main(args):

    if args.model == "coal":
        test_coal(args)
    elif args.model == "bikes":
        test_bikes(args)
    elif args.model == "propensity":
        test_propensity(args)
    elif args.model == "asymmetric":
        test_asymmetric_laplace(args)
    else:
        raise TypeError("Invalid model argument passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="coal", help="Model name")
    parser.add_argument("--trees", type=int, default=50, help="Number of trees")
    parser.add_argument("--particles", type=int, default=10, help="Number of particles")
    parser.add_argument("--tune", type=int, default=1000, help="Number of tuning steps")
    parser.add_argument("--draws", type=int, default=1000, help="Number of draws")
    parser.add_argument("--batch", nargs="+", default=(1.0, 1.0), type=float)
    args = parser.parse_args()
    main(args)
