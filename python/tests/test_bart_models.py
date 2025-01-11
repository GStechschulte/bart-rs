import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb


NUM_TUNE = 300
NUM_DRAWS = 600
NUM_CHAINS = 4
BATCH_SIZE = (0.1, 0.1)
NUM_TREES = 50
NUM_PARTICLES = 10
RANDOM_SEED = 42


def test_bikes():
    bikes = pd.read_csv(pm.get_data("bikes.csv"))
    X = bikes[["hour", "temperature", "humidity", "workingday"]]
    Y = bikes["count"]

    with pm.Model() as model:
        alpha = pm.Exponential("alpha", 1.0)
        mu = pmb.BART("mu", X, np.log(Y), m=NUM_TREES)
        y = pm.NegativeBinomial("y", mu=pm.math.exp(mu), alpha=alpha, observed=Y)

        idata = pm.sample(
            tune=NUM_TUNE,
            draws=NUM_DRAWS,
            chains=NUM_CHAINS,
            step=[pmb.PGBART([mu], batch=BATCH_SIZE, num_particles=NUM_PARTICLES)],
            random_seed=RANDOM_SEED,
        )


def test_coal():
    coal = np.loadtxt(pm.get_data("coal.csv"))
    years = int(coal.max() - coal.min())
    bins = years // 4
    hist, x_edges = np.histogram(coal, bins=bins)
    x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
    x_data = x_centers[:, None]
    # Express data as the rate number of disaster per year
    y_data = hist

    with pm.Model() as model:
        mu = pmb.BART("mu", X=x_data, Y=np.log(y_data), m=NUM_TREES)
        exp_mu = pm.Deterministic("exp_mu", pm.math.exp(mu))
        y_pred = pm.Poisson("y_pred", mu=exp_mu, observed=y_data)

        idata = pm.sample(
            tune=NUM_TUNE,
            draws=NUM_DRAWS,
            chains=NUM_CHAINS,
            step=[pmb.PGBART([mu], batch=BATCH_SIZE, num_particles=NUM_PARTICLES)],
            random_seed=RANDOM_SEED,
        )
