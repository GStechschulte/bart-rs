import numpy as np
import pymc as pm
import pymc_bart_rs as pmb

from pymc_bart_rs.utils import _sample_posterior


def test_sampling_uses_multiple_features_and_reasonable_rmse():
    rng = np.random.default_rng(123)
    n = 60
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2] + rng.normal(scale=0.1, size=n)

    with pm.Model() as model:
        mu = pmb.BART("mu", X, y, m=10)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)
        pm.sample(
            draws=20,
            tune=10,
            chains=1,
            cores=1,
            step=[pmb.PGBART([mu], batch=(1.0, 1.0), num_particles=6)],
            random_seed=123,
            progressbar=False,
            compute_convergence_checks=False,
        )

    all_trees = list(mu.owner.op.all_trees)
    assert len(all_trees) > 0

    counts = np.zeros(X.shape[1], dtype=int)
    for draw in all_trees:
        for tree in draw[0]:
            for feature in tree.split_feature:
                if feature >= 0:
                    counts[feature] += 1

    # Ensure that splitting does not collapse onto a single feature
    assert (counts > 0).sum() >= 2

    preds = _sample_posterior(
        all_trees,
        X=X,
        rng=np.random.default_rng(321),
        size=len(all_trees),
        shape=1,
    )
    posterior_mean = preds.mean(axis=0).squeeze()
    rmse_rust = np.sqrt(np.mean((posterior_mean - y) ** 2))

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ref_preds = X @ coef
    rmse_ref = np.sqrt(np.mean((ref_preds - y) ** 2))

    # Keep the sampler within a reasonable factor of a simple least-squares baseline
    assert rmse_rust <= 2.0 * rmse_ref


def _run_three_feature_chain(random_seed: int):
    rng = np.random.default_rng(random_seed)
    n = 40
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2] + rng.normal(scale=0.1, size=n)

    with pm.Model() as model:
        mu = pmb.BART("mu", X, y, m=8)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)
        pm.sample(
            draws=8,
            tune=4,
            chains=1,
            cores=1,
            step=[pmb.PGBART([mu], batch=(1.0, 1.0), num_particles=6)],
            random_seed=random_seed,
            progressbar=False,
            compute_convergence_checks=False,
        )

    all_trees = list(mu.owner.op.all_trees)
    preds = _sample_posterior(
        all_trees,
        X=X,
        rng=np.random.default_rng(999),
        size=len(all_trees),
        shape=1,
    )
    posterior_mean = preds.mean(axis=0).squeeze()
    return posterior_mean


def test_sampling_is_deterministic_with_fixed_seed():
    first_run = _run_three_feature_chain(2024)
    second_run = _run_three_feature_chain(2024)
    np.testing.assert_allclose(first_run, second_run)


def test_variable_inclusion_balanced_and_low_rmse():
    rng = np.random.default_rng(321)
    n = 80
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2] + rng.normal(scale=0.1, size=n)

    with pm.Model() as model:
        mu = pmb.BART("mu", X, y, m=12)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)
        pm.sample(
            draws=30,
            tune=15,
            chains=1,
            cores=1,
            step=[pmb.PGBART([mu], batch=(1.0, 1.0), num_particles=8)],
            random_seed=321,
            progressbar=False,
            compute_convergence_checks=False,
        )

    all_trees = list(mu.owner.op.all_trees)
    counts = np.zeros(X.shape[1], dtype=int)
    for draw in all_trees:
        for tree in draw[0]:
            for feature in tree.split_feature:
                if feature >= 0:
                    counts[feature] += 1

    total_splits = counts.sum()
    assert total_splits > 0
    assert counts[0] < 0.7 * total_splits

    preds = _sample_posterior(
        all_trees,
        X=X,
        rng=np.random.default_rng(654),
        size=len(all_trees),
        shape=1,
    )
    posterior_mean = preds.mean(axis=0).squeeze()
    rmse_rust = np.sqrt(np.mean((posterior_mean - y) ** 2))

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ref_preds = X @ coef
    rmse_ref = np.sqrt(np.mean((ref_preds - y) ** 2))

    # Regression guardrail: keep RMSE close to the linear baseline
    assert rmse_rust <= 1.5 * rmse_ref
