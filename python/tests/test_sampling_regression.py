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
