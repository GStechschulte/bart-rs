import numpy as np
import pymc as pm
import pymc_bart_rs as pmb
import pytest

from pymc_bart_rs.utils import _sample_posterior


@pytest.fixture(scope="module")
def bart_posterior():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(25, 2))
    Y = rng.normal(size=25)

    with pm.Model() as model:
        mu = pmb.BART("mu", X, Y, m=5)
        pm.Normal("y", mu=mu, sigma=1.0, observed=Y)
        pm.sample(
            draws=2,
            tune=1,
            chains=1,
            cores=1,
            step=[pmb.PGBART([mu], batch=(1.0, 1.0), num_particles=3)],
            random_seed=123,
            progressbar=False,
            compute_convergence_checks=False,
        )

    return X, mu


def test_all_trees_tree_dump_shape(bart_posterior):
    X, mu = bart_posterior
    all_trees = list(mu.owner.op.all_trees)

    assert len(all_trees) == 2
    assert len(all_trees[0]) == 1
    assert len(all_trees[0][0]) == 5

    tree = all_trees[0][0][0]
    lengths = {
        len(tree.split_feature),
        len(tree.split_value),
        len(tree.left_child),
        len(tree.right_child),
        len(tree.leaf_value),
    }
    assert len(lengths) == 1

    rng = np.random.default_rng(0)
    preds = _sample_posterior(all_trees, X=X, rng=rng, size=2, shape=1)
    assert preds.shape == (2, X.shape[0], 1)


def test_tree_dump_variable_inclusion_counts(bart_posterior):
    X, mu = bart_posterior
    all_trees = list(mu.owner.op.all_trees)
    n_features = X.shape[1]

    counts = np.zeros(n_features, dtype=int)
    for draw in all_trees:
        for tree in draw[0]:
            for feature in tree.split_feature:
                if feature >= 0:
                    counts[feature] += 1

    assert counts.shape == (n_features,)
    assert (counts >= 0).all()
