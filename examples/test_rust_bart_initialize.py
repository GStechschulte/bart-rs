import os

from pathlib import Path

import numpy as np

from bart_rs.bart_rs import initialize, step

def main():

    np.random.seed(0)
    n = 50
    X = np.random.uniform(0, 10, n)
    Y = np.sin(X) + np.random.normal(0, 0.5, n)

    print(f"y.mean() = {Y.mean()}")
    print(f"X.shape: {np.array(X[..., None].shape[1])}")

    state = initialize(
        X=X[..., None],
        y=Y,
        logp=10,
        alpha=0.95,
        beta=2.0,
        split_prior=np.array([1.0]),
        response="constant",
        n_trees=20,
        n_particles=5,
        leaf_sd=0.25,
        batch=(0.1, 0.1)
    )

    print(state)

    sum_trees = step(state, True)
    print(sum_trees)


if __name__ == "__main__":
    main()
