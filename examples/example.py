import os

import bart_rs
import numpy as np

def main():

    print(os.getcwd())

    for filename in os.listdir():
        print(filename)

    X = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]])
    print(X)
    print(X.shape)

    y = np.array([20.0, 21.0, 22.3])

    result_rs = bart_rs.shape(X)
    print(result_rs)

    # ----------------------

    coal = np.loadtxt("data/coal.txt")
    # discretize data
    years = int(coal.max() - coal.min())
    bins = years // 4
    hist, x_edges = np.histogram(coal, bins=bins)
    # compute the location of the centers of the discretized data
    x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
    # xdata needs to be 2D for BART
    x_data = x_centers[:, None]
    # express data as the rate number of disaster per year
    y_data = hist.astype(np.float64)

    # or

    np.random.seed(0)
    n = 50
    X = np.random.uniform(0, 10, n)
    Y = np.sin(X) + np.random.normal(0, 0.5, n)

    print(f"y.mean() = {Y.mean()}")

    state = bart_rs.initialize(
        X=X[..., None],
        y=Y,
        logp=10,
        alpha=0.50,
        n_trees=20,
        n_particles=5,
        kfactor=0.25,
        batch=(0.1, 0.1),
        split_prior=np.array([0.75, 0.25])
    )

    print(state)

    sum_trees = bart_rs.step(state, True)

    print(sum_trees)


if __name__ == "__main__":
    main()
