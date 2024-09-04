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

    bart_rs.initialize(X, y)

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

    bart_rs.initialize_bart(
        X=x_data,
        y=y_data,
        logp=10,
        alpha=0.50,
        n_trees=50,
        n_particles=10,
        kfactor=0.25,
        batch=(0.1, 0.1),
        split_prior=np.array([0.75, 0.25])
    )


if __name__ == "__main__":
    main()
