# bart-rs

Rust implementation of [PyMC-BART](https://github.com/pymc-devs/pymc-bart).


## Usage

...

## Modifications

The core Particle Gibbs (PG) sampling algorithm for Bayesian Additive Regression Trees (BART) remains the same
in this Rust implementation. What differs is the choice of data structure to represent the Binary Decision Tree.

A `DecisionTree` structure is implemented as a number of parallel vectors. The i-th element of each vector holds
information about node `i`. Node 0 is the tree's root. Some of the arrays only apply to either leaves or split
nodes. In this case, the values of the nodes of the other vector is arbitrary. For example, `feature` and `threshold`
vectors only apply to split nodes. The values for leaf nodes in these arrays are therefore arbitrary.

## Design

In this section, the architecture of `bart-rs` is given.

TODO...

## Seeding RNGs

The implementation of BART utilizes randomness in the growing of trees. The `thread_rng` function from the `randr_distr` crate provides a thread-local random number generator that is automatically seeded by the operating system or environment, ensuring that it is unique for each thread and run of the program. Therefore, we do not explicitly set a specific seed, and expect different values, e.g. sampled values from a Normal distribution, each time the program is ran.
