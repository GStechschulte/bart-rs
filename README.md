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

### Traits

Traits, together with generics, are the bread and butter of Rust programming. Traits allow you to define shared functionality for Rust types.

### Errors

Rust groups errors into two main categories: (1) recoverable, and (2) non-recoverable. For errors that are recoverable, we most likely
want to report the problem to the user and retry the operation. Unrecoverable errors are always symptoms of bugs such as trying to access
a location beyond the end of an array.

In `bart-rs` if a function or method can fail, it will have a return type `Result<T, E>`. The Result type indicates
possible failure.

In particular, there are ... areas of possible failure.

- **Growing of particles.** During the growing of particles there can be particle growth `Ok(true)`, no particle growth, `Ok(false)`
and unsuccessful particle growth as a result of some error being raised `Err(ParticleError)`.
