# PyMC-BART-rs

Rust implementation of [PyMC-BART](https://github.com/pymc-devs/pymc-bart). PyMC-BART extends the [PyMC](https://github.com/pymc-devs/pymc) probabilistic programming framework to be able to define and solve models including a Bayesian Additive Regression Tree (BART) random variable. PyMC-BART also includes a few helpers function to aid with the interpretation of those models and perform variable selection.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Modifications](#modifications)

## Installation

PyMC-BART is available on PyPI with pre-built wheels for Linux (x86_64, aarch64), Windows (x64), and macOS (x86_64, aarch64). To install using `pip`

```bash
pip install pymc-bart-rs
```

## Usage

Get started by using PyMC-BART to set up a BART model

```python
import pymc as pm
import pymc_bart_rs as pmb

X, y = ... # Your data replaces "..."
with pm.Model() as model:
    bart = pmb.BART('bart', X, y)
    ...
    idata = pm.sample()
```

## Modifications

The core Particle Gibbs (PG) sampling algorithm for BART remains the same in this Rust implementation as the original Python implementation. What differs is the choice of data structure to represent the Binary Decision Tree.

A `DecisionTree` structure is implemented as a number of parallel arrays. The i-th element of each array holds information about node `i`. The zero'th node is the tree's root. Some of the arrays only apply to either leaves or split nodes. In this case, the values of the nodes of the other arrays are arbitrary. For example, `feature` and `threshold` arrays only apply to split nodes. The values for leaf nodes in these arrays are therefore arbitrary.
