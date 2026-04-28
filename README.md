# PyMC-BART-rs

High-performance Rust implementation of [PyMC-BART](https://github.com/pymc-devs/pymc-bart). This implementation provides an optimized Particle Gibbs BART (PGBART) sampler designed for performance and extensibility.

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
import pymc_bart as pmb

X, y = ... # Your data replaces "..."
with pm.Model() as model:
    bart = pmb.BART('bart', X, y)
    ...
    idata = pm.sample()
```
