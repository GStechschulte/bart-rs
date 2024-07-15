use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use crate::tree::Tree;

pub struct ParticleParams {
    n_points: usize,    // Number of rows in dataset
    n_features: usize,  // Number of features (covariates) in the dataset
    k_factor: f64,      // Standard deviation of noise added during leaf value sampling
}

struct Indices {
    leaf_nodes: BTreeSet<usize>,                // Set of leaf node indices
    expansion_nodes: VecDeque<usize>,           // Nodes still eligible for growing
    data_indices: BTreeMap<usize, Vec<usize>>,  // Indices of data points at each node
}

pub struct Weight {
    log_w: f64,             // Log weight of this particle
    log_likelihood: f64,    // Log-likelihood from the previous iteration
}

pub struct Particle {
    params: ParticleParams,
    tree: Tree,
    indices: Indices,
    weight: Weight,
}

impl Weight {
    fn new() -> Self {
        Weight { log_w: 0., log_likelihood: 0.,}
    }

    pub fn reset(&mut self, log_likelihood: f64) {
        self.log_w = log_likelihood;
        self.log_likelihood = log_likelihood;
    }

    // Update the log-weight of this particle and sets the log-likelihood to a new value
    pub fn update(&mut self, log_likelihood: f64) {
        let log_w = self.log_w + log_likelihood - self.log_likelihood;

        self.log_w = log_w;
        self.log_likelihood = log_likelihood;
    }

    pub fn log_w(&self) -> f64 {
        self.log_w
    }

    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    pub fn set_log_w(&mut self, log_w: f64) {
        self.log_w = log_w;
    }

}
