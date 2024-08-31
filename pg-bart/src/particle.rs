use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

use ndarray::Array2;
use rand::{thread_rng, Rng};

use crate::data::Matrix;

use crate::{sampler::PgBartState, tree::DecisionTree};

/// Particle parameters
#[derive(Debug)]
pub struct ParticleParams {
    n_points: usize,
    n_features: usize,
    k_factor: f64,
}

impl ParticleParams {
    pub fn new(n_points: usize, n_features: usize, k_factor: f64) -> Self {
        ParticleParams {
            n_points,
            n_features,
            k_factor,
        }
    }

    pub fn with_new_kf(&self, k_factor: f64) -> Self {
        ParticleParams {
            n_points: self.n_points,
            n_features: self.n_features,
            k_factor,
        }
    }
}

/// SampleIndices tracks which training sample belong to node i
#[derive(Debug)]
pub struct SampleIndices {
    leaf_nodes: HashSet<usize>,       // Set of leaf node indices
    expansion_nodes: VecDeque<usize>, // Nodes that we still can expand
    data_indices: Vec<Vec<usize>>,    // Indices of points at each node
}

impl SampleIndices {
    fn new(num_samples: usize) -> Self {
        SampleIndices {
            leaf_nodes: HashSet::from([0]),
            expansion_nodes: VecDeque::from([0]),
            data_indices: vec![Vec::from_iter(0..num_samples)],
        }
    }

    fn add_index(&mut self, idx: usize, data_rows: Vec<usize>) {
        self.leaf_nodes.insert(idx);
        self.expansion_nodes.push_back(idx);
        if idx >= self.data_indices.len() {
            self.data_indices.resize(idx + 1, Vec::new())
        }
        self.data_indices[idx] = data_rows;
    }

    fn remove_index(&mut self, idx: usize) {
        self.leaf_nodes.remove(&idx);
        self.data_indices[idx].clear();
    }

    fn is_empty(&self) -> bool {
        self.expansion_nodes.is_empty()
    }

    fn pop_expansion_index(&mut self) -> Option<usize> {
        self.expansion_nodes.pop_front()
    }
}

#[derive(Debug)]
pub struct Weight {
    log_w: f64,
    log_likelihood: f64,
}

impl Weight {
    fn new() -> Self {
        Weight {
            log_w: 0.0,
            log_likelihood: 0.0,
        }
    }

    // Sets the log-weight and log-likelihood of this particle to a fixed value
    pub fn reset(&mut self, log_likelihood: f64) {
        self.log_w = log_likelihood;
        self.log_likelihood = log_likelihood;
    }

    // Updates the log-weight of this particle and sets the log-likelohood to a new value
    pub fn update(&mut self, log_likelihood: f64) {
        self.log_w += log_likelihood - self.log_likelihood;
        self.log_likelihood = log_likelihood;
    }

    // --- Getters ---
    pub fn log_w(&self) -> f64 {
        self.log_w
    }

    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    // --- Setters ---
    pub fn set_log_w(&mut self, log_w: f64) {
        self.log_w = log_w;
    }
}

/// A Particle wraps DecisionTree along with the SampleIndices of the
/// training samples that land in node i, and the Weight (log-likelihood)
/// of the Particle
#[derive(Debug)]
pub struct Particle {
    params: ParticleParams,
    tree: DecisionTree,
    indices: SampleIndices,
    weight: Weight,
}

impl Particle {
    // Creates a new Particle
    pub fn new(params: ParticleParams, init_value: f64, num_samples: usize) -> Self {
        let mut tree = DecisionTree::new();

        let indices = SampleIndices::new(num_samples);
        let weight = Weight::new();

        Particle {
            params,
            tree,
            indices,
            weight,
        }
    }

    pub fn grow(&mut self, X: &Array2<f64>, state: &PgBartState) -> bool {
        if let Some(node_index) = self.indices.pop_expansion_index() {
            if !state
                .probabilities
                .sample_expand_flag(self.tree.node_depth(node_index))
            {
                return false;
            }
        }

        let samples = &self.indices.data_indices[node_index];
        let feature = state.probabilities.sample_split_index();
        let feature_values: Vec<f64> = samples.iter().map(|&i| X[[i, feature]]).collect();
    }
}
