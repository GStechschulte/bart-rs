use ndarray::{Array1, Array2};
use std::collections::{HashSet, VecDeque};

use crate::{pgbart::PgBartState, tree::DecisionTree};

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

/// A Particle wraps a DecisionTree along with the SampleIndices of the
/// training samples that land in node i, and the Weight (log-likelihood)
/// of the Particle
#[derive(Debug)]
pub struct Particle {
    pub params: ParticleParams,
    pub tree: DecisionTree,
    pub indices: SampleIndices,
    pub weight: Weight,
}

impl Particle {
    pub fn new(params: ParticleParams, init_value: f64, num_samples: usize) -> Self {
        let tree = DecisionTree::new(init_value);
        let indices = SampleIndices::new(num_samples);
        let weight = Weight::new();

        Particle {
            params,
            tree,
            indices,
            weight,
        }
    }

    // TODO: Handle different `split_rules` and `response`
    pub fn grow(&mut self, X: &Array2<f64>, state: &PgBartState) -> bool {
        let node_index = match self.indices.pop_expansion_index() {
            Some(value) => value,
            None => {
                return false;
            }
        };

        let samples = &self.indices.data_indices[node_index];
        let feature = state.probabilities.sample_split_index();
        let feature_values: Vec<f64> = samples.iter().map(|&i| X[[i, feature]]).collect();

        if let Some(split_value) = state.probabilities.sample_split_value(&feature_values) {
            let (left_samples, right_samples): (Vec<usize>, Vec<usize>) = samples
                .iter()
                .partition(|&&i| X[[i, feature]] <= split_value);

            let left_value = state
                .probabilities
                .sample_leaf_value(0.0, self.params.k_factor);
            let right_value = state
                .probabilities
                .sample_leaf_value(0.0, self.params.k_factor);

            if let Ok((left_index, right_index)) =
                self.tree
                    .split_node(node_index, feature, split_value, left_value, right_value)
            {
                self.indices.add_index(left_index, left_samples);
                self.indices.add_index(right_index, right_samples);
                self.indices.remove_index(node_index);
                return true;
            }
        }
        false
    }

    pub fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(X.nrows());

        for (node_index, samples) in self.indices.data_indices.iter().enumerate() {
            if self.tree.is_leaf(node_index) {
                let leaf_value = self.tree.value[node_index];
                for &sample_index in samples {
                    predictions[sample_index] = leaf_value
                }
            }
        }

        predictions
    }

    pub fn finished(&self) -> bool {
        self.indices.is_empty()
    }
}
