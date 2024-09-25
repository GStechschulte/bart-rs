use ndarray::{s, Array1, Array2};
use std::collections::{HashSet, VecDeque};

use crate::{pgbart::PgBartState, tree::DecisionTree};

/// Particle parameters
#[derive(Debug)]
pub struct ParticleParams {
    n_points: usize,
    n_features: usize,
    leaf_sd: f64,
}

impl ParticleParams {
    pub fn new(n_points: usize, n_features: usize, leaf_sd: f64) -> Self {
        ParticleParams {
            n_points,
            n_features,
            leaf_sd,
        }
    }

    pub fn with_new_kf(&self, leaf_sd: f64) -> Self {
        ParticleParams {
            n_points: self.n_points,
            n_features: self.n_features,
            leaf_sd,
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
    /// Creates a new...
    ///
    /// `leaf_nodes` and `expansion_nodes` start from 0 as this is the
    /// index of the root node, i.e when creating a new Particle, only
    /// the root node is eligible to be grown.
    fn new(num_samples: usize) -> Self {
        SampleIndices {
            leaf_nodes: HashSet::from([0]),
            expansion_nodes: VecDeque::from([0]),
            data_indices: vec![Vec::from_iter(0..num_samples)],
        }
    }

    /// Adds the index of a leaf to be expanded.
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

/// A Particle wraps a decision tree along with fields for the paricle
/// parameters, indices of the observed samples that land in node i,
/// and the weight of the Particle
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
        println!("\nGrowing tree");
        println!("------------");

        println!("Start. expansion nodes: {:?}", self.indices.expansion_nodes);

        let node_index = match self.indices.pop_expansion_index() {
            Some(value) => value,
            None => {
                println!("Unable to grow leaf node: no expansion index available");
                return false;
            }
        };

        println!("Growing node_index: {}", node_index);

        let node_index_depth = self.tree.node_depth(node_index);

        println!("node_index_depth: {}", node_index_depth);

        if !state.tree_ops.sample_expand_flag(node_index_depth) {
            println!("Node not selected for expansion");
            return false;
        }

        let samples = &self.indices.data_indices[node_index];
        let feature = state.tree_ops.sample_split_index();
        println!("Splitting on feature: {}", feature);

        let feature_values: Vec<f64> = samples.iter().map(|&i| X[[i, feature]]).collect();
        println!("Available splitting values: {:?}", feature_values);

        let split_value = match state.tree_ops.sample_split_value(&feature_values) {
            Some(value) => value,
            None => {
                println!("No valid split value found");
                return false;
            }
        };

        let (left_samples, right_samples): (Vec<usize>, Vec<usize>) = samples
            .iter()
            .partition(|&&i| X[[i, feature]] <= split_value);

        if left_samples.is_empty() || right_samples.is_empty() {
            println!("Invalid split: one side is empty");
            self.indices.expansion_nodes.push_back(node_index);
            return false;
        }

        let (left_predictions, right_predictions): (Vec<f64>, Vec<f64>) = (
            left_samples.iter().map(|&i| state.predictions[i]).collect(),
            right_samples
                .iter()
                .map(|&i| state.predictions[i])
                .collect(),
        );

        let (left_obs, right_obs): (Vec<f64>, Vec<f64>) = (
            left_samples.iter().map(|&i| X[[i, feature]]).collect(),
            right_samples.iter().map(|&i| X[[i, feature]]).collect(),
        );

        println!("Left obs: {:?}", left_obs);
        println!("Right obs: {:?}", right_obs);

        // TODO: Use state.params.shape once implemented
        let shape = 1;

        println!("\nSampling leaf values");
        println!("----------------------");

        let left_value = state.tree_ops.sample_leaf_value(
            &left_predictions,
            &left_obs,
            state.params.n_trees,
            &state.params.leaf_sd,
            shape,
            &state.params.response,
        );
        let right_value = state.tree_ops.sample_leaf_value(
            &right_predictions,
            &right_obs,
            state.params.n_trees,
            &state.params.leaf_sd,
            shape,
            &state.params.response,
        );

        println!("Sampled left value: {}", left_value);
        println!("Sampled right value: {}", right_value);

        match self
            .tree
            .split_node(node_index, feature, split_value, left_value, right_value)
        {
            Ok((left_index, right_index)) => {
                self.indices.remove_index(node_index);
                self.indices.add_index(left_index, left_samples);
                self.indices.add_index(right_index, right_samples);
                println!("self.expansion_nodes: {:?}", self.indices.expansion_nodes);
                return true;
            }
            Err(e) => {
                println!("Failed to split node {}: {:?}", node_index, e);
                return false;
            }
        }
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

        println!("\nPredict");
        println!("----------------------");
        println!("particle predictions: {:?}", predictions);

        predictions
    }

    pub fn finished(&self) -> bool {
        self.indices.is_empty()
    }
}
