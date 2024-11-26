#![allow(non_snake_case)]

use crate::{
    pgbart::PgBartState,
    split_rules::{SplitRule, SplitRuleType},
    tree::DecisionTree,
};

use std::collections::{HashSet, VecDeque};

use ndarray::{Array1, Array2};

/// Particle parameters
#[derive(Debug)]
pub struct ParticleParams {
    n_points: usize,
    n_features: usize,
}

impl ParticleParams {
    pub fn new(n_points: usize, n_features: usize) -> Self {
        Self {
            n_points,
            n_features,
        }
    }
}

/// SampleIndices tracks which training sample belong to node `i`.
#[derive(Debug)]
pub struct SampleIndices {
    leaf_nodes: HashSet<usize>,       // Set of leaf node indices
    expansion_nodes: VecDeque<usize>, // Nodes that we still can expand
    data_indices: Vec<Vec<usize>>,    // Indices of points at each node
}

impl SampleIndices {
    /// Creates a new `SampleIndices` struct used to track the training
    /// samples that belong to node `i`.
    ///
    /// `leaf_nodes` and `expansion_nodes` start from 0 as this is the
    /// index of the root node, i.e when creating a new Particle, only
    /// the root node is eligible to be grown.
    fn new(num_samples: usize) -> Self {
        Self {
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
    pub log_w: f64,
    pub log_likelihood: f64,
}

impl Weight {
    fn new() -> Self {
        Self {
            log_w: 0.0,
            log_likelihood: 0.0,
        }
    }

    // Sets the log-weight and log-likelihood of this particle to a fixed value
    pub fn set(&mut self, log_likelihood: f64) {
        self.log_w = log_likelihood;
        self.log_likelihood = log_likelihood;
    }

    // Updates the log-weight of this particle and sets the log-likelohood to a new value
    pub fn update(&mut self, log_likelihood: f64) {
        self.log_w += log_likelihood - self.log_likelihood;
        self.log_likelihood = log_likelihood;
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

        Self {
            params,
            tree,
            indices,
            weight,
        }
    }

    // TODO: Handle different `response`
    pub fn grow(&mut self, X: &Array2<f64>, state: &PgBartState) -> bool {
        let node_index = match self.indices.pop_expansion_index() {
            Some(value) => value,
            None => {
                return false;
            }
        };

        let node_index_depth = self.tree.node_depth(node_index);

        if !state.tree_ops.sample_expand_flag(node_index_depth) {
            return false;
        }

        let samples = &self.indices.data_indices[node_index];
        let feature = state.tree_ops.sample_split_feature();
        // Select the rule to be used sample a split value from _this_ feature
        let rule = &state.params.split_rules[feature];

        let (left_samples, right_samples, split_value) = match rule {
            SplitRuleType::Continuous(continuous_rule) => {
                let feature_values: Vec<f64> = samples
                    .iter()
                    .map(|&i| X[[i, feature]])
                    .filter(|&x| x.is_finite())
                    .collect();

                if let Some(split_val) = continuous_rule.sample_split_value(&feature_values) {
                    let (left, right) = continuous_rule.divide(&feature_values, &split_val);
                    (left, right, split_val)
                } else {
                    return false;
                }
            }
            SplitRuleType::OneHot(one_hot_rule) => {
                let feature_values: Vec<i32> = samples
                    .iter()
                    .map(|&i| X[[i, feature]] as i32) // Explicit type cast to i32
                    .filter(|&x| x >= 0)
                    .collect();

                if let Some(split_val) = one_hot_rule.sample_split_value(&feature_values) {
                    let (left, right) = one_hot_rule.divide(&feature_values, &split_val);
                    (left, right, split_val as f64) // Convert i32 to f64 for consistency
                } else {
                    return false;
                }
            }
        };

        if left_samples.is_empty() || right_samples.is_empty() {
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

        // TODO: Use state.params.shape once implemented
        let shape = 1;

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

        match self
            .tree
            .split_node(node_index, feature, split_value, left_value, right_value)
        {
            Ok((left_index, right_index)) => {
                self.indices.remove_index(node_index);
                self.indices.add_index(left_index, left_samples);
                self.indices.add_index(right_index, right_samples);

                true
            }
            // TODO: Proper error handling
            Err(_e) => false,
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

        predictions
    }

    pub fn finished(&self) -> bool {
        self.indices.is_empty()
    }
}
