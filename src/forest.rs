use core::fmt;
use std::ops::Range;
use std::{cmp::Ordering, collections::VecDeque};

use ndarray::{Array1, Array2};
use pyo3::ffi::PyPreConfig_InitIsolatedConfig;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};

use crate::base::PgBartState;
use crate::split_rules::{ContinuousSplit, OneHotSplit, SplitRule, SplitRuleType};

/// A `DecisionTree` is an array-based implementation of the binary decision tree.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTree {
    /// Stores the feature index for splitting at the i'th node.
    pub feature: Vec<usize>,
    /// Stores the threshold value for the i'th node split.
    pub threshold: Vec<f64>,
    /// Stores output values for the i'th node
    pub value: Vec<f64>,
    /// Sample indices buffer
    pub leaf_sample_indices: Vec<Vec<usize>>,
    /// Start and end indices into `leaf_sample_indices` leaf buffer
    // pub leaf_offsets: Vec<(usize, usize)>,
    /// Stores index of nodes that can still be expanded
    pub expansion_nodes: VecDeque<usize>,
}

/// Represents errors related to binary decision tree operations.
#[derive(Debug)]
pub enum TreeError {
    /// When attempting to split a leaf node, if the node is not a leaf.
    NonLeafSplit,
    /// When attempting to split a leaf node, if the index is valid or not
    InvalidNodeIndex,
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TreeError::NonLeafSplit => write!(f, "Cannot split a non-leaf node"),
            TreeError::InvalidNodeIndex => write!(f, "Node index does not exist"),
        }
    }
}

impl DecisionTree {
    /// Creates a new `DecisionTree` with an initial value set as the root node.
    /// A decision tree is implemented as three parallel vectors.
    ///
    /// Using parallel vectors allows for a more cache-efficient implementation.
    /// Moreover, index accessing allows one to more easily avoid borrow checker
    /// issues related to classical recursive binary tree implementations.
    ///
    /// The `i-th` element of each vector holds information about node `i`. Node 0
    /// is the tree's root. Some of the vectors only apply to either leaves or
    /// split nodes. In this case, the values of the nodes of the other vectors is
    /// arbitrary. For example, `feature` and `threshold` vectors only apply to
    /// split nodes. The values for leaf nodes in these vectors are therefore
    /// arbitrary.
    pub fn new(init_value: f64, num_samples: usize, max_size: usize) -> Self {
        let mut feature: Vec<usize> = Vec::with_capacity(max_size);
        let mut threshold: Vec<f64> = Vec::with_capacity(max_size);
        let mut value: Vec<f64> = Vec::with_capacity(max_size);

        feature.push(0);
        threshold.push(0.0);
        value.push(init_value);

        Self {
            feature: feature,
            threshold: threshold,
            value: value,
            leaf_sample_indices: vec![Vec::from_iter(0..num_samples)],
            // leaf_offsets: Vec::new(),
            expansion_nodes: VecDeque::from([0]),
        }
    }

    /// Adds a new node to the `DecisionTree` and returns the location of _this_ node in the
    /// `feature` vector.
    pub fn add_node(&mut self, feature: usize, threshold: f64, value: f64) -> usize {
        let node_id = self.feature.len();
        self.feature.push(feature);
        self.threshold.push(threshold);
        self.value.push(value);
        node_id
    }

    /// Computes the left child index of _this_ node.
    pub fn left_child(&self, index: usize) -> Option<usize> {
        let left_index = index * 2 + 1;
        if left_index < self.feature.len() {
            Some(left_index)
        } else {
            None
        }
    }

    /// Computes the right child index of _this_ node.
    pub fn right_child(&self, index: usize) -> Option<usize> {
        let right_index = index * 2 + 2;
        if right_index < self.feature.len() {
            Some(right_index)
        } else {
            None
        }
    }

    /// Checks whether the passed index is a leaf node.
    ///
    /// An index is a leaf node if both of its potential children are outside
    /// the valid array bounds.
    pub fn is_leaf(&self, index: usize) -> bool {
        self.left_child(index).is_none() && self.right_child(index).is_none()
    }

    /// Computes the depth of _this_ node in the `DecisionTree`.
    #[inline]
    pub fn node_depth(&self, index: usize) -> usize {
        let mut depth = 0;
        let mut current_index = index;

        while current_index != 0 {
            depth += 1;
            current_index = (current_index - 1) / 2;
        }

        depth
    }

    /// Splits a leaf node into an internal node.
    pub fn split_node(
        &mut self,
        node_index: usize,
        feature: usize,
        threshold: f64,
        left_value: f64,
        right_value: f64,
    ) -> Result<(usize, usize), TreeError> {
        if node_index >= self.value.len() {
            return Err(TreeError::InvalidNodeIndex);
        }

        if !self.is_leaf(node_index) {
            return Err(TreeError::NonLeafSplit);
        }

        // Update the current node
        self.feature[node_index] = feature;
        self.threshold[node_index] = threshold;

        // Add new left and right leaf nodes
        let left_child_index = self.add_node(0, 0.0, left_value);
        let right_child_index = self.add_node(0, 0.0, right_value);

        Ok((left_child_index, right_child_index))
    }

    fn add_index(&mut self, idx: usize, data_rows: Vec<usize>) {
        self.expansion_nodes.push_back(idx);
        // Extend the buffer
        // self.leaf_sample_indices.extend(data_rows);
        if idx >= self.leaf_sample_indices.len() {
            self.leaf_sample_indices.resize(idx + 1, Vec::new());
        }
        self.leaf_sample_indices[idx] = data_rows;
        // println!("{:?}", self.leaf_sample_indices);
    }

    fn is_empty(&self) -> bool {
        self.expansion_nodes.is_empty()
    }

    fn pop_expansion_index(&mut self) -> Option<usize> {
        self.expansion_nodes.pop_front()
    }
}

pub trait Predict {
    fn predict(&self, X: &Array2<f64>) -> Array1<f64>;
}

/// Systematic traversal
impl Predict for DecisionTree {
    fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(X.nrows());

        for (node_index, samples) in self.leaf_sample_indices.iter().enumerate() {
            if self.is_leaf(node_index) {
                let leaf_value = self.value[node_index];
                for &sample_index in samples {
                    predictions[sample_index] = leaf_value
                }
            }
        }

        predictions
    }
}

#[derive(Debug)]
pub struct Resampler {
    cdf: Vec<f64>,
    indices: Vec<usize>,
    new_trees: Vec<DecisionTree>,
    new_weights: Vec<f64>,
    new_offsets: Vec<usize>,
}

#[derive(Debug)]
pub struct Forest {
    pub trees: Vec<DecisionTree>,
    pub weights: Vec<f64>,
    pub likelihoods: Vec<f64>,
    pub offsets: Vec<usize>,
    pub resample_buffer: Vec<usize>,
    pub indices_buffer: Vec<usize>,
    pub cdf_buffer: Vec<f64>,
}

impl Forest {
    pub fn new(n_particles: usize, n_samples: usize, init_value: f64, max_size: usize) -> Self {
        let trees: Vec<DecisionTree> = (0..n_particles)
            .map(|_| DecisionTree::new(init_value, n_samples, max_size))
            .collect();
        let weights: Vec<f64> = vec![0.0; n_particles];
        let likelihood: Vec<f64> = vec![0.0; n_particles];
        let offsets = (0..n_particles).collect();

        Self {
            trees: trees,
            weights: weights,
            likelihoods: likelihood,
            offsets: offsets,
            resample_buffer: Vec::with_capacity(n_particles),
            indices_buffer: Vec::with_capacity(n_particles - 1),
            cdf_buffer: Vec::with_capacity(n_particles),
        }
    }

    pub fn resample(&mut self) {
        let n = self.trees.len();
        if n <= 1 {
            return;
        }

        let skip = 0;

        // Build a cumulative distribution of the "resampleable" weights (the subset from index 1..)
        let sub_weights = &self.weights[skip..];
        // let mut cdf = Vec::with_capacity(n - skip);
        self.cdf_buffer.clear();
        let mut running_sum = 0.0;
        for &w in sub_weights {
            running_sum += w;
            self.cdf_buffer.push(running_sum);
        }

        // If all weights are zero or negative, we can just skip
        if running_sum <= 0.0 {
            // Or handle it however you prefer
            return;
        }

        // Systematic sampling for indices [1..n]
        let step = 1.0 / ((n - skip) as f64);
        let mut rng = rand::thread_rng();
        let start = rng.gen::<f64>() * step;
        let mut idx_cdf = 0;
        // let mut sampled_indices = Vec::with_capacity(n - skip);
        self.indices_buffer.clear();

        for i in 0..(n - skip) {
            let target = start + (i as f64) * step;
            while idx_cdf < (n - skip) && self.cdf_buffer[idx_cdf] < target {
                idx_cdf += 1;
            }
            // Because we're skipping the first tree in sub_weights and cdf,
            // the actual tree index is 1 + idx_cdf
            // sampled_indices.push(1 + idx_cdf);
            self.indices_buffer.push(idx_cdf);
        }

        // Create a final list of new indices, putting the old index 0 (the first tree)
        // as is, then the sampled indices for the rest
        self.resample_buffer.clear();
        self.resample_buffer.push(0); // Keep the first tree as is
        self.resample_buffer.extend(&self.indices_buffer);

        // Use a functional Rust style to recreate self.trees, self.weights, self.offsets
        self.trees = self
            .resample_buffer
            .iter()
            .map(|&idx| self.trees[idx].clone())
            .collect();

        self.weights = self
            .resample_buffer
            .iter()
            .map(|&idx| self.weights[idx])
            .collect();

        self.offsets = self
            .resample_buffer
            .iter()
            .map(|&idx| self.offsets[idx])
            .collect();
    }

    pub fn normalize_weights(&mut self) {
        // println!("--- normalize_weights ---");
        // println!("log weights: {:?}", self.weights);

        let max_log_weight = self
            .weights
            .iter()
            // .skip(1)
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // 2. Exponentiate shifted log-weights and sum them
        let exp_shifted: Vec<f64> = self
            .weights
            .iter()
            // .skip(1)
            .map(|&w| (w - max_log_weight).exp())
            .collect();

        let sum_exp: f64 = exp_shifted.iter().sum();

        self.weights
            .iter_mut()
            .zip(exp_shifted.iter())
            .for_each(|(w, &e)| *w = e / sum_exp);
    }

    pub fn has_expandable_nodes(&self) -> bool {
        // Check if any Particle tree has an expandable node
        self.trees
            .iter()
            .any(|tree| !tree.expansion_nodes.is_empty())
    }

    pub fn grow(&mut self, X: &Array2<f64>, base_preds: &Array1<f64>, state: &PgBartState) {
        for (i, (w, tree)) in self
            .weights
            .iter_mut()
            .zip(self.trees.iter_mut())
            .enumerate()
        // .skip(1)
        {
            if let Some(node_idx) = tree.pop_expansion_index() {
                let depth = tree.node_depth(node_idx);
                // println!("node: {:?}, depth: {}", node_idx, depth);

                // sample_expand_flag is an example function from your TreeSamplingOps
                if !state.tree_ops.sample_expand_flag(depth) {
                    // Put node back since we decided not to expand
                    // tree.add_index(node_idx);
                    continue;
                }
                let feature = state.tree_ops.sample_split_feature();
                let rule = &state.params.split_rules[feature];
                // println!("feature: {}", feature);

                // First loop uses all row indices
                // let samples = Vec::from_iter(0..X.nrows());
                let samples = &tree.leaf_sample_indices[node_idx];
                // println!("samples: {:?}", samples);

                let (left_samples, right_samples, split_value) = match rule {
                    SplitRuleType::Continuous(continuous_rule) => {
                        let feature_values: Vec<f64> = samples
                            .iter()
                            .map(|&i| X[[i, feature]])
                            .filter(|&x| x.is_finite())
                            .collect();

                        if let Some(split_val) = continuous_rule.sample_split_value(&feature_values)
                        {
                            let (left, right) = continuous_rule.divide(&feature_values, split_val);
                            (left, right, split_val)
                        } else {
                            // tree.add_index(node_idx);
                            continue;
                        }
                    }
                    SplitRuleType::OneHot(one_hot_rule) => {
                        let feature_values: Vec<i32> = samples
                            .iter()
                            .map(|&i| X[[i, feature]] as i32)
                            .filter(|&x| x >= 0)
                            .collect();

                        if let Some(split_val) = one_hot_rule.sample_split_value(&feature_values) {
                            let (left, right) = one_hot_rule.divide(&feature_values, split_val);
                            (left, right, split_val as f64)
                        } else {
                            // tree.add_index(node_idx);
                            continue;
                        }
                    }
                };

                let left_value = {
                    let predictions = left_samples.iter().map(|&i| state.predictions[i]);
                    let observations = left_samples.iter().map(|&i| X[[i, feature]]);
                    state.tree_ops.sample_leaf_value(
                        &predictions.collect::<Vec<_>>(),
                        &observations.collect::<Vec<_>>(),
                        state.params.n_trees,
                        &state.params.leaf_sd,
                        &state.params.n_dim,
                        &state.params.response,
                    )
                };

                let right_value = {
                    let predictions = right_samples.iter().map(|&i| state.predictions[i]);
                    let observations = right_samples.iter().map(|&i| X[[i, feature]]);
                    state.tree_ops.sample_leaf_value(
                        &predictions.collect::<Vec<_>>(),
                        &observations.collect::<Vec<_>>(),
                        state.params.n_trees,
                        &state.params.leaf_sd,
                        &state.params.n_dim,
                        &state.params.response,
                    )
                };

                // Attempt the split
                match tree.split_node(node_idx, feature, split_value, left_value, right_value) {
                    Ok((left_idx, right_idx)) => {
                        // Mark children as expandable
                        tree.add_index(left_idx, left_samples);
                        tree.add_index(right_idx, right_samples);
                        // Recompute weight after splitting if desired
                        let preds = base_preds + &tree.predict(X);
                        *w = state.data.evaluate_logp(preds);
                    }
                    Err(_) => {
                        // If error, push node_idx back
                        // tree.add_index(node_idx);
                        continue;
                    }
                }
            }
        }
    }
}
