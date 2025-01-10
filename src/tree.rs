//! A Binary Decision Tree is the core data structure for Bayesian Additive
//! Regression Trees (BART). The tree is implemented using an array (vector)
//! representation.

use core::fmt;
use std::cmp::Ordering;

use ndarray::{s, Array1, ArrayView1};

/// A `DecisionTree` is an array-based implementation of the binary decision tree.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTree {
    /// Stores the feature index for splitting at the i'th node.
    pub feature: Vec<usize>,
    /// Stores the threshold value for the i'th node split.
    pub threshold: Vec<f64>,
    /// Stores output values for the i'th node
    // pub value: Vec<f64>,
    pub values_buffer: Array1<f64>,
    pub n_dims: usize,
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
    pub fn new(init_value: &Array1<f64>) -> Self {
        Self {
            feature: vec![0],
            threshold: vec![0.0],
            values_buffer: init_value.clone(),
            n_dims: init_value.len(),
        }
    }

    #[inline(always)]
    pub fn node_values(&self, node_idx: usize) -> ArrayView1<f64> {
        let start = node_idx * self.n_dims;
        let end = start + self.n_dims;
        self.values_buffer.slice(ndarray::s![start..end])
    }

    /// Adds a new node to the `DecisionTree` and returns the location of _this_ node in the
    /// `feature` vector.
    pub fn add_node(&mut self, feature: usize, threshold: f64, value: &Array1<f64>) -> usize {
        let node_id = self.feature.len();

        self.feature.push(feature);
        self.threshold.push(threshold);

        // Extend values buffer
        let mut new_buffer = Array1::zeros(self.values_buffer.len() + self.n_dims);
        new_buffer
            .slice_mut(s![..self.values_buffer.len()])
            .assign(&self.values_buffer);
        new_buffer
            .slice_mut(s![self.values_buffer.len()..])
            .assign(value);
        self.values_buffer = new_buffer;

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
        left_value: &Array1<f64>,
        right_value: &Array1<f64>,
    ) -> Result<(usize, usize), TreeError> {
        if node_index >= self.feature.len() {
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

    /// Predicts values for a given input sample.
    /// Uses SIMD-friendly slice operations for performance.
    pub fn predict<'a>(&'a self, sample: &[f64], output: &'a mut [f64]) {
        debug_assert!(output.len() == self.n_dims);

        let mut node = 0;
        loop {
            if self.is_leaf(node) {
                let node_vals = self.node_values(node);
                output.copy_from_slice(node_vals.as_slice().unwrap());
                return;
            }

            let feature = self.feature[node];
            let threshold = self.threshold[node];

            node = match sample[feature].partial_cmp(&threshold).unwrap() {
                Ordering::Less => self.left_child(node).unwrap(),
                _ => self.right_child(node).unwrap(),
            };
        }
    }

    /// Returns the number of dimensions for leaf values
    #[inline(always)]
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }

    /// Implementation for serialization if needed
    pub fn values_as_slice(&self) -> &[f64] {
        self.values_buffer.as_slice().unwrap()
    }

    // pub fn predict(&self, sample: &[f64]) -> f64 {
    //     let mut node = 0;
    //     loop {
    //         if self.is_leaf(node) {
    //             return self.value[node];
    //         }
    //         let feature = self.feature[node];
    //         let threshold = self.threshold[node];
    //         node = match sample[feature].partial_cmp(&threshold).unwrap() {
    //             Ordering::Less => self.left_child(node).unwrap(),
    //             _ => self.right_child(node).unwrap(),
    //         };
    //     }
    // }
}
