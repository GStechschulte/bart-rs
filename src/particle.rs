use std::{collections::VecDeque, f64, rc::Rc, usize};

use numpy::ndarray::{Array, Ix1, Ix2};

// A Particle is a shared pointer to a tree
pub type Particle<const MAX_NODES: usize> = Rc<Tree<MAX_NODES>>;

pub type SplitVar = usize;
pub type SplitVal = f64;
pub type LeafVal = f64;
pub type LeafIdx = usize;

#[derive(Clone, Debug)]
pub struct Tree<const MAX_NODES: usize> {
    // pub split_var: Vec<SplitVar>,
    // pub split_val: Vec<SplitVal>,
    // pub leaf_val: Vec<LeafVal>,
    // pub leaf_indices: Vec<LeafIdx>,
    pub split_var: [SplitVar; MAX_NODES],
    pub split_val: [SplitVal; MAX_NODES],
    pub leaf_val: [LeafVal; MAX_NODES],
    pub leaf_indices: Vec<LeafIdx>,
    pub size: usize,
}

impl<const MAX_NODES: usize> Tree<MAX_NODES> {
    /// Create a new tree with just a root leaf node
    pub fn new(init_leaf_value: LeafVal, n_samples: usize) -> Self {
        let split_var = [usize::MAX; MAX_NODES];
        let split_val = [f64::NAN; MAX_NODES];
        let mut leaf_val = [0.0; MAX_NODES];

        // Set only root values
        leaf_val[0] = init_leaf_value;

        Self {
            split_var,
            split_val,
            leaf_val,
            leaf_indices: vec![0; n_samples],
            size: 1,
        }

        // let mut split_var = Vec::with_capacity(MAX_NODES);
        // let mut split_val = Vec::with_capacity(MAX_NODES);
        // let mut leaf_val = Vec::with_capacity(MAX_NODES);
        // let leaf_indices = vec![0; n_samples]; // Initially, all samples belong to root node

        // // Initialize the root as a leaf
        // split_var.push(usize::MAX);
        // split_val.push(f64::NAN);
        // leaf_val.push(init_leaf_value);

        // Self {
        //     split_var,
        //     split_val,
        //     leaf_val,
        //     leaf_indices,
        //     size: 1,
        // }
    }

    // Get the depth of a node in the binary tree
    pub fn get_depth(&self, node_idx: usize) -> usize {
        if node_idx == 0 {
            0
        } else {
            1 + self.get_depth((node_idx - 1) / 2)
        }
    }

    /// Check if a node is a leaf (no split variable assigned)
    pub fn is_leaf(&self, node_idx: usize) -> bool {
        node_idx < self.split_var.len() && self.split_var[node_idx] == usize::MAX
    }

    /// Get all leaf node indices
    pub fn get_leaf_indices(&self) -> Vec<usize> {
        (0..self.size).filter(|&i| self.is_leaf(i)).collect()
    }

    /// Calculate maximum depth from a node index in the tree
    pub fn max_depth_from_node(node_idx: usize) -> usize {
        if node_idx == 0 {
            0
        } else {
            1 + Self::max_depth_from_node((node_idx - 1) / 2)
        }
    }

    /// Calculate maximum allowable depth for this tree capacity
    pub fn max_allowable_depth() -> usize {
        let max_depth = ((MAX_NODES + 1) as f64).log2().floor() as usize;
        max_depth.saturating_sub(1)
    }

    /// Get data (samples) indices for a leaf node.
    pub fn get_leaf_samples(&self, leaf_idx: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.is_leaf(leaf_idx), "Node {} is not a leaf", leaf_idx);

        self.leaf_indices
            .iter()
            .enumerate()
            .filter_map(move |(sample_idx, &assigned_leaf)| {
                if assigned_leaf == leaf_idx {
                    Some(sample_idx)
                } else {
                    None
                }
            })
    }

    /// Splits (converts) a leaf node into an internal node and adds two new children leaf nodes.
    pub fn split_node(
        &mut self,
        leaf_idx: usize,
        split_var: usize,
        split_val: f64,
        left_val: f64,
        right_val: f64,
    ) {
        // Ensure we have space for two new children
        let left_child = 2 * leaf_idx + 1;
        let right_child = 2 * leaf_idx + 2;

        // With constant generics, we know the capacity at compile time
        assert!(
            right_child < MAX_NODES,
            "Tree mutation would exceed maximum capacity of {}",
            MAX_NODES
        );

        self.split_var[leaf_idx] = split_var;
        self.split_val[leaf_idx] = split_val;
        self.leaf_val[leaf_idx] = f64::NAN;

        self.split_var[left_child] = usize::MAX;
        self.split_val[left_child] = f64::NAN;
        self.leaf_val[left_child] = left_val;

        self.split_var[right_child] = usize::MAX;
        self.split_val[right_child] = f64::NAN;
        self.leaf_val[right_child] = right_val;

        self.size = self.size.max(right_child + 1);

        // NOTE: Old implementation below

        // self.split_var[leaf_idx] = split_var;
        // self.split_val[leaf_idx] = split_val;
        // self.leaf_val[leaf_idx] = f64::NAN; // Leaf node becomes an internal node

        // // Left child
        // self.split_var.push(usize::MAX);
        // self.split_val.push(f64::MAX);
        // self.leaf_val.push(left_val);

        // // Right child
        // self.split_var.push(usize::MAX);
        // self.split_val.push(f64::MAX);
        // self.leaf_val.push(right_val);

        // self.size = self.size.max(right_child + 1);
    }

    /// Updates leaf assignments with context after a split
    ///
    /// Updating of a Tree's `leaf_indices` occurs after a mutation (converting a leaf node to
    /// and internal node) as the samples that belonged (fell in) to the parent leaf get
    /// distributed to the two new child leaves.
    pub fn update_leaf_assignments(
        &mut self,
        split_node_idx: usize,
        split_var: usize,
        split_val: f64,
        affected_samples: &[usize],
        x_data: &Array<f64, Ix2>,
    ) {
        let base_child = 2 * split_node_idx + 1; // Left child

        // Update assignments for samples that were in the split node using bit manipulation
        for &sample_idx in affected_samples {
            let sample_val = x_data[[sample_idx, split_var]];
            let child_offset = (sample_val >= split_val) as usize;
            self.leaf_indices[sample_idx] = base_child + child_offset;
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        todo!("Not implemented")
    }
}

/// Predict interface for computing predictions using an array-based binary decision tree.
///
/// A distinction is made between predicting on training and test data. When computing predictions
/// on training data, the `leaf_indices` vector effectively serves as a lookup to determine the leaf
/// value for this data sample. When computing predictions on test data (new unseen data), one needs
/// to perform a tree traversal to compute which leaf node a data sample falls into.
pub trait Predict {
    fn predict_training(&self) -> Array<f64, Ix1>;
    // fn predict_single_test(&self, data: &[f64]) -> f64;
    // Computes predictions for two-dimensional data using the binary decision tree.
    //
    // Takes a two-dimensional design matrix and returns a one-dimensional array of predictions.
    // For each leaf node in the tree, assigns that node's leaf value to all data samples that
    // fall into that node.
    fn predict_batch_test(&self, data: &Array<f64, Ix2>) -> Array<f64, Ix1>;
}

impl<const MAX_NODES: usize> Predict for Tree<MAX_NODES> {
    fn predict_training(&self) -> Array<f64, Ix1> {
        self.leaf_indices
            .iter()
            .map(|&leaf_idx| self.leaf_val[leaf_idx])
            .collect()
    }

    /// Predict on test data by traversing the tree
    fn predict_batch_test(&self, data: &Array<f64, Ix2>) -> Array<f64, Ix1> {
        let mut predictions = Array::zeros(data.nrows());

        for (sample_idx, sample) in data.outer_iter().enumerate() {
            let mut node_idx = 0;

            // Traverse tree until we reach a leaf
            while node_idx < self.size && !self.is_leaf(node_idx) {
                let split_var = self.split_var[node_idx];
                let split_val = self.split_val[node_idx];

                // TODO: Update to arithmetic
                if sample[split_var] < split_val {
                    node_idx = 2 * node_idx + 1; // Left child
                } else {
                    node_idx = 2 * node_idx + 2; // Right child
                }

                // Safety check to prevent infinite loops
                if node_idx >= MAX_NODES {
                    break;
                }
            }

            // Assign leaf value
            if node_idx < self.size && self.is_leaf(node_idx) {
                predictions[sample_idx] = self.leaf_val[node_idx];
            }
        }

        predictions
    }
}
