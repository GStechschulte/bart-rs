use std::{f64, rc::Rc, usize};

use numpy::ndarray::{Array, Ix1, Ix2};

// A Particle is a shared pointer to a tree
pub type Particle<const MAX_NODES: usize> = Rc<Tree<MAX_NODES>>;

pub type SplitVar = usize;
pub type SplitVal = f64;
pub type LeafVal = f64;
pub type LeafIdx = usize;

#[derive(Clone, Debug)]
pub struct Tree<const MAX_NODES: usize> {
    pub split_var: Vec<SplitVar>,
    pub split_val: Vec<SplitVal>,
    pub leaf_val: Vec<LeafVal>,
    pub leaf_indices: Vec<LeafIdx>,
    pub size: usize,
}

impl<const MAX_NODES: usize> Tree<MAX_NODES> {
    /// Create a new tree with just a root leaf node
    pub fn new(init_leaf_value: LeafVal, n_samples: usize) -> Self {
        let mut split_var = Vec::with_capacity(MAX_NODES);
        let mut split_val = Vec::with_capacity(MAX_NODES);
        let mut leaf_val = Vec::with_capacity(MAX_NODES);
        let leaf_indices = vec![0; n_samples];

        // Initialize the root as a leaf
        split_var.push(usize::MAX);
        split_val.push(f64::NAN);
        leaf_val.push(init_leaf_value);

        Self {
            split_var,
            split_val,
            leaf_val,
            leaf_indices,
            size: 1,
        }
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

    pub fn get_node_samples(&self, node_idx: usize, x_data: &Array<f64, Ix2>) -> Vec<usize> {
        (0..x_data.len()).collect()
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
        self.leaf_val[leaf_idx] = f64::NAN; // Leaf node becomes an internal node

        // Left child
        self.split_var.push(usize::MAX);
        self.split_val.push(f64::MAX);
        self.leaf_val.push(left_val);

        // Right child
        self.split_var.push(usize::MAX);
        self.split_val.push(f64::MAX);
        self.leaf_val.push(right_val);

        self.size = self.size.max(right_child + 1);
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        todo!("Not implemented")
    }
}
