use std::collections::TryReserveError;

use bumpalo::{collections::Vec as BumpVec, Bump};
use numpy::{ndarray::Array, Ix1, Ix2};
use rand::{rngs::SmallRng, Rng};

use crate::base::PgBartState;
use crate::response::ResponseStrategy;
use crate::split_rules::{SplitRule, SplitRules};

pub type SplitVariable = usize;
pub type SplitValue = f64;
pub type LeafValue = f64;
pub type LeafIndex = usize;

pub trait Predict {
    /// Computes predictions given a design matrix using the learned
    /// decision tree
    fn predict(&self, X: Array<f64, Ix2>) -> Array<f64, Ix1>;
}

// Core strategy traits
pub trait Update<S> {
    type Info;
    type Error;

    fn update(&self, rng: &mut impl Rng, state: S) -> Result<(S, Self::Info), Self::Error>;
}

pub struct TreeUpdate<Split, Response> {
    max_depth: usize,
    split_strategy: Split,
    response_strategy: Response,
}

// impl<Split, Response, S> Update for TreeUpdate<Split, Response>
// where
//     S: PgBartState,
//     Split: SplitRule,
//     Response: ResponseStrategy,
// {
//     type Info = TreeUpdateInfo;
//     type Error = TreeError;

//     fn update(&self, rng: &mut impl Rng, state: S) -> Result<(S, Self::Info), Self::Error> {
//         todo!("Tree growing logic using the SplitStrategy and ResponseStrategy go here...")
//     }
// }

/// A Forest owns the arena and passes a reference to it to a Tree.
#[derive(Debug)]
pub struct Tree<'arena, const MAX_DEPTH: usize> {
    split_var: BumpVec<'arena, SplitVariable>,
    split_value: BumpVec<'arena, SplitValue>,
    leaf_values: BumpVec<'arena, LeafValue>,
    leaf_indices: BumpVec<'arena, LeafIndex>,
}

impl<'arena, const MAX_DEPTH: usize> Tree<'arena, MAX_DEPTH> {
    fn stump(arena: &'arena Bump, init_leaf: LeafValue, n_samples: usize) -> Self {
        let max_leaf_nodes = 1 << (MAX_DEPTH + 1); // 2^(depth)
        let max_internal_nodes = max_leaf_nodes - 1;

        // A stump starts with one leaf node
        let mut leaf_values = BumpVec::with_capacity_in(max_leaf_nodes, arena);
        leaf_values.push(init_leaf);

        // All samples initially point to the first leaf (index = 0)
        let leaf_indices = BumpVec::from_iter_in((0..n_samples).map(|_| 0), arena);

        // Pre-allocate remaining vectors with enough capacity for a full tree to avoid
        // reallocations
        Self {
            split_var: BumpVec::with_capacity_in(max_internal_nodes, arena),
            split_value: BumpVec::with_capacity_in(max_internal_nodes, arena),
            leaf_values,
            leaf_indices,
        }
    }

    fn get_node_depth(&self, node_idx: usize) -> usize {
        if node_idx == 0 {
            return 0;
        }

        let mut depth = 0;
        let mut current = node_idx;
        // Parent of node i is at (i - 1 / 2)
        while current > 0 {
            current = (current - 1) / 2;
            depth += 1
        }
        depth
    }

    fn get_leaf_data_indices(&self, node_idx: usize) -> Vec<usize> {
        // TODO: Compute leaf indices without allocating?
        let mut indices = Vec::with_capacity(self.leaf_indices.len());
        let iter =
            self.leaf_indices
                .iter()
                .enumerate()
                .filter_map(|(sample_idx, &assigned_leaf)| {
                    if assigned_leaf == node_idx {
                        Some(sample_idx)
                    } else {
                        None
                    }
                });

        indices.extend(iter);

        indices
    }

    fn can_grow(&self, leaf_idx: usize) -> bool {
        let depth = self.get_node_depth(leaf_idx);
        if depth >= MAX_DEPTH {
            return false;
        }

        let data_indices = self.get_leaf_data_indices(leaf_idx);
        data_indices.len() > 1 // Need at least two samples
    }

    /// Clone this tree into a new vector in the arena
    pub fn clone_into(&self, arena: &'arena Bump) -> Self {
        let split_var = BumpVec::from_iter_in(self.split_var.iter().copied(), arena);
        let split_value = BumpVec::from_iter_in(self.split_value.iter().copied(), arena);
        let leaf_values = BumpVec::from_iter_in(self.leaf_values.iter().copied(), arena);
        let leaf_indices = BumpVec::from_iter_in(self.leaf_indices.iter().copied(), arena);

        Self {
            split_var,
            split_value,
            leaf_values,
            leaf_indices,
        }
    }

    /// Get the number of internal nodes (splits) in this tree
    pub fn num_splits(&self) -> usize {
        self.split_var.len()
    }

    /// Get the number of leaf nodes in this tree
    pub fn num_leaves(&self) -> usize {
        self.leaf_values.len()
    }

    /// Get the number of samples this tree is tracking
    pub fn num_samples(&self) -> usize {
        self.leaf_indices.len()
    }

    /// Add a random split to grow the tree (for benchmarking purposes)
    pub fn add_random_split(&mut self, rng: &mut SmallRng, n_features: usize) {
        if self.leaf_values.len() == 0 {
            return;
        }

        // Pick a random leaf to split
        let leaf_idx = rng.gen_range(0..self.leaf_values.len());

        // Add split variables and values
        let split_var = rng.gen_range(0..n_features);
        let split_value = rng.gen::<f64>();

        self.split_var.push(split_var);
        self.split_value.push(split_value);

        // Add two new leaf values
        let original_leaf_value = self.leaf_values[leaf_idx];
        self.leaf_values
            .push(original_leaf_value + rng.gen::<f64>() * 0.1);
        self.leaf_values
            .push(original_leaf_value - rng.gen::<f64>() * 0.1);
    }
}

/// Ephemeral set of Particles.
///
/// Represents the transient collection of candidate trees (particles) that are
/// being proposed and evaluated for a tree update in the ensemble of trees (forest).
#[derive(Debug)]
pub struct Forest<'arena, const MAX_DEPTH: usize> {
    arena: &'arena Bump,
    trees: BumpVec<'arena, Tree<'arena, MAX_DEPTH>>,
    weights: BumpVec<'arena, f64>,
}

impl<'arena, const MAX_DEPTH: usize> Forest<'arena, MAX_DEPTH> {
    pub fn new(arena: &'arena Bump, n_particles: usize) -> Self {
        Self {
            arena,
            trees: BumpVec::with_capacity_in(n_particles, arena),
            weights: BumpVec::with_capacity_in(n_particles, arena),
        }
    }

    /// Adds a new Particle (tree) to the Forest
    pub fn plant_tree(&mut self, init_leaf: LeafValue, n_samples: usize) {
        let tree = Tree::<'arena, MAX_DEPTH>::stump(self.arena, init_leaf, n_samples);
        self.trees.push(tree);
    }

    /// Add a weight for the last tree
    pub fn add_weight(&mut self, weight: f64) {
        self.weights.push(weight);
    }

    /// Get the number of trees in the forest
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    /// Get a reference to the trees
    pub fn trees(&self) -> &[Tree<'arena, MAX_DEPTH>] {
        &self.trees
    }

    /// Get a reference to the weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Resample trees based on indices, creating copies when needed
    pub fn resample_trees(&mut self, indices: &[usize]) -> Result<(), &'static str> {
        if indices.len() != self.trees.len() {
            return Err("Number of indices must match number of trees");
        }

        // Create new trees based on resampled indices
        let mut new_trees = BumpVec::with_capacity_in(indices.len(), self.arena);

        for &idx in indices {
            if idx >= self.trees.len() {
                return Err("Invalid tree index");
            }

            // Clone the selected tree
            let cloned_tree = self.trees[idx].clone_into(self.arena);
            new_trees.push(cloned_tree);
        }

        self.trees = new_trees;
        Ok(())
    }

    /// Clear all trees and weights
    pub fn clear(&mut self) {
        self.trees.clear();
        self.weights.clear();
    }

    /// Create a tree with random splits for benchmarking
    pub fn plant_random_tree(
        &mut self,
        init_leaf: LeafValue,
        n_samples: usize,
        n_splits: usize,
        n_features: usize,
        rng: &mut SmallRng,
    ) {
        let mut tree = Tree::<'arena, MAX_DEPTH>::stump(self.arena, init_leaf, n_samples);

        for _ in 0..n_splits {
            tree.add_random_split(rng, n_features);
        }

        self.trees.push(tree);
    }
}
