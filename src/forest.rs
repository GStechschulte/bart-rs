use bumpalo::{collections::Vec, Bump};
use numpy::ndarray::{Array1, Array2};

pub type NodeIndex = usize;
pub type SplitFeature = usize;
pub type TreeIndex = usize;
pub type DataIndex = usize;

pub type SplitValue = f64;
pub type LeafValue = f64;
pub type LeafIndex = usize;

/// Ephemeral set of Particles.
///
/// Represents the transient collection of candidate trees (particles) that are
/// being proposed and evaluated for a tree update in the ensemble of trees (forest).
#[derive(Debug)]
pub struct Forest<'arena> {
    pub weights: Vec<'arena, f64>,
    pub active_nodes: Vec<'arena, usize>,
    pub n_particles: usize,
    pub max_depth: usize,
    pub trees: Vec<'arena, Tree<'arena>>,
}

#[derive(Debug)]
pub struct Tree<'arena> {
    // Each tree gets a pre-allocated slice of the arena
    // Using heap-based indexing: parent at i, children at 2i and 2i+1
    pub split_features: &'arena mut [SplitFeature],
    pub split_thresholds: &'arena mut [SplitValue],
    pub leaf_values: &'arena mut [LeafValue],
    pub leaf_indices: &'arena mut [LeafIndex],
    pub is_leaf: &'arena mut [bool], // Explicit leaf marking for faster traversal

    pub size: usize,      // Current number of nodes used
    pub capacity: usize,  // Maximum nodes (2^(max_depth+1) - 1)
    pub max_depth: usize, // Maximum depth for probability calculations

    // Data indices for each node - stored as a flattened structure
    // Each node gets a slice of indices representing the data points that belong to it
    pub node_data_indices: &'arena mut [Vec<'arena, DataIndex>],
}

impl<'arena> Forest<'arena> {
    pub fn new_in(
        arena: &'arena Bump,
        n_particles: usize,
        n_samples: usize,
        max_depth: usize,
        init_leaf_value: LeafValue,
    ) -> Self {
        let max_nodes_per_tree = (1 << (max_depth + 1)) - 1;

        // Allocate hot path metadata using Vec for SIMD operations
        let mut weights = Vec::with_capacity_in(n_particles, arena);
        let mut active_nodes = Vec::with_capacity_in(n_particles, arena);
        let mut trees = Vec::with_capacity_in(n_particles, arena);

        // Allocate one large block for all tree data, then slice it so
        // each Particle tree gets a "slice"
        let total_tree_capacity = n_particles * max_nodes_per_tree;
        let mut split_features_block = arena.alloc_slice_fill_copy(total_tree_capacity, 0);
        let mut split_thresholds_block = arena.alloc_slice_fill_copy(total_tree_capacity, 0.0);
        let mut leaf_values_block =
            arena.alloc_slice_fill_copy(total_tree_capacity, init_leaf_value);

        // All data points initially belong to the root node
        let mut leaf_indices_block = arena.alloc_slice_fill_copy(total_tree_capacity, 0);
        let mut is_leaf_block = arena.alloc_slice_fill_copy(total_tree_capacity, true);

        // Allocate data indices for each node - initially all data goes to root
        let total_node_data_capacity = n_particles * max_nodes_per_tree;
        let mut node_data_indices_block =
            arena.alloc_slice_fill_with(total_node_data_capacity, |_| Vec::new_in(arena));

        for i in 0..n_particles {
            weights.push(0.0);
            active_nodes.push(1); // Start with root node

            // Use split_at_mut to create non-overlapping mutable slices
            let (split_features_slice, split_features_rest) =
                split_features_block.split_at_mut(max_nodes_per_tree);
            let (split_thresholds_slice, split_thresholds_rest) =
                split_thresholds_block.split_at_mut(max_nodes_per_tree);
            let (leaf_values_slice, leaf_values_rest) =
                leaf_values_block.split_at_mut(max_nodes_per_tree);

            let (leaf_indices_slice, leaf_indices_rest) =
                leaf_indices_block.split_at_mut(max_nodes_per_tree);

            let (is_leaf_slice, is_leaf_rest) = is_leaf_block.split_at_mut(max_nodes_per_tree);

            let (node_data_indices_slice, node_data_indices_rest) =
                node_data_indices_block.split_at_mut(max_nodes_per_tree);

            // Update the remaining slices for the next iteration
            split_features_block = split_features_rest;
            split_thresholds_block = split_thresholds_rest;
            leaf_values_block = leaf_values_rest;
            leaf_indices_block = leaf_indices_rest;
            is_leaf_block = is_leaf_rest;
            node_data_indices_block = node_data_indices_rest;

            // Initialize root node with all data indices
            let mut root_data_indices = Vec::with_capacity_in(n_samples, arena);
            for j in 0..n_samples {
                root_data_indices.push(j);
            }
            node_data_indices_slice[1] = root_data_indices;

            // Each tree gets its own slice of the pre-allocated blocks
            let tree = Tree {
                split_features: split_features_slice,
                split_thresholds: split_thresholds_slice,
                leaf_values: leaf_values_slice,
                leaf_indices: leaf_indices_slice,
                is_leaf: is_leaf_slice,
                size: 1, // Root node only
                capacity: max_nodes_per_tree,
                max_depth,
                node_data_indices: node_data_indices_slice,
            };

            // Initialize root as leaf
            tree.is_leaf[1] = true; // Heap indexing starts at 1
            tree.leaf_values[1] = init_leaf_value;

            trees.push(tree);
        }

        Self {
            weights,
            active_nodes,
            n_particles,
            max_depth,
            trees,
        }
    }
}

impl<'arena> Tree<'arena> {
    /// Calculate the depth of a node given its index (heap-based indexing)
    pub fn node_depth(&self, node_idx: usize) -> usize {
        if node_idx == 0 {
            return 0; // Invalid node
        }
        (node_idx as f64).log2().floor() as usize
    }

    /// Calculate the probability that a node at given depth should expand
    /// Uses the standard BART prior: P(split) = alpha * (1 + depth)^(-beta)
    pub fn expansion_probability(&self, depth: usize, alpha: f64, beta: f64) -> f64 {
        if depth >= self.max_depth {
            return 0.0; // Cannot expand beyond max depth
        }
        alpha * (1.0 + depth as f64).powf(-beta)
    }

    /// Get the data indices that belong to a specific node
    pub fn get_node_data_indices(&self, node_idx: usize) -> &[DataIndex] {
        &self.node_data_indices[node_idx]
    }

    /// Check if a node can be split (has enough data and is not at max depth)
    pub fn can_split(&self, node_idx: usize, min_samples_leaf: usize) -> bool {
        let depth = self.node_depth(node_idx);
        let data_count = self.node_data_indices[node_idx].len();

        depth < self.max_depth && data_count >= 2 * min_samples_leaf
    }

    /// Split a leaf node into two children
    pub fn split_node(
        &mut self,
        node_idx: usize,
        feature_idx: SplitFeature,
        threshold: SplitValue,
        left_data_indices: std::vec::Vec<DataIndex>,
        right_data_indices: std::vec::Vec<DataIndex>,
        left_leaf_value: LeafValue,
        right_leaf_value: LeafValue,
    ) -> Result<(), &'static str> {
        if !self.is_leaf[node_idx] {
            return Err("Cannot split non-leaf node");
        }

        let left_child_idx = 2 * node_idx;
        let right_child_idx = 2 * node_idx + 1;

        if right_child_idx >= self.capacity {
            return Err("Tree capacity exceeded");
        }

        // Convert leaf to internal node
        self.is_leaf[node_idx] = false;
        self.split_features[node_idx] = feature_idx;
        self.split_thresholds[node_idx] = threshold;

        // Create left child
        self.is_leaf[left_child_idx] = true;
        self.leaf_values[left_child_idx] = left_leaf_value;
        // Clear and populate left child data indices
        self.node_data_indices[left_child_idx].clear();
        for idx in left_data_indices {
            self.node_data_indices[left_child_idx].push(idx);
        }

        // Create right child
        self.is_leaf[right_child_idx] = true;
        self.leaf_values[right_child_idx] = right_leaf_value;
        // Clear and populate right child data indices
        self.node_data_indices[right_child_idx].clear();
        for idx in right_data_indices {
            self.node_data_indices[right_child_idx].push(idx);
        }

        // Update size
        self.size += 2;

        Ok(())
    }

    /// Predict values for given data points using this tree
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(x.shape()[0]);

        for (sample_idx, sample) in x.outer_iter().enumerate() {
            let mut node_idx = 1; // Start at root

            // Traverse tree until we reach a leaf
            while !self.is_leaf[node_idx] {
                let feature_idx = self.split_features[node_idx];
                let threshold = self.split_thresholds[node_idx];

                if sample[feature_idx] < threshold {
                    node_idx = 2 * node_idx; // Left child
                } else {
                    node_idx = 2 * node_idx + 1; // Right child
                }

                // Safety check to avoid infinite loops
                if node_idx >= self.capacity {
                    break;
                }
            }

            // Get leaf value
            if node_idx < self.capacity && self.is_leaf[node_idx] {
                predictions[sample_idx] = self.leaf_values[node_idx];
            }
        }

        predictions
    }

    /// Check if the tree has finished growing (no more expandable nodes)
    pub fn finished(&self) -> bool {
        for node_idx in 1..=self.size {
            if node_idx < self.capacity && self.is_leaf[node_idx] && self.can_split(node_idx, 1) {
                return false;
            }
        }
        true
    }
}

impl<'arena> Forest<'arena> {
    /// Compute predictions for all particles in the forest
    pub fn predict_all(&self, x: &Array2<f64>) -> std::vec::Vec<Array1<f64>> {
        self.trees.iter().map(|tree| tree.predict(x)).collect()
    }

    /// Normalize weights using log-sum-exp for numerical stability
    pub fn normalize_weights(&mut self) {
        let max_weight = self
            .weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Subtract max for numerical stability
        let exp_weights: std::vec::Vec<f64> = self
            .weights
            .iter()
            .map(|&w| (w - max_weight).exp())
            .collect();

        let sum_exp: f64 = exp_weights.iter().sum();

        // Normalize
        for (i, &exp_w) in exp_weights.iter().enumerate() {
            self.weights[i] = exp_w / sum_exp;
        }
    }

    /// Check if any particles are still growing
    pub fn any_growing(&self) -> bool {
        self.trees.iter().any(|tree| !tree.finished())
    }
}
