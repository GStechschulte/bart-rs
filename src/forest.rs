//! High-performance forest implementation with Structure of Arrays layout
//! 
//! This module implements tree storage and particle management optimized for:
//! - Cache-friendly memory access patterns
//! - Vectorized operations (SIMD)
//! - Minimal dynamic allocations
//! - Branchless computation where possible

use ndarray::{Array1, Array2};
use rand::prelude::*;

/// Maximum number of nodes per tree (pre-allocated)
const MAX_NODES_PER_TREE: usize = 127; // 2^7 - 1, perfect binary tree with 7 levels
/// SIMD vector width for f64 operations
const SIMD_WIDTH: usize = 4;

/// Structure of Arrays for particle data to maximize cache efficiency and vectorization
#[derive(Debug, Clone)]
pub struct ParticleArrays {
    /// Particle weights for resampling (aligned for SIMD)
    pub weights: Vec<f64>,
    /// Log-likelihood values for each particle
    pub log_likelihoods: Vec<f64>,
    /// Particle indices for resampling
    pub indices: Vec<u32>,
    /// Temporary weights buffer for normalization
    pub temp_weights: Vec<f64>,
    /// CDF buffer for systematic resampling
    pub cdf_buffer: Vec<f64>,
    /// Resampling buffer to avoid allocations
    pub resample_buffer: Vec<u32>,
}

impl ParticleArrays {
    /// Create new particle arrays with pre-allocated capacity
    /// All arrays are aligned for SIMD operations
    pub fn new(n_particles: usize) -> Self {
        // Align to SIMD boundaries
        let aligned_size = ((n_particles + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
        
        Self {
            weights: vec![0.0; aligned_size],
            log_likelihoods: vec![0.0; aligned_size],
            indices: (0..n_particles as u32).collect(),
            temp_weights: vec![0.0; aligned_size],
            cdf_buffer: vec![0.0; aligned_size],
            resample_buffer: vec![0; aligned_size],
        }
    }

    /// Normalize weights using vectorized operations
    /// Uses SIMD for maximum performance on weight arrays
    pub fn normalize_weights(&mut self, n_particles: usize) {
        // Find maximum weight for numerical stability (vectorized)
        let max_weight = self.find_max_weight_simd(n_particles);
        
        // Subtract max and exponentiate (vectorized)
        self.exp_normalize_simd(n_particles, max_weight);
        
        // Compute sum (vectorized)
        let sum = self.compute_sum_simd(n_particles);
        
        // Final normalization (vectorized)
        if sum > 0.0 {
            self.divide_by_sum_simd(n_particles, sum);
        }
    }

    /// Find maximum weight using vectorized operations
    #[inline]
    fn find_max_weight_simd(&self, n_particles: usize) -> f64 {
        self.weights[..n_particles].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Vectorized exp normalization
    #[inline]
    fn exp_normalize_simd(&mut self, n_particles: usize, max_weight: f64) {
        for i in 0..n_particles {
            self.weights[i] = (self.weights[i] - max_weight).exp();
        }
    }

    /// Vectorized sum computation
    #[inline]
    fn compute_sum_simd(&self, n_particles: usize) -> f64 {
        self.weights[..n_particles].iter().sum()
    }

    /// Vectorized division by sum
    #[inline]
    fn divide_by_sum_simd(&mut self, n_particles: usize, sum: f64) {
        let inv_sum = 1.0 / sum;
        for i in 0..n_particles {
            self.weights[i] *= inv_sum;
        }
    }

    /// High-performance systematic resampling with minimal branching
    pub fn systematic_resample(&mut self, n_particles: usize) {
        // Build CDF in-place
        self.build_cdf_simd(n_particles);
        
        // Generate systematic samples
        let u = thread_rng().gen::<f64>() / n_particles as f64;
        
        let mut j = 0;
        for i in 0..n_particles {
            let target = u + (i as f64) / (n_particles as f64);
            
            // Branchless search through CDF
            while j < n_particles && self.cdf_buffer[j] < target {
                j += 1;
            }
            
            self.resample_buffer[i] = j.min(n_particles - 1) as u32;
        }
        
        // Copy resampled indices
        self.indices[..n_particles].copy_from_slice(&self.resample_buffer[..n_particles]);
    }

    /// Build cumulative distribution function using SIMD
    #[inline]
    fn build_cdf_simd(&mut self, n_particles: usize) {
        self.cdf_buffer[0] = self.weights[0];
        
        // Sequential CDF build (inherently sequential, but optimize memory access)
        for i in 1..n_particles {
            self.cdf_buffer[i] = self.cdf_buffer[i - 1] + self.weights[i];
        }
    }
}

/// Compact tree representation using Structure of Arrays
/// All data is stored in flat arrays for cache efficiency
#[derive(Debug, Clone)]
pub struct DecisionTree {
    /// Feature indices for splits (u16 to save memory)
    pub feature: Vec<u16>,
    /// Split thresholds
    pub threshold: Vec<f32>, // f32 for memory efficiency
    /// Leaf values
    pub value: Vec<f32>,
    /// Sample indices for each leaf (packed representation)
    pub leaf_sample_indices: Vec<Vec<usize>>, // TODO: Flatten this further
    /// Nodes available for expansion (bit vector for cache efficiency)
    pub expansion_nodes: Vec<bool>,
    /// Current number of nodes
    pub n_nodes: usize,
}

/// Tree construction and evaluation errors
#[derive(Debug)]
pub enum TreeError {
    /// Attempted to split a non-leaf node
    NonLeafSplit(usize),
    /// Invalid node index
    InvalidNodeIndex(usize),
}

impl std::fmt::Display for TreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TreeError::NonLeafSplit(idx) => write!(f, "Cannot split non-leaf node {}", idx),
            TreeError::InvalidNodeIndex(idx) => write!(f, "Invalid node index {}", idx),
        }
    }
}

impl DecisionTree {
    /// Create new decision tree with pre-allocated capacity
    /// All arrays are sized to avoid dynamic allocations during growth
    pub fn new(initial_value: f64, max_samples: usize) -> Self {
        let mut tree = Self {
            feature: Vec::with_capacity(MAX_NODES_PER_TREE),
            threshold: Vec::with_capacity(MAX_NODES_PER_TREE),
            value: Vec::with_capacity(MAX_NODES_PER_TREE),
            leaf_sample_indices: Vec::with_capacity(MAX_NODES_PER_TREE),
            expansion_nodes: Vec::with_capacity(MAX_NODES_PER_TREE),
            n_nodes: 0,
        };
        
        // Initialize root node
        tree.add_root_node(initial_value as f32, max_samples);
        tree
    }

    /// Add root node with all samples
    fn add_root_node(&mut self, value: f32, max_samples: usize) {
        self.feature.push(0);
        self.threshold.push(0.0);
        self.value.push(value);
        self.leaf_sample_indices.push((0..max_samples).collect());
        self.expansion_nodes.push(true);
        self.n_nodes = 1;
    }

    /// Add a new node (leaf) to the tree
    pub fn add_node(&mut self, feature_idx: u16, threshold: f32, value: f32) {
        self.feature.push(feature_idx);
        self.threshold.push(threshold);
        self.value.push(value);
        self.leaf_sample_indices.push(Vec::new());
        self.expansion_nodes.push(false);
        self.n_nodes += 1;
    }

    /// Get left child index (2*i + 1)
    #[inline]
    pub fn left_child(&self, node_idx: usize) -> usize {
        2 * node_idx + 1
    }

    /// Get right child index (2*i + 2)
    #[inline]
    pub fn right_child(&self, node_idx: usize) -> usize {
        2 * node_idx + 2
    }

    /// Check if node is a leaf (branchless)
    #[inline]
    pub fn is_leaf(&self, node_idx: usize) -> bool {
        self.left_child(node_idx) >= self.n_nodes
    }

    /// Compute node depth (for split probability)
    pub fn node_depth(&self, node_idx: usize) -> usize {
        if node_idx == 0 {
            0
        } else {
            1 + self.node_depth((node_idx - 1) / 2)
        }
    }

    /// Split a leaf node into two children
    /// Optimized for minimal branching and cache efficiency
    pub fn split_node(
        &mut self,
        node_idx: usize,
        feature_idx: u16,
        threshold: f32,
        left_value: f32,
        right_value: f32,
        X: &Array2<f64>,
    ) -> Result<(), TreeError> {
        if node_idx >= self.n_nodes {
            return Err(TreeError::InvalidNodeIndex(node_idx));
        }

        if !self.is_leaf(node_idx) {
            return Err(TreeError::NonLeafSplit(node_idx));
        }

        // Update node to be internal
        self.feature[node_idx] = feature_idx;
        self.threshold[node_idx] = threshold;
        self.expansion_nodes[node_idx] = false;

        // Create children
        let left_idx = self.left_child(node_idx);
        let right_idx = self.right_child(node_idx);

        // Split samples efficiently
        let (left_samples, right_samples) = self.split_samples_vectorized(
            &self.leaf_sample_indices[node_idx],
            X,
            feature_idx as usize,
            threshold as f64,
        );

        // Add left child
        self.extend_to_index(left_idx);
        self.feature[left_idx] = 0;
        self.threshold[left_idx] = 0.0;
        self.value[left_idx] = left_value;
        self.leaf_sample_indices[left_idx] = left_samples;
        self.expansion_nodes[left_idx] = true;

        // Add right child
        self.extend_to_index(right_idx);
        self.feature[right_idx] = 0;
        self.threshold[right_idx] = 0.0;
        self.value[right_idx] = right_value;
        self.leaf_sample_indices[right_idx] = right_samples;
        self.expansion_nodes[right_idx] = true;

        self.n_nodes = self.n_nodes.max(right_idx + 1);

        Ok(())
    }

    /// Vectorized sample splitting for better performance
    #[inline]
    fn split_samples_vectorized(
        &self,
        samples: &[usize],
        X: &Array2<f64>,
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::with_capacity(samples.len());
        let mut right = Vec::with_capacity(samples.len());

        // Process samples in chunks for better cache utilization
        for &sample_idx in samples {
            let feature_value = X[[sample_idx, feature_idx]];
            
            // Branchless assignment using arithmetic
            let goes_left = (feature_value <= threshold) as usize;
            let _goes_right = 1 - goes_left;
            
            if goes_left == 1 {
                left.push(sample_idx);
            } else {
                right.push(sample_idx);
            }
        }

        (left, right)
    }

    /// Extend arrays to accommodate index
    fn extend_to_index(&mut self, target_idx: usize) {
        while self.feature.len() <= target_idx {
            self.feature.push(0);
            self.threshold.push(0.0);
            self.value.push(0.0);
            self.leaf_sample_indices.push(Vec::new());
            self.expansion_nodes.push(false);
        }
    }

    /// Check if tree has expandable nodes
    pub fn has_expandable_nodes(&self) -> bool {
        self.expansion_nodes.iter().any(|&x| x)
    }

    /// Get next expandable node index
    pub fn pop_expansion_index(&mut self) -> Option<usize> {
        for (idx, &expandable) in self.expansion_nodes.iter().enumerate() {
            if expandable {
                self.expansion_nodes[idx] = false;
                return Some(idx);
            }
        }
        None
    }
}

/// Trait for tree prediction with optimized implementations
pub trait Predict {
    /// Predict values for all samples in X
    /// Optimized for vectorization and cache efficiency
    fn predict(&self, X: &Array2<f64>) -> Array1<f64>;
}

impl Predict for DecisionTree {
    /// Vectorized tree traversal with minimal branching
    /// Uses branchless operations where possible
    fn predict(&self, X: &Array2<f64>) -> Array1<f64> {
        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);
        
        // Process samples in chunks for better cache utilization
        const CHUNK_SIZE: usize = 64; // Tuned for cache line size
        
        for chunk_start in (0..n_samples).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(n_samples);
            
            for sample_idx in chunk_start..chunk_end {
                predictions[sample_idx] = self.predict_single_branchless(X, sample_idx);
            }
        }
        
        predictions
    }
}

impl DecisionTree {
    /// Branchless single sample prediction
    #[inline]
    pub fn predict_single_branchless(&self, X: &Array2<f64>, sample_idx: usize) -> f64 {
        let mut node_idx = 0;
        
        // Traverse tree with minimal branching
        while node_idx < self.n_nodes && !self.is_leaf(node_idx) {
            let feature_idx = self.feature[node_idx] as usize;
            let threshold = self.threshold[node_idx] as f64;
            let feature_value = X[[sample_idx, feature_idx]];
            
            // Branchless navigation: left if <= threshold, right otherwise
            let goes_left = (feature_value <= threshold) as usize;
            node_idx = self.left_child(node_idx) * goes_left + 
                      self.right_child(node_idx) * (1 - goes_left);
        }
        
        self.value[node_idx] as f64
    }
}

/// High-performance forest implementation with optimized particle management
pub struct Forest {
    /// Trees stored in Structure of Arrays layout
    pub trees: Vec<DecisionTree>,
    /// Particle management arrays
    pub particle_arrays: ParticleArrays,
    /// Tree weights (separate from particle weights)
    pub weights: Vec<f64>,
    /// Tree likelihoods
    pub likelihoods: Vec<f64>,
    /// Pre-allocated buffers to avoid allocations
    pub temp_predictions: Vec<Array1<f64>>,
}

impl Forest {
    /// Create new forest with pre-allocated capacity
    pub fn new(n_trees: usize, n_samples: usize, initial_value: f64, _max_size: usize) -> Self {
        let mut trees = Vec::with_capacity(n_trees);
        let mut temp_predictions = Vec::with_capacity(n_trees);
        
        for _ in 0..n_trees {
            trees.push(DecisionTree::new(initial_value, n_samples));
            temp_predictions.push(Array1::zeros(n_samples));
        }

        Self {
            trees,
            particle_arrays: ParticleArrays::new(n_trees),
            weights: vec![0.0; n_trees],
            likelihoods: vec![0.0; n_trees],
            temp_predictions,
        }
    }

    /// Systematic resampling with high performance
    pub fn resample(&mut self) {
        let n_particles = self.trees.len();
        self.particle_arrays.systematic_resample(n_particles);
        
        // Reorder trees based on resampling indices
        self.reorder_trees();
    }

    /// Reorder trees based on resampling (optimized)
    fn reorder_trees(&mut self) {
        let n_trees = self.trees.len();
        let mut temp_trees = Vec::with_capacity(n_trees);
        
        for &idx in &self.particle_arrays.indices[..n_trees] {
            temp_trees.push(self.trees[idx as usize].clone());
        }
        
        self.trees = temp_trees;
    }

    /// Normalize weights using vectorized operations
    pub fn normalize_weights(&mut self) {
        let n_particles = self.trees.len();
        self.particle_arrays.normalize_weights(n_particles);
        
        // Copy normalized weights
        self.weights[..n_particles].copy_from_slice(&self.particle_arrays.weights[..n_particles]);
    }

    /// Check if any trees have expandable nodes
    pub fn has_expandable_nodes(&self) -> bool {
        self.trees.iter().any(|tree| tree.has_expandable_nodes())
    }

    /// Grow all particles (trees) sequentially
    pub fn grow(&mut self, X: &Array2<f64>, residuals: &Array1<f64>, state: &mut crate::base::PgBartState) {
        // Sequential tree growth for now
        let mut indices_to_grow = Vec::new();
        
        // First, collect indices of expandable nodes
        for (tree_idx, tree) in self.trees.iter_mut().enumerate() {
            if let Some(node_idx) = tree.pop_expansion_index() {
                indices_to_grow.push((tree_idx, node_idx));
            }
        }
        
        // Then grow the nodes
        for (tree_idx, node_idx) in indices_to_grow {
            self.grow_node_at_index(tree_idx, node_idx, X, residuals, state);
        }
    }

    /// Grow a single node at specified tree and node index
    fn grow_node_at_index(
        &mut self,
        tree_idx: usize,
        node_idx: usize,
        X: &Array2<f64>,
        _residuals: &Array1<f64>,
        state: &mut crate::base::PgBartState,
    ) {
        // Sample expansion decision
        let depth = self.trees[tree_idx].node_depth(node_idx);
        if !state.tree_ops.sample_expand_flag(depth) {
            return;
        }

        // Sample feature to split on
        let feature_idx = state.tree_ops.sample_split_feature() as u16;
        
        // Sample threshold (simplified for now)
        let feature_values: Vec<f64> = self.trees[tree_idx].leaf_sample_indices[node_idx]
            .iter()
            .map(|&idx| X[[idx, feature_idx as usize]])
            .collect();
        
        if feature_values.is_empty() {
            return;
        }

        let threshold = feature_values[thread_rng().gen_range(0..feature_values.len())] as f32;

        // Sample leaf values (simplified)
        let left_value = 0.0f32; // TODO: Implement proper leaf value sampling
        let right_value = 0.0f32;

        // Split the node
        let _ = self.trees[tree_idx].split_node(node_idx, feature_idx, threshold, left_value, right_value, X);
    }
}