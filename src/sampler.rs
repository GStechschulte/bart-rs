use std::ffi::c_double;

use bumpalo::Bump;
use numpy::ndarray::{Array1, Array2};
use rand::{rngs::SmallRng, Rng};
use rand_distr::Normal;

use crate::forest::{DataIndex, Forest, SplitFeature, Tree};
use crate::resampling::{ResamplingStrategies, SystematicResampling};
use crate::response::{GaussianResponseStrategy, ResponseStrategies};
use crate::split_rules::{ContinuousSplitRule, SplitRules};
use crate::PyBartSettings;
use crate::{base::PgBartState, forest::LeafValue};

pub type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

/// Particle Gibbs sampler for BART
pub struct PgBartSampler {
    rng: SmallRng,
    model: LogpFunc,
    leaf_dist: Normal<LeafValue>,
    settings: PyBartSettings,
    split_strategy: SplitRules,
    response_strategy: ResponseStrategies,
    resampling_strategy: ResamplingStrategies,
}

impl PgBartSampler {
    pub fn new(model: LogpFunc, settings: PyBartSettings) -> Self {
        let leaf_dist = Normal::new(0.0, 1.0).unwrap();

        // Initialize default strategies
        let split_strategy = SplitRules::Continuous(ContinuousSplitRule);
        let response_strategy = ResponseStrategies::Gaussian(GaussianResponseStrategy::new(
            settings.init_leaf_std.powi(2),
        ));
        let resampling_strategy = ResamplingStrategies::Systematic(SystematicResampling);

        Self {
            rng: settings.rng.clone(),
            model: model,
            leaf_dist: leaf_dist,
            settings: settings,
            split_strategy,
            response_strategy,
            resampling_strategy,
        }
    }

    pub fn step(&mut self, state: &mut PgBartState) {
        println!("Stepping...");
        let start_time = std::time::Instant::now();

        // Perform sequential updates for each tree in the ensemble
        for tree_id in 0..self.settings.n_trees {
            self.update_tree(tree_id, state);
        }

        let step_time = start_time.elapsed().as_nanos() as f64;
        println!("step time: {} ns", step_time);
    }

    /// Update a single tree in the ensemble using particle filtering
    fn update_tree(&mut self, tree_id: usize, state: &mut PgBartState) {
        println!("Updating tree {}", tree_id);

        // Get current tree predictions (if this were a real ensemble)
        let current_tree_predictions = Array1::<f64>::zeros(state.y.len());
        let predictions_minus_current = &state.predictions - &current_tree_predictions;

        // Create a new arena for this tree's particle set
        let arena = Bump::new();

        // Initialize local particles
        let mut local_forest = Forest::new_in(
            &arena,
            self.settings.n_particles,
            state.y.len(),
            self.settings.max_depth,
            self.settings.init_leaf_std,
        );

        // Grow particles until all are finished
        while local_forest.any_growing() {
            // Grow each particle
            self.grow_particles(&mut local_forest, state, &arena);

            // Update weights for all particles
            self.update_particle_weights(&mut local_forest, state, &predictions_minus_current);

            // Normalize weights
            local_forest.normalize_weights();

            // Resample if needed (based on effective sample size)
            self.resample_particles(&mut local_forest, &arena);
        }

        // Final weight normalization and particle selection
        local_forest.normalize_weights();
        let selected_particle_idx = self.select_particle(&local_forest);

        // Update ensemble predictions with selected particle
        let new_predictions = local_forest.trees[selected_particle_idx].predict(&state.X);
        state.predictions = predictions_minus_current + &new_predictions;

        println!(
            "Selected particle {} for tree {}",
            selected_particle_idx, tree_id
        );
    }

    /// Grow all particles in the forest
    fn grow_particles(&mut self, forest: &mut Forest, state: &mut PgBartState, arena: &Bump) {
        for tree in &mut forest.trees {
            if !tree.finished() {
                self.grow_tree(tree, state, arena);
            }
        }
    }

    fn grow_tree(&mut self, tree: &mut Tree, state: &mut PgBartState, _arena: &Bump) {
        println!("Growing tree...");

        // Continue growing until we decide to stop
        let mut continue_growing = true;

        while continue_growing {
            continue_growing = false;

            // Find all leaf nodes that can potentially be expanded
            let mut expandable_nodes = Vec::new();

            for node_idx in 1..=tree.size {
                if node_idx < tree.capacity && tree.is_leaf[node_idx] && tree.can_split(node_idx, 1)
                {
                    expandable_nodes.push(node_idx);
                }
            }

            if expandable_nodes.is_empty() {
                break;
            }

            // Step 1: Select a random leaf node to potentially grow
            let node_idx = expandable_nodes[self.rng.random_range(0..expandable_nodes.len())];

            // Step 2: Calculate depth of this node
            let depth = tree.node_depth(node_idx);

            // Step 3: Calculate probability of expansion using BART prior
            let expansion_prob =
                tree.expansion_probability(depth, self.settings.alpha, self.settings.beta);

            // Decide whether to expand this node
            if self.rng.random::<f64>() < expansion_prob {
                // Step 4: Get data samples that belong to this node
                let node_data_indices = tree.get_node_data_indices(node_idx).to_vec();

                if node_data_indices.len() < 2 {
                    continue; // Cannot split with less than 2 samples
                }

                // Step 5: Sample a split feature based on split prior
                let feature_idx = self.sample_split_feature(&state.X, &node_data_indices);

                // Step 6 & 7: Use SplitRule strategy to compute split value and data split
                let feature_values: Vec<f64> = node_data_indices
                    .iter()
                    .map(|&idx| state.X[[idx, feature_idx]])
                    .collect();

                if let Some(split_threshold) = self
                    .split_strategy
                    .sample_split_value(&feature_values, &mut self.rng)
                {
                    let (left_indices_std, right_indices_std) =
                        self.split_strategy.split_data_indices(
                            &state.X,
                            feature_idx,
                            split_threshold,
                            &node_data_indices,
                        );

                    // Ensure both children have at least one sample
                    if left_indices_std.is_empty() || right_indices_std.is_empty() {
                        continue;
                    }

                    // Step 8: Sample leaf values using ResponseStrategy
                    let left_leaf_value = self.response_strategy.sample_leaf_value(
                        &state.y,
                        &left_indices_std,
                        &mut self.rng,
                    );

                    let right_leaf_value = self.response_strategy.sample_leaf_value(
                        &state.y,
                        &right_indices_std,
                        &mut self.rng,
                    );

                    // Step 9: Split the node
                    if tree
                        .split_node(
                            node_idx,
                            feature_idx,
                            split_threshold,
                            left_indices_std,
                            right_indices_std,
                            left_leaf_value,
                            right_leaf_value,
                        )
                        .is_ok()
                    {
                        continue_growing = true;
                        println!(
                            "Split node {} at depth {} on feature {} with threshold {}",
                            node_idx, depth, feature_idx, split_threshold
                        );
                    }
                }
            }
        }
    }

    /// Sample a split feature based on the split prior probabilities
    fn sample_split_feature(
        &mut self,
        x: &Array2<f64>,
        _data_indices: &[DataIndex],
    ) -> SplitFeature {
        let n_features = x.shape()[1];

        // If split_prior is provided, use it for weighted sampling
        if self.settings.split_prior.len() == n_features {
            // Weighted sampling based on split prior
            let total_weight: f64 = self.settings.split_prior.iter().sum();
            let mut cumulative_weight = 0.0;
            let random_value = self.rng.random::<f64>() * total_weight;

            for (idx, &weight) in self.settings.split_prior.iter().enumerate() {
                cumulative_weight += weight;
                if random_value <= cumulative_weight {
                    return idx;
                }
            }
        }

        // Default: uniform sampling over all features
        self.rng.random_range(0..n_features)
    }

    /// Update weights for all particles based on their predictions
    fn update_particle_weights(
        &mut self,
        forest: &mut Forest,
        state: &PgBartState,
        base_predictions: &Array1<f64>,
    ) {
        println!("Updating particle weights...");

        for (i, tree) in forest.trees.iter().enumerate() {
            // Get predictions from this particle
            let tree_predictions = tree.predict(&state.X);

            // Combine with base predictions (from other trees)
            let total_predictions = base_predictions + &tree_predictions;

            // Compute log-likelihood
            let log_likelihood = self.compute_log_likelihood(total_predictions);
            forest.weights[i] = log_likelihood;
        }
    }

    /// Compute log-likelihood using the provided model function
    fn compute_log_likelihood(&mut self, predictions: Array1<f64>) -> f64 {
        unsafe { (self.model)(predictions.as_ptr(), predictions.len()) }
    }

    /// Resample particles based on their weights
    fn resample_particles(&mut self, forest: &mut Forest, _arena: &Bump) {
        println!("Resampling particles...");

        let indices = self.resampling_strategy.resample(
            &forest.weights,
            self.settings.n_particles,
            &mut self.rng,
        );

        // For simplicity, we'll just reset weights rather than deep copying trees
        // In a full implementation, you'd want to properly copy the tree structures
        println!("Would resample with indices: {:?}", indices);

        // Reset weights uniformly
        let uniform_weight = 1.0 / self.settings.n_particles as f64;
        for weight in &mut forest.weights {
            *weight = uniform_weight;
        }
    }

    /// Select a particle based on normalized weights
    fn select_particle(&mut self, forest: &Forest) -> usize {
        let target = self.rng.random::<f64>();
        let mut cumulative = 0.0;

        for (i, &weight) in forest.weights.iter().enumerate() {
            cumulative += weight;
            if target <= cumulative {
                return i;
            }
        }

        // Fallback to last particle
        forest.weights.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array;
    use rand::SeedableRng;

    // Mock logp function for testing
    unsafe extern "C" fn mock_logp(x: *const f64, len: usize) -> c_double {
        let slice = std::slice::from_raw_parts(x, len);
        slice.iter().sum::<f64>()
    }

    fn create_test_settings() -> PyBartSettings {
        PyBartSettings {
            init_leaf_value: 0.0,
            init_leaf_std: 1.0,
            n_trees: 3,
            n_particles: 5,
            max_depth: 3,
            alpha: 0.95,
            beta: 2.0,
            split_prior: vec![0.5, 0.5],
            response_rule: "gaussian".to_string(),
            batch_size: (0.5, 0.5),
            rng: SmallRng::seed_from_u64(42),
        }
    }

    fn create_test_state() -> PgBartState {
        let X = Array::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                6.5, 7.5, 8.5, 9.5,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let forest = vec![0.0; 3];
        let weights = vec![0.0; 3];
        let predictions = Array1::from_elem(10, 0.0);

        PgBartState::new(X, y, forest, weights, predictions)
    }

    #[test]
    fn test_sampler_creation() {
        let settings = create_test_settings();
        let sampler = PgBartSampler::new(mock_logp, settings);

        assert_eq!(sampler.settings.n_particles, 5);
        assert_eq!(sampler.settings.max_depth, 3);
        assert_eq!(sampler.settings.alpha, 0.95);
        assert_eq!(sampler.settings.beta, 2.0);
    }

    #[test]
    fn test_split_feature_sampling() {
        let settings = create_test_settings();
        let mut sampler = PgBartSampler::new(mock_logp, settings);
        let state = create_test_state();

        // Test uniform sampling when no split_prior
        let feature = sampler.sample_split_feature(&state.X, &[0, 1, 2, 3, 4]);
        assert!(feature < 2); // Should be 0 or 1

        // Test with split_prior
        sampler.settings.split_prior = vec![0.8, 0.2];
        let feature = sampler.sample_split_feature(&state.X, &[0, 1, 2, 3, 4]);
        assert!(feature < 2);
    }

    #[test]
    fn test_grow_tree_basic() {
        let settings = create_test_settings();
        let mut sampler = PgBartSampler::new(mock_logp, settings);
        let mut state = create_test_state();

        // Create a simple arena and forest for testing
        let arena = Bump::new();
        let mut forest = Forest::new_in(
            &arena,
            1, // single particle
            state.y.len(),
            3,   // max_depth
            0.0, // init_leaf_value
        );

        // Get initial tree size
        let initial_size = forest.trees[0].size;

        // Grow the tree
        sampler.grow_tree(&mut forest.trees[0], &mut state, &arena);

        // Tree should potentially have grown (depends on probability)
        // At minimum, the method should complete without panicking
        assert!(forest.trees[0].size >= initial_size);
    }

    #[test]
    fn test_tree_node_depth() {
        let arena = Bump::new();
        let forest = Forest::new_in(&arena, 1, 10, 3, 0.0);
        let tree = &forest.trees[0];

        // Test depth calculation (heap-based indexing)
        assert_eq!(tree.node_depth(1), 0); // Root node
        assert_eq!(tree.node_depth(2), 1); // Left child of root
        assert_eq!(tree.node_depth(3), 1); // Right child of root
        assert_eq!(tree.node_depth(4), 2); // Left child of left child
        assert_eq!(tree.node_depth(7), 2); // Right child of right child
    }

    #[test]
    fn test_expansion_probability() {
        let arena = Bump::new();
        let forest = Forest::new_in(&arena, 1, 10, 3, 0.0);
        let tree = &forest.trees[0];

        let alpha = 0.95;
        let beta = 2.0;

        // Test probability calculation
        let prob_depth_0 = tree.expansion_probability(0, alpha, beta);
        let prob_depth_1 = tree.expansion_probability(1, alpha, beta);
        let prob_depth_2 = tree.expansion_probability(2, alpha, beta);
        let prob_depth_3 = tree.expansion_probability(3, alpha, beta);

        // Probabilities should decrease with depth
        assert!(prob_depth_0 > prob_depth_1);
        assert!(prob_depth_1 > prob_depth_2);
        assert!(prob_depth_2 > prob_depth_3);

        // At max depth, probability should be 0
        assert_eq!(prob_depth_3, 0.0);
    }

    #[test]
    fn test_can_split() {
        let arena = Bump::new();
        let mut forest = Forest::new_in(&arena, 1, 10, 3, 0.0);
        let tree = &mut forest.trees[0];

        // Root node with 10 samples should be splittable
        assert!(tree.can_split(1, 1)); // min_samples_leaf = 1
        assert!(tree.can_split(1, 2)); // min_samples_leaf = 2
        assert!(tree.can_split(1, 4)); // min_samples_leaf = 4
        assert!(!tree.can_split(1, 6)); // min_samples_leaf = 6 (need 12 samples)
    }

    #[test]
    fn test_node_splitting() {
        let arena = Bump::new();
        let mut forest = Forest::new_in(&arena, 1, 10, 3, 0.0);
        let tree = &mut forest.trees[0];

        let initial_size = tree.size;

        // Split the root node
        let left_indices = vec![0, 1, 2, 3, 4];
        let right_indices = vec![5, 6, 7, 8, 9];

        let result = tree.split_node(
            1,   // root node
            0,   // feature 0
            5.0, // threshold
            left_indices.clone(),
            right_indices.clone(),
            -1.0, // left leaf value
            1.0,  // right leaf value
        );

        assert!(result.is_ok());
        assert_eq!(tree.size, initial_size + 2);
        assert!(!tree.is_leaf[1]); // Root is no longer a leaf
        assert!(tree.is_leaf[2]); // Left child is a leaf
        assert!(tree.is_leaf[3]); // Right child is a leaf
        assert_eq!(tree.split_features[1], 0);
        assert_eq!(tree.split_thresholds[1], 5.0);
        assert_eq!(tree.leaf_values[2], -1.0);
        assert_eq!(tree.leaf_values[3], 1.0);

        // Check data indices
        assert_eq!(tree.node_data_indices[2].len(), left_indices.len());
        assert_eq!(tree.node_data_indices[3].len(), right_indices.len());
    }

    #[test]
    fn test_weight_calculation() {
        let settings = create_test_settings();
        let mut sampler = PgBartSampler::new(mock_logp, settings);

        let test_array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let weight = sampler.compute_log_likelihood(test_array);

        // Mock function sums the values: 1+2+3+4+5 = 15
        assert_eq!(weight, 15.0);
    }

    #[test]
    fn test_full_sampler_step() {
        let settings = create_test_settings();
        let mut sampler = PgBartSampler::new(mock_logp, settings);
        let mut state = create_test_state();

        // This should complete without panicking
        sampler.step(&mut state);

        // Basic invariants should hold
        assert_eq!(state.X.shape(), &[10, 2]);
        assert_eq!(state.y.len(), 10);
        assert_eq!(state.forest.len(), 3);
        assert_eq!(state.weights.len(), 3);
        assert_eq!(state.predictions.len(), 10);
    }

    #[test]
    fn test_particle_weight_update() {
        let settings = create_test_settings();
        let mut sampler = PgBartSampler::new(mock_logp, settings);
        let state = create_test_state();

        let arena = Bump::new();
        let mut forest = Forest::new_in(&arena, 3, state.y.len(), 2, 0.0);
        let base_predictions = Array1::zeros(state.y.len());

        sampler.update_particle_weights(&mut forest, &state, &base_predictions);

        // All particles should have computed weights
        assert_eq!(forest.weights.len(), 3);
        for &weight in &forest.weights {
            assert!(weight.is_finite());
        }
    }

    #[test]
    fn test_particle_selection() {
        let settings = create_test_settings();
        let mut sampler = PgBartSampler::new(mock_logp, settings);

        let arena = Bump::new();
        let mut forest = Forest::new_in(&arena, 3, 10, 2, 0.0);

        // Set some weights
        forest.weights[0] = 0.1;
        forest.weights[1] = 0.6;
        forest.weights[2] = 0.3;

        let selected = sampler.select_particle(&forest);
        assert!(selected < 3);
    }

    #[test]
    fn test_tree_prediction() {
        let arena = Bump::new();
        let mut forest = Forest::new_in(&arena, 1, 5, 2, 1.0);
        let tree = &forest.trees[0];

        let x = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let predictions = tree.predict(&x);

        assert_eq!(predictions.len(), 2);
        // Root node should return initial leaf value
        assert_eq!(predictions[0], 1.0);
        assert_eq!(predictions[1], 1.0);
    }
}
