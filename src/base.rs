//! High-performance PGBART implementation with optimized algorithms and data structures.
//!
//! This module implements the Particle Gibbs sampler for Bayesian Additive Regression Trees (BART)
#![allow(non_snake_case)]

use core::f64;

use ndarray::{Array1, Array2};
use rand::distributions::WeightedIndex;
use rand_distr::Distribution;

use crate::data::PyData;
use crate::forest::{DecisionTree, Forest, Predict};
use crate::math::{normalized_cumsum, RunningStd};
use crate::ops::{Response, TreeSamplingOps};
use crate::split_rules::SplitRuleType;
use rand::thread_rng;

/// Configuration parameters for PGBART sampler initialization.
///
/// This structure contains all hyperparameters and settings needed to initialize
/// and configure the PGBART sampler. The parameters control tree growth behavior,
/// particle filtering, and splitting strategies.
///
/// # Algorithm Parameters
/// - Tree structure: `n_trees`, `alpha`, `beta` control ensemble size and tree depth
/// - Particle filtering: `n_particles`, `batch` control sampling behavior
/// - Splitting: `split_rules`, `init_alpha_vec` control how trees split
/// - Response: `response`, `leaf_sd`, `n_dim` control leaf value generation
pub struct PgBartSettings {
    /// Number of trees.
    pub n_trees: usize,
    /// Number of particles.
    pub n_particles: usize,
    /// alpha parameter to control node depth.
    pub alpha: f64,
    /// beta parameter to control node depth.
    pub beta: f64,
    /// Leaf node standard deviation.
    pub leaf_sd: Vec<f64>,
    /// Batch size to use during tuning and draws.
    pub batch: (f64, f64),
    /// Initial prior probability over feature splitting probability.
    pub init_alpha_vec: Vec<f64>,
    /// Response strategy for computing leaf node response values.
    pub response: Response,
    /// Split rule strategy to use for sampling threshold (split) values.
    pub split_rules: Vec<SplitRuleType>,
    /// Number of dimensions for multi-output leaf values
    pub n_dim: usize,
}

impl PgBartSettings {
    /// Creates a new `PgBartSettings` with the specified configuration.
    ///
    /// # Parameters
    /// - `n_trees`: Number of trees in the BART ensemble
    /// - `n_particles`: Number of particles for particle filtering
    /// - `alpha`: Tree depth control parameter (usually 0.95)
    /// - `beta`: Tree depth control parameter (usually 2.0)
    /// - `leaf_sd`: Standard deviation(s) for leaf value sampling
    /// - `batch`: (tuning_batch_fraction, sampling_batch_fraction)
    /// - `init_alpha_vec`: Initial splitting probabilities per feature
    /// - `response`: Response computation strategy (Constant or Linear)
    /// - `split_rules`: Split rule for each feature (Continuous or OneHot)
    /// - `n_dim`: Number of dimensions for multi-output responses
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_trees: usize,
        n_particles: usize,
        alpha: f64,
        beta: f64,
        leaf_sd: Vec<f64>,
        batch: (f64, f64),
        init_alpha_vec: Vec<f64>,
        response: Response,
        split_rules: Vec<SplitRuleType>,
        n_dim: usize,
    ) -> Self {
        Self {
            n_trees,
            n_particles,
            alpha,
            beta,
            leaf_sd,
            batch,
            init_alpha_vec,
            response,
            split_rules,
            n_dim,
        }
    }
}

/// Main state structure for the PGBART sampler.
///
/// This structure maintains all state needed for the Particle Gibbs sampler,
/// including the current forest of trees, predictions, and sampling statistics.
/// The implementation is optimized for performance with pre-allocated buffers
/// and cache-friendly data layouts.
///
/// # Key Components
/// - `forest`: Collection of trees with optimized particle management
/// - `predictions`: Current sum of all trees (μ in the algorithm)
/// - `tree_ops`: Sampling operations for tree growth
/// - `variable_inclusion`: Feature usage tracking for importance estimation
///
/// # Usage Pattern
/// ```text
/// 1. Create PgBartState::new(settings, data)
/// 2. Loop: state.step() for each MCMC iteration
/// 3. Access: state.predictions() and state.variable_inclusion()
/// ```
pub struct PgBartState {
    /// User-provided data (from Python).
    pub data: Box<dyn PyData>,
    /// Parameters to initialize `PgBartState`.
    pub params: PgBartSettings,
    /// Container for storing data and functions related to tree
    /// sampling operations.
    pub tree_ops: TreeSamplingOps,
    /// Sum of trees (predictions).
    pub predictions: Array1<f64>,
    /// Particle is a wrapper around a `DecisionTree`.
    pub forest: Forest,
    /// Feature counter for performing feature importance.
    pub variable_inclusion: Vec<i32>,
    /// Whether to perform tuning before drawing from the posterior.
    pub tune: bool,
    /// If tuning is true, then updates the tuning statistics such as
    /// the leaf node value's standard deviation.
    pub tuning_stats: RunningStd,
    /// Manages tree ids during tuning and drawing phases.
    pub lower: usize,
    /// Current iteration of tree growing (includes tuning and draws).
    pub iter: usize,
}

impl PgBartState {
    /// Creates a new `PgBartState` with the given settings and data.
    ///
    /// This initializes all components of the PGBART sampler:
    /// - Creates forest of trees initialized to mean/n_trees
    /// - Sets up tree sampling operations with splitting probabilities
    /// - Initializes predictions to the response mean
    /// - Allocates buffers for variable inclusion tracking
    ///
    /// # Performance Notes
    /// - All major data structures are pre-allocated to avoid runtime allocations
    /// - Forest uses Structure of Arrays layout for cache efficiency
    /// - Tree operations are configured with normalized splitting probabilities
    pub fn new(params: PgBartSettings, data: Box<dyn PyData>) -> Self {
        let X = data.X();
        let y = data.y();

        let mu = y.mean().unwrap();
        let leaf_value = mu / params.n_trees as f64;
        // TODO: ???
        let max_size = 100;

        let predictions = Array1::from_elem(y.len(), mu);

        // Tree sampling operations
        let alpha_vec: Vec<f64> = params.init_alpha_vec.clone();
        let splitting_probs: Vec<f64> = normalized_cumsum(&alpha_vec);

        // Initialize the Forest
        let forest = Forest::new(params.n_trees, X.nrows(), leaf_value, max_size);

        let tree_ops = TreeSamplingOps::new(
            alpha_vec,
            splitting_probs,
            params.alpha,
            params.beta,
        );

        Self {
            data,
            params,
            tree_ops,
            predictions,
            forest,
            variable_inclusion: vec![0; X.ncols()],
            tune: true,
            tuning_stats: RunningStd::new(X.nrows()),
            lower: 0,
            iter: 0,
        }
    }

    /// Performs one step of the PGBART algorithm.
    ///
    /// This implements the main MCMC update step following the algorithm specification:
    /// 1. Determine batch size based on tuning status
    /// 2. For each tree in the batch:
    ///    - Calculate residual (μ_{-i} ← μ - G_i^μ)
    ///    - Initialize particles (first = current tree, rest from scratch)
    ///    - Grow particles with resampling until finished
    ///    - Select new tree via weighted sampling
    ///    - Update sum of trees (μ ← μ_{-i} + G_i^μ)
    ///    - Track variable inclusion if not tuning
    ///
    /// # Algorithm Details
    /// - Batch processing allows updating multiple trees per step
    /// - Particle filtering maintains diversity while exploring tree space
    /// - Systematic resampling provides low-variance estimates
    /// - Reference particle (current tree) ensures non-zero acceptance
    ///
    /// # Performance
    /// - Uses pre-allocated buffers to minimize memory overhead
    /// - SoA data layout enables vectorized weight operations
    /// - Efficient tree growth with breadth-first expansion
    pub fn step(&mut self) {
        // At each step, reset variable inclusion counter to zero
        self.variable_inclusion.fill(0);

        // Logic for determining how many trees to update in a batch given tuning and the
        // batch size
        let batch_size = if self.tune {
            (self.params.n_trees as f64 * self.params.batch.0).ceil() as usize
        } else {
            (self.params.n_trees as f64 * self.params.batch.1).ceil() as usize
        };

        // Determine tree_ids based on tuning status
        let upper = (self.lower + batch_size).min(self.params.n_trees);
        // Determine range of tree_ids based on tuning status
        let tree_ids = self.lower..upper;
        self.lower = if upper < self.params.n_trees {
            upper
        } else {
            0
        };

        let X = self.data.X();
        let mu = self.data.y().mean().unwrap();

        for tree_id in tree_ids {
            self.iter += 1;

            // Step 1: Calculate residual (remove current tree contribution)
            let old_tree_predictions = self.forest.trees[tree_id].predict(&X);
            let mu_minus_i = &self.predictions - &old_tree_predictions;

            // Step 2: Initialize particles (first particle = current tree, rest grown from scratch)
            let mut local_forest = self.initialize_particles(&X, &old_tree_predictions, mu);

            // Step 3: Tree Growth Phase - grow particles until finished
            while local_forest.has_expandable_nodes() {
                local_forest.grow(&X, &mu_minus_i, self);

                // Normalize weights and resample (systematic resampling)
                local_forest.normalize_weights();
                local_forest.resample();
            }

            // Step 4: Tree Selection - final weight normalization and selection
            local_forest.normalize_weights();
            let (selected_tree, selected_weight) = self.select_particle(&mut local_forest);

            // Step 5: Update State
            let new_tree_predictions = selected_tree.predict(&X);
            self.predictions = &mu_minus_i + &new_tree_predictions;

            // Update the global Forest with the selected tree
            self.forest.trees[tree_id] = selected_tree;
            self.forest.weights[tree_id] = selected_weight;

            // Update variable inclusion during non-tuning phase
            if !self.tune {
                self.update_variable_inclusion(tree_id);
            }
        }
    }

    /// Initialize particles for the current tree update.
    ///
    /// Creates a local forest with n_particles trees where:
    /// - First particle = current tree being replaced
    /// - Remaining particles = new trees grown from scratch
    /// - All particles get initial weights computed
    ///
    /// This follows the PGBART algorithm requirement to maintain one
    /// "reference" particle to ensure non-zero acceptance probability.
    ///
    /// # Parameters
    /// - `X`: Feature matrix
    /// - `current_tree_preds`: Predictions from the current tree
    /// - `mu`: Overall response mean
    ///
    /// # Returns
    /// Forest with initialized particles and computed weights
    fn initialize_particles(
        &self,
        X: &Array2<f64>,
        current_tree_preds: &Array1<f64>,
        mu: f64,
    ) -> Forest {
        let leaf_value = mu / (self.params.n_trees as f64);
        let mut local_forest = Forest::new(self.params.n_particles, X.nrows(), leaf_value, 100);

        // First particle is the current tree being replaced
        if !self.forest.trees.is_empty() {
            local_forest.trees[0] = self.forest.trees[0].clone();
        }

        // Initialize weights for all particles
        let residual = &self.predictions - current_tree_preds;
        for (i, tree) in local_forest.trees.iter().enumerate() {
            local_forest.weights[i] = self.update_weight(X, tree, &residual);
            local_forest.particle_arrays.weights[i] = local_forest.weights[i];
        }

        local_forest
    }

    /// Compute the log-likelihood weight for a tree.
    ///
    /// Evaluates the likelihood of the data given the predictions from
    /// this tree added to the local predictions (residuals). This is
    /// the core weighting function for particle filtering.
    ///
    /// # Parameters
    /// - `X`: Feature matrix (for tree prediction)
    /// - `tree`: Tree to evaluate
    /// - `local_preds`: Base predictions (μ_{-i})
    ///
    /// # Returns
    /// Log-likelihood value for this tree
    fn update_weight(
        &self,
        X: &Array2<f64>,
        tree: &DecisionTree,
        local_preds: &Array1<f64>,
    ) -> f64 {
        let preds = local_preds + &tree.predict(X);
        let log_likelihood = self.data.evaluate_logp(preds);
        log_likelihood
    }

    /// Select a particle from the forest using weighted sampling.
    ///
    /// Uses the final normalized weights to select a tree from the particle forest.
    /// This implements the tree selection step of the PGBART algorithm.
    ///
    /// # Parameters
    /// - `forest`: Forest containing particles with normalized weights
    ///
    /// # Returns
    /// Tuple of (selected_tree, selected_weight)
    ///
    /// # Algorithm
    /// Uses weighted random sampling where particles with higher weights
    /// have higher probability of selection. This maintains the correct
    /// sampling distribution for MCMC convergence.
    fn select_particle(&self, forest: &mut Forest) -> (DecisionTree, f64) {
        let mut rng = thread_rng();
        let dist = WeightedIndex::new(&forest.particle_arrays.weights[..forest.trees.len()]).unwrap();
        let index = dist.sample(&mut rng);

        let selected_tree = forest.trees.swap_remove(index);
        let selected_weight = forest.particle_arrays.weights[index];

        (selected_tree, selected_weight)
    }

    /// Update variable inclusion statistics for the selected tree.
    ///
    /// Tracks which features were used for splitting in the selected tree.
    /// This is used during the non-tuning phase to estimate variable importance.
    ///
    /// # Parameters
    /// - `tree_id`: Index of the tree in the forest
    fn update_variable_inclusion(&mut self, tree_id: usize) {
        // Track which features were used for splitting in the selected tree
        for &feature_idx in &self.forest.trees[tree_id].feature {
            if (feature_idx as usize) < self.variable_inclusion.len() {
                self.variable_inclusion[feature_idx as usize] += 1;
            }
        }
    }

    /// Returns the variable inclusion counter.
    ///
    /// During non-tuning phases, this tracks how many times each feature
    /// was used for splitting across all selected trees. Higher values
    /// indicate more important features.
    pub fn variable_inclusion(&self) -> &Vec<i32> {
        &self.variable_inclusion
    }

    /// Returns the current predictions (sum of all trees).
    ///
    /// This is the μ quantity from the algorithm - the sum of predictions
    /// from all trees in the current forest. These are the model's current
    /// predictions for the training data.
    pub fn predictions(&self) -> &Array1<f64> {
        &self.predictions
    }
}


