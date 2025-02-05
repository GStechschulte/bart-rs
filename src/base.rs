//! Functions that implement the BART Particle Gibbs initialization and update step
//! to grow a decision tree.
//!
//! Functions that do Particle Gibbs steps operate by taking as input a PgBartState
//! struct, and then iterate (step) on this PgBartState.
#![allow(non_snake_case)]

use core::f64;
use std::collections::HashMap;
use std::iter::from_fn;

use ndarray::{Array1, Array2};
use rand::distributions::WeightedIndex;
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

use crate::data::PyData;
use crate::forest::{DecisionTree, Forest, Predict};
use crate::math::{normalized_cumsum, RunningStd};
use crate::ops::{Response, TreeSamplingOps};
// use crate::pgbart::select_particle;
use crate::split_rules::SplitRuleType;

/// PgBartSetting are parameters used to initialize a new `PgBartState`.
///
/// `split_rules` is a vector of `SplitRuleType` enum variants as the user
/// may pass different split rule types.
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
    /// Creates a new `PgBartSettings` struct to be used in BART for growing
    /// particle trees.
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

/// PgBartState is the main entry point of the Particle-Gibbs sampler for BART.
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
    /// Creates a `PgBartState` with the given `PgBartSettings` and `PyData`.
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

        let tree_ops = TreeSamplingOps {
            alpha_vec,
            splitting_probs,
            alpha: params.alpha,
            beta: params.beta,
            normal: Normal::new(0.0, 1.0).unwrap(),
        };

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
        // let leaf_value = mu / (self.params.n_trees as f64);

        // println!("tree_ids: {:?}", tree_ids);

        for tree_id in tree_ids {
            self.iter += 1;

            let old_particle_predictions = self.forest.trees[tree_id].predict(&X);
            let local_particle_predictions = &self.predictions - &old_particle_predictions;

            let mut local_forest = self.initialize_particles(&X, &old_particle_predictions, mu);

            // Grow each candidate tree until there are no more expandable nodes
            //
            // At _this_ growth iteration, use the state from the previous iteration
            while local_forest.has_expandable_nodes() {
                local_forest.grow(&X, &local_particle_predictions, self);
                // println!(
                //     "forest weights (before normalizing): {:?}",
                //     local_forest.weights
                // );
                // After this grow iteration, normalize weights and resample
                local_forest.normalize_weights();
                // println!(
                //     "forest weights (after normalizing): {:?}",
                //     local_forest.weights
                // );
                local_forest.resample();
            }

            // Select the best candidate tree to replace the current tree
            local_forest.normalize_weights();
            let (particle, weight) = select_particle(&mut local_forest);

            // Update sum of trees (predictions)
            let new_predictions = particle.predict(&X);
            // println!("new_predictions: {:?}", new_predictions);
            self.predictions = local_particle_predictions + new_predictions;
            // println!("self.predictions: {:?}", self.predictions);

            // Update the global Forest with the selected tree
            self.forest.trees[tree_id] = particle;
            self.forest.weights[tree_id] = weight;

            // break;
        }
    }

    fn initialize_particles(
        &self,
        X: &Array2<f64>,
        sum_trees_noi: &Array1<f64>,
        mu: f64,
    ) -> Forest {
        let leaf_value = mu / (self.params.n_trees as f64);
        let mut local_particles = Forest::new(self.params.n_particles, X.nrows(), leaf_value, 100);

        for (weight, tree) in local_particles
            .weights
            .iter_mut()
            .zip(local_particles.trees.iter_mut())
        {
            *weight = self.update_weight(X, tree, sum_trees_noi);
        }

        local_particles
    }

    fn update_weight(
        &self,
        X: &Array2<f64>,
        particle: &mut DecisionTree,
        local_preds: &Array1<f64>,
    ) -> f64 {
        let preds = local_preds + &particle.predict(X);
        let log_likelihood = self.data.evaluate_logp(preds);

        log_likelihood
    }

    pub fn variable_inclusion(&self) -> &Vec<i32> {
        &self.variable_inclusion
    }

    pub fn predictions(&self) -> &Array1<f64> {
        &self.predictions
    }
}

fn select_particle(forest: &mut Forest) -> (DecisionTree, f64) {
    // println!("--- select_particle ---");
    let mut rng = thread_rng();
    // println!("select_particle forest weights: {:?}", forest.weights);
    let dist = WeightedIndex::new(&forest.weights).unwrap();
    let index = dist.sample(&mut rng);
    // println!("sampled index: {}", index);

    (forest.trees.swap_remove(index), forest.weights[index])
}
