//! Functions that implement the BART Particle Gibbs initialization and update step.
//!
//! Functions that do Particle Gibbs steps operate by taking as input a PgBartState
//! struct, and then iterate (step) on this PgBartState.

#![allow(non_snake_case)]

use crate::data::PyData;
use crate::math::{normalized_cumsum, RunningStd};
use crate::ops::{Response, TreeSamplingOps};
use crate::particle::{Particle, ParticleParams};
use crate::split_rules::SplitRuleType;

use core::f64;

use ndarray::Array1;
use rand::distributions::WeightedIndex;
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal, Uniform};

/// PgBartSetting are used to initialize a new PgBartState
///
/// `split_rules` is a vector of `SplitRuleType` enum variants as the user
/// may pass different split rule types.
pub struct PgBartSettings {
    pub n_trees: usize,
    pub n_particles: usize,
    pub alpha: f64,
    pub beta: f64,
    pub leaf_sd: f64,
    pub batch: (f64, f64),
    pub init_alpha_vec: Vec<f64>,
    pub response: Response,
    pub split_rules: Vec<SplitRuleType>,
}

impl PgBartSettings {
    pub fn new(
        n_trees: usize,
        n_particles: usize,
        alpha: f64,
        beta: f64,
        leaf_sd: f64,
        batch: (f64, f64),
        init_alpha_vec: Vec<f64>,
        response: Response,
        split_rules: Vec<SplitRuleType>,
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
        }
    }
}

/// PgBartState is the main entry point of the Particle-Gibbs sampler
/// for Bayesian Additive Regression Trees (BART).
pub struct PgBartState {
    pub data: Box<dyn PyData>,
    pub params: PgBartSettings,
    pub tree_ops: TreeSamplingOps,
    pub predictions: Array1<f64>,
    pub particles: Vec<Particle>,
    pub variable_inclusion: Vec<i32>,
    pub tune: bool,
    pub tuning_stats: RunningStd,
    pub lower: usize, // lower manages tree ids during tuning and drawing phases
    pub iter: usize,
}

impl PgBartState {
    /// Creates a `PgBartState` with the given `PgBartSettings` and `PyData`.
    ///
    /// # Examples
    ///
    /// ```
    /// // data is ExternalData from the Python user
    /// let data = Box::new(data);
    /// // PgBartSettings are passed from the Python user
    /// let params = PgBartSettings::new(
    ///     n_trees,
    ///     n_particles,
    ///     alpha,
    ///     batch,
    ///     split_prior.to_vec().unwrap(),
    ///     response,
    ///     );
    /// let state = PgBartState::new(params, data);
    /// ```
    pub fn new(params: PgBartSettings, data: Box<dyn PyData>) -> Self {
        let X = data.X();
        let y = data.y();

        let m = params.n_trees as f64;
        let mu = y.mean().unwrap();
        let leaf_value = mu / m;
        let predictions = Array1::from_elem(y.len(), mu);
        let variable_inclusion = vec![0; X.ncols()];

        // Particles can grow (mutate)
        let particles = (0..params.n_trees)
            .map(|_| {
                let p_params = ParticleParams::new(X.nrows(), X.ncols());
                Particle::new(p_params, leaf_value, X.nrows())
            })
            .collect();

        // Tree sampling operations
        let alpha_vec: Vec<f64> = params.init_alpha_vec.clone();
        let splitting_probs: Vec<f64> = normalized_cumsum(&alpha_vec);

        let tree_ops = TreeSamplingOps {
            alpha_vec,
            splitting_probs,
            alpha: params.alpha,
            beta: params.beta,
            normal: Normal::new(0.0, 1.0).unwrap(),
            uniform: Uniform::new(0.0, 1.0),
        };

        Self {
            data,
            params,
            tree_ops,
            predictions,
            particles,
            variable_inclusion,
            tune: true,
            tuning_stats: RunningStd::new(X.nrows()),
            lower: 0,
            iter: 1,
        }
    }

    /// Runs the Particle Gibbs sampler sequentially for `M` iterations where `M` is the number
    /// of trees.
    ///
    /// A single step will initialize a set of particles `N`, of which one will replace the
    /// current tree `M_i`. To decide which particle will replace the current tree, the `N`
    /// particles are grown until the probability of a leaf node expanding is less than a
    /// random value in the interval [0, 1].
    ///
    /// The grown particles are then resampled according to their log-likelihood, of which
    /// one is selected to replace the current tree `M_i`.
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

        let mu = self.data.y().mean().unwrap();

        // Mutate each tree sequentially
        for tree_id in tree_ids {
            self.iter += 1;
            // Immutable borrow of the particle (aka tree) to modify
            let selected_particle = &self.particles[tree_id];

            // Compute the sum of trees without the old particle we are attempting to replace
            let old_predictions = selected_particle.predict(&self.data.X());
            let predictions_minus_old = &self.predictions - &old_predictions;

            // Initialize local particles. These local particles are to be mutated (grown)
            // Lengths are: self.particles.len() = n_trees and local_particles.len() = n_particles
            let mut local_particles = self.initialize_particles(&old_predictions, mu);

            // Grow each particle until the probability that the node in this particle
            // will remain a leaf node is "high"
            local_particles.iter_mut().skip(1).for_each(|particle| {
                while !particle.finished() {
                    // Attempt to grow the particle
                    if particle.grow(&self.data.X(), self) {
                        self.update_weight(particle, &predictions_minus_old);
                    }
                }
            });

            // Normalize log-likelihood and resample particles
            let normalized_weights = self.normalize_weights(&local_particles);

            let mut resampled_particles =
                self.resample_particles(&mut local_particles, &normalized_weights);

            // Normalize log-likelihood again and select a particle to replace M_i
            let normalized_weights = self.normalize_weights(&resampled_particles);
            let new_particle = self.select_particle(&mut resampled_particles, &normalized_weights);

            // Update the sum of trees
            let new_particle_preds = &new_particle.predict(&self.data.X());
            let updated_preds = predictions_minus_old + new_particle_preds;

            self.predictions = updated_preds;

            // During tuning, update feature split probability and leaf standard deviation
            if self.tune {
                self.update_splitting_probability(&new_particle);

                // TODO!!!
                // if self.iter > 2 {
                //     self.params.leaf_sd = self.tuning_stats.update(&new_particle_preds.to_vec())[0];
                //     println!("leaf_sd: {}", self.params.leaf_sd);
                // } else {
                //     self.tuning_stats.update(&new_particle_preds.to_vec());
                // }
            } else {
                self.update_variable_inclusion(&new_particle);
            }
            // println!("variable_inclusion: {:?}", self.variable_inclusion);

            // Replace tree M_i with the new particle
            self.particles[tree_id] = new_particle;
        }
    }

    /// Generate an initial set of particles for _this_ tree.
    fn initialize_particles(&self, sum_trees_noi: &Array1<f64>, mu: f64) -> Vec<Particle> {
        let X = self.data.X();
        let leaf_value = mu / (self.params.n_trees as f64);

        // Create a new vector of Particles with the same ParticleParams passed to
        // PgBartState::new(...)
        let particles: Vec<Particle> = (0..self.params.n_particles)
            .map(|i| {
                let p_params = ParticleParams::new(X.nrows(), X.ncols());
                let mut particle = Particle::new(p_params, leaf_value, X.nrows());

                if i == 0 {
                    self.update_weight(&mut particle, sum_trees_noi);
                }

                particle
            })
            .collect();

        particles
    }

    /// Update the weight (log-likelihood) of a Particle.
    fn update_weight(&self, particle: &mut Particle, local_preds: &Array1<f64>) {
        // To update the weight, the grown Particle needs to make predictions
        let preds = local_preds + &particle.predict(&self.data.X());
        let log_likelihood = self.data.evaluate_logp(preds);

        particle.weight.set(log_likelihood);
    }

    /// Normalize Particle weights to be between [0, 1] using the Softmax function.
    ///
    /// The Softmax function is implemented using the log-sum-exp trick to ensure
    /// the normalization of particle weights is numerically stable.
    fn normalize_weights(&self, particles: &[Particle]) -> Vec<f64> {
        let log_weights: Vec<f64> = particles.iter().map(|p| p.weight.log_w).collect();

        let max_log_weight = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_shifted: Vec<f64> = log_weights
            .iter()
            .map(|&w| (w - max_log_weight).exp())
            .collect();

        let sum_exp: f64 = exp_shifted.iter().sum();

        exp_shifted.iter().map(|&w| w / sum_exp).collect()
    }

    /// Systematic resampling to sample new Particles.
    fn resample_particles(&self, particles: &mut Vec<Particle>, weights: &[f64]) -> Vec<Particle> {
        let num_particles = particles.len();

        // Keep first particle (original tree) and get resampled indices
        let mut resampled_particles = vec![particles.swap_remove(0)];
        let resampled_indices = self.systematic_resample(&weights[1..], num_particles - 1);

        // Collect resampled particles
        resampled_particles.extend(
            resampled_indices
                .into_iter()
                .filter_map(|idx| (idx < particles.len()).then(|| particles.swap_remove(idx))),
        );

        // Remove remaining elements (because of the logic inside of .filter_map above) in particles
        // and add to resampled_particles
        resampled_particles.extend(particles.drain(..));

        resampled_particles
    }

    /// Systematic resampling using weights and number of particles to return
    /// indices of the Particles.
    ///
    /// Note: adapted from https://github.com/nchopin/particles
    fn systematic_resample(&self, weights: &[f64], num_samples: usize) -> Vec<usize> {
        // Generate a uniform random number and use it to create evenly spaced points
        let mut rng = rand::thread_rng();
        let u = rng.gen::<f64>() / num_samples as f64;

        let cumulative_sum = weights
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect::<Vec<f64>>();

        // Find the indices where the cumulative sum exceeds the evenly spaced points
        let mut indices = Vec::with_capacity(num_samples);
        let mut j = 0;
        for i in 0..num_samples {
            while j < cumulative_sum.len() && cumulative_sum[j] < u + i as f64 / num_samples as f64
            {
                j += 1;
            }
            indices.push(j);
        }

        indices
    }

    /// Sample a Particle proportional to its weight.
    fn select_particle(&self, particles: &mut Vec<Particle>, weights: &[f64]) -> Particle {
        let mut rng = thread_rng();
        let dist = WeightedIndex::new(weights).unwrap();
        let index = dist.sample(&mut rng);

        // Remove and return the selected particle, transferring ownership
        particles.swap_remove(index)
    }

    /// Updates the probabilities of sampling each covariate if in the tuning phase
    fn update_splitting_probability(&mut self, particle: &Particle) {
        self.tree_ops.splitting_probs = normalized_cumsum(&self.tree_ops.alpha_vec);

        particle.tree.feature.iter().for_each(|&idx| {
            if let Some(alpha) = self.tree_ops.alpha_vec.get_mut(idx as usize) {
                *alpha += 1.0;
            }
        });
    }

    pub fn update_variable_inclusion(&mut self, particle: &Particle) {
        particle.tree.feature.iter().for_each(|&idx| {
            self.variable_inclusion[idx] += 1;
        });
    }

    /// Returns variable inclusion counter.
    pub fn variable_inclusion(&self) -> &Vec<i32> {
        &self.variable_inclusion
    }

    /// Returns a borrowed reference to predictions (sum of trees).
    pub fn predictions(&self) -> &Array1<f64> {
        &self.predictions
    }
}
