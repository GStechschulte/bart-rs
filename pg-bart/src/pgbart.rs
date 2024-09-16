use core::f64;

use ndarray::Array1;

use rand::{thread_rng, Rng};

use rand::distributions::WeightedIndex;
use rand_distr::{Distribution, Normal, Uniform};

use crate::data::PyData;
use crate::math;
use crate::particle::{Particle, ParticleParams};
use crate::probabilities::TreeProbabilities;

// Functions that implement the BART Particle Gibbs initialization and update step.
//
// Functions that do Particle Gibbs steps operate by taking as input a PgBartState
// struct, and then iterate (step) on this PgBartState.

// PgBartSetting are used to initialize a new PgBartState
pub struct PgBartSettings {
    n_trees: usize,
    n_particles: usize,
    alpha: f64,
    beta: f64,
    default_kf: f64,
    batch: (f64, f64),
    init_alpha_vec: Vec<f64>,
}

impl PgBartSettings {
    pub fn new(
        n_trees: usize,
        n_particles: usize,
        alpha: f64,
        beta: f64,
        default_kf: f64,
        batch: (f64, f64),
        init_alpha_vec: Vec<f64>,
    ) -> Self {
        Self {
            n_trees,
            n_particles,
            alpha,
            beta,
            default_kf,
            batch,
            init_alpha_vec,
        }
    }
}

/// PgBartState is the main entry point of the Particle-Gibbs sampler
/// for Bayesian Additive Regression Trees (BART).
pub struct PgBartState {
    pub data: Box<dyn PyData>,
    pub params: PgBartSettings,
    pub probabilities: TreeProbabilities,
    pub predictions: Array1<f64>,
    pub particles: Vec<Particle>,
    pub variable_inclusion: Vec<usize>,
    pub tune: bool,
}

impl PgBartState {
    /// Creates a PgBartState with the given PgBartSettings and PyData.
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
    ///     kfactor,
    ///     batch,
    ///     split_prior.to_vec().unwrap(),
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
        let mut particles = (0..params.n_trees)
            .map(|_| {
                let p_params = ParticleParams::new(X.nrows(), X.ncols(), params.default_kf);
                Particle::new(p_params, leaf_value, X.nrows())
            })
            .collect();

        // Standard deviation for binary and continuous data
        let binary = y.iter().all(|v| (*v == 0.0) || (*v == 1.0));
        let std = if binary {
            3.0 / m.powf(0.5)
        } else {
            y.std(1.0)
        };

        let N = Normal::new(0.0, std).unwrap();

        // Tree probabilities
        let alpha_vec: Vec<f64> = params.init_alpha_vec.clone(); // TODO: Remove clone?
        let splitting_probs: Vec<f64> = math::normalized_cumsum(&alpha_vec);
        let probabilities = TreeProbabilities {
            alpha_vec,
            splitting_probs,
            alpha: params.alpha,
            beta: params.beta,
            normal: Normal::new(0.0, std).unwrap(), // TODO: Should mu be fixed?
            uniform: Uniform::new(0.33, 0.75),      // TODO: Should these params. be fixed?
        };

        PgBartState {
            data,
            params,
            probabilities,
            predictions,
            particles,
            variable_inclusion,
            tune: false,
        }
    }

    /// Runs the Particle Gibbs sampler sequentially for M iterations where M is the number
    /// of trees.
    ///
    /// A single step will initialize a set of particles N, of which one will replace the
    /// current tree M_i. To decide which particle will replace the current tree, the N
    /// particles are grown until the probability of a leaf node expanding is less than a
    /// random value in the interval [0, 1].
    ///
    /// The grown particles are then resampled according to their log-likelihood, of which
    /// one is selected to replace the current tree M_i.
    pub fn step(&mut self) {
        let batch: (usize, usize) = (
            (self.params.n_trees as f64 * self.params.batch.0).ceil() as usize,
            (self.params.n_trees as f64 * self.params.batch.1).ceil() as usize,
        );

        // Logic for determining how many trees to update in a batch given tuning and the
        // batch size
        let batch_size = if self.tune { batch.0 } else { batch.1 };
        let lower: usize = 0;
        let upper = (lower as f64 + batch_size as f64).floor() as usize;
        let tree_ids = lower..upper;
        let lower = if upper >= self.params.n_trees {
            0
        } else {
            upper
        };

        let mu = self.data.y().mean().unwrap() / (self.params.n_particles as f64);

        // TODO: Use Rayon for parallel processing (would need to refactor to use Arc types...)
        // Modify each tree sequentially
        for (iter, tree_id) in tree_ids.enumerate() {
            // Immutable borrow of the particle (aka tree) to modify
            let selected_particle = &self.particles[tree_id];

            // Compute the sum of trees without the old particle we are attempting to replace
            let old_predictions = selected_particle.predict(&self.data.X());
            let predictions_minus_old = &self.predictions - &old_predictions;

            // Initialize local particles. These local particles are to be mutated (grown)
            // Lengths are: self.particles.len() = n_trees and local_particles.len() = n_particles
            let mut local_particles = self.initialize_particles(&old_predictions, mu);

            // Grow each particle (skipping the first) until the probability that the node
            // in this particle will remain a leaf node is "high"
            while !local_particles.iter().all(|p| p.finished()) {
                for p in local_particles.iter_mut().skip(1) {
                    if p.grow(&self.data.X(), self) {
                        self.update_weight(p, &old_predictions);
                    }
                }
            }
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

            // Replace particle (tree) M_i with the new particle
            self.particles[tree_id] = new_particle;

            println!("Done...");

            // TODO: !!!!
            // Update variable inclusion
            // self.update_variable_inclusion(&new_particle.tree)

            // if self.tune {
            //     self.update_probabilities();
            //     }
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
                let p_params = ParticleParams::new(X.nrows(), X.ncols(), self.params.default_kf);
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
        let log_likelihood = self.data.model_logp(preds);
        particle.weight.reset(log_likelihood);
    }

    /// Ensures the weights (log-likelihood) of the Particles sums to 1.
    fn normalize_weights(&self, particles: &[Particle]) -> Vec<f64> {
        let log_weights: Vec<f64> = particles.iter().map(|p| p.weight.log_w()).collect();

        let max_log_weight = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let weights: Vec<f64> = log_weights
            .iter()
            .map(|&w| (w - max_log_weight).exp())
            .collect();

        let sum_weights: f64 = weights.iter().sum();

        weights.iter().map(|&w| w / sum_weights).collect()
    }

    /// Uses systematic resampling to sample new Particles.
    fn resample_particles(&self, particles: &mut Vec<Particle>, weights: &[f64]) -> Vec<Particle> {
        let num_particles = particles.len();

        // Keep first particle (original tree)
        let mut resampled_particles: Vec<Particle> = Vec::with_capacity(num_particles);
        resampled_particles.push(particles.remove(0));

        // Systematic resample all but the first particle
        let resampled_indices = self.systematic_resample(&weights[1..], num_particles - 1);

        // Transfer ownership of Particles in particles to resampled_particles
        for &idx in &resampled_indices {
            // swap_remove does not preserve ordering but this is okay since we don't use
            // particles (local_particles) after calling resample_particles
            resampled_particles.push(particles.swap_remove(idx + 1));
        }

        resampled_particles
    }

    /// Systematic resampling using weights and number of particles to return
    /// indices of the Particles.
    ///
    /// Note: adapted from https://github.com/nchopin/particles
    fn systematic_resample(&self, weights: &[f64], num_samples: usize) -> Vec<usize> {
        // Generate a uniform ranom number and use it to create evenly spaced points
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

    /// Get predictions
    pub fn predictions(&self) -> &Array1<f64> {
        &self.predictions
    }
}
