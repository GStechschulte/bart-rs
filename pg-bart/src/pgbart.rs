use core::f64;

use ndarray::Array1;

use rand::seq::IteratorRandom;
use rand::thread_rng;

use rand::distributions::WeightedIndex;
use rand_distr::{Distribution, Normal, Uniform};

use crate::data::PyData;
use crate::math;
use crate::particle::{Particle, ParticleParams};
use crate::probabilities::TreeProbabilities;
use crate::tree::DecisionTree;

// Functions that implement the BART Particle Gibbs initialization and update step.
//
// Functions that do Particle Gibbs steps operate by taking as input a PgBartState
// struct, and output a new BART struct with the new state.

/// PgBartSetting are used to initialize a new PgBartState
pub struct PgBartSettings {
    n_trees: usize,
    n_particles: usize,
    alpha: f64,
    default_kf: f64,
    batch: (f64, f64),
    init_alpha_vec: Vec<f64>,
}

impl PgBartSettings {
    pub fn new(
        n_trees: usize,
        n_particles: usize,
        alpha: f64,
        default_kf: f64,
        batch: (f64, f64),
        init_alpha_vec: Vec<f64>,
    ) -> Self {
        Self {
            n_trees,
            n_particles,
            alpha,
            default_kf,
            batch,
            init_alpha_vec,
        }
    }
}

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
    pub fn new(params: PgBartSettings, data: Box<dyn PyData>) -> Self {
        let X = data.X();
        let y = data.y();

        let m = params.n_trees as f64;
        let mu = y.mean().unwrap();
        let leaf_value = mu / m;
        let predictions = Array1::from_elem(y.len(), mu);

        let variable_inclusion = vec![0; X.ncols()];

        let particles = (0..params.n_trees)
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
        let alpha_vec: Vec<f64> = params.init_alpha_vec.clone(); // TODO: remove clone?
        let splitting_probs: Vec<f64> = math::normalized_cumsum(&alpha_vec);
        let probabilities = TreeProbabilities {
            alpha_vec,
            splitting_probs,
            alpha: params.alpha,
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

    pub fn step(&mut self) {
        let batch: (usize, usize) = (
            (self.params.n_trees as f64 * self.params.batch.0).ceil() as usize,
            (self.params.n_trees as f64 * self.params.batch.1).ceil() as usize,
        );

        // Logic for determining how many trees to update in a batch given
        // tuning and the batch size
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

        // TODO: Use Rayon for parallel processing. Would need to refactor to use Arc types, etc.
        // Modify each tree sequentially
        for (iter, tree_id) in tree_ids.enumerate() {
            // Fetch the particle (aka tree) to modify
            let selected_particle = &self.particles[tree_id];

            // Compute the sum of trees without the old tree that we are attempting to replace
            let old_predictions = selected_particle.predict(&self.data.X());
            let predictions_minus_old = &self.predictions - &old_predictions;

            // Initialize local particles. These local particles are to be mutated (grown)
            // Lengths are: self.particles.len() = n_trees and local_particles.len() = n_particles
            let mut local_particles = self.initialize_particles(&old_predictions, mu);

            // while !local_particles.iter().all(|p| p.finished()) {
            //     for p in local_particles.iter_mut().skip(1) {
            //         if p.grow(&self.data.X(), self) {
            //             self.update_weight(p, &local_preds);
            //         }
            //     }

            //     let normalized_weights = self.normalize_weights(&local_particles);
            //     let new_particle = self.select_particle(&local_particles, &normalized_weights);

            //     // Update predictions and variable inclusion
            //     self.predictions += &new_particle.predict(&self.data.X())
            //         - &selected_particle.predict(&self.data.X());
            //     // self.update_variable_inclusion(&new_particle.tree)
            //     self.particles[particle_index] = new_particle;

            // if self.tune {
            //     self.update_probabilities();
            // }
            // }
        }
    }

    fn initialize_particles(&self, sum_trees_noi: &Array1<f64>, mu: f64) -> Vec<Particle> {
        let mut particles: Vec<Particle> = Vec::with_capacity(self.params.n_particles);

        let X = self.data.X();
        let leaf_value = mu / (self.params.n_trees as f64);

        let mut particles: Vec<Particle> = (0..self.params.n_particles)
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

    fn update_weight(&self, particle: &mut Particle, local_preds: &Array1<f64>) {
        let preds = local_preds + &particle.predict(&self.data.X());
        let log_likelihood = self.data.model_logp(preds);
        particle.weight.reset(log_likelihood);
    }

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

    // fn resample_particles(
    //     &self,
    //     particles: Vec<Particle>,
    //     normalized_weights: &[f64],
    // ) -> Vec<Particle> {
    //     let mut rng = thread_rng();
    //     let dist = WeightedIndex::new(normalized_weights).unwrap();
    //     let mut new_particles = Vec::with_capacity(particles.len());
    //     new_particles.push(particles[0].clone());

    //     for _ in 1..particles.len() {
    //         let index = dist.sample(&mut rng);
    //         new_particles.push(particles[index].clone());
    //     }

    //     new_particles
    // }

    // fn select_particle(&self, particles: &[Particle], normalized_weights: &[f64]) -> Particle {
    //     let mut rng = thread_rng();
    //     let dist = WeightedIndex::new(normalized_weights).unwrap();
    //     let index = dist.sample(&mut rng);
    //     particles[index].clone()
    // }

    // fn update_variable_inclusion(&mut self) {
    //     for (i, &count) in self.variable_inclusion.iter().enumerate() {
    //         self.probabilities.alpha_vec[i] += count as f64;
    //     }

    //     self.probabilities.update_splitting_probs();
    // }

    fn num_to_update(&self) -> usize {
        let fraction = if self.tune { 0.5 } else { 0.25 };
        ((self.particles.len() as f64) * fraction).floor() as usize
    }

    pub fn predictions(&self) -> &Array1<f64> {
        &self.predictions
    }
}
