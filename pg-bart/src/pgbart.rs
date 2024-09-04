use ndarray::Array1;

use rand::seq::IteratorRandom;
use rand::thread_rng;

use rand_distr::{Normal, Uniform};

use crate::data::PyData;
use crate::math;
use crate::particle::{Particle, ParticleParams};
use crate::probabilities::TreeProbabilities;

// Functions that implement the BART Particle Gibbs initialization and update step.
//
// Functions that do Particle Gibbs steps operate by taking as input a PgBartState
// struct, and output a new BART struct with the new state.

// Parameter settings are used to initialize a new PgBartState
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
            normal: Normal::new(0.0, std).unwrap(),
            uniform: Uniform::new(0.33, 0.75), // TODO!!!
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
        let num_to_update = self.num_to_update();
        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..self.params.n_trees).choose_multiple(&mut rng, num_to_update);

        for &particle_index in &indices {
            let selected_particle = &self.particles[particle_index];
            // let local_preds = &self.predictions - &selected_particle.predict(&self.data.X);
        }
    }

    fn num_to_update(&self) -> usize {
        let fraction = if self.tune { 0.5 } else { 0.25 };
        ((self.particles.len() as f64) * fraction).floor() as usize
    }
}
