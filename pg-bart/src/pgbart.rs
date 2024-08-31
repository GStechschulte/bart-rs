use ndarray::{Array1, Array2};

use crate::particle::{Particle, ParticleParams};
use crate::split::{LeafValueSampler, SplitProbabilitySampler};

// Functions that implement the BART Particle Gibbs initialization and update step.
//
// Functions that do Particle Gibbs steps operate by taking as input a PgBartState
// struct, and output a new BART struct with the new state.

pub trait PyData {
    fn X(&self) -> &Array2<f64>;
    fn y(&self) -> &Array1<f64>;
    fn model_logp(&self, v: &Array1<f64>) -> f64;
}

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
    data: Box<dyn PyData>,
    params: PgBartSettings,
    probabilities: Probabilities,
    predictions: Array1<f64>,
    particles: Vec<Particle>,
    variable_inclusion: Vec<usize>,
    tune: bool,
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

        PgBartState {
            data,
            params,
            // probabilities,
            // leaf_sampler,
            predictions,
            particles,
            variable_inclusion,
            tune: false,
        }
    }

    pub fn step(&mut self) {
        todo!("Implement")
    }
}
