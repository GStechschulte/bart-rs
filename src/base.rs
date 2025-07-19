use bumpalo::Bump;
use numpy::ndarray::{Array, Ix1, Ix2};

// pub trait SamplingAlgorithm<'arena, S, I> {
//     type Error;

//     fn init(&self, arena: &'arena Bump)
// }

/// Current state for the Particle Gibbs algorithm
#[derive(Debug)]
pub struct PgBartState {
    /// Ensemble of selected particles
    pub forest: Vec<f64>,
    /// Log-likelihood
    pub weights: Vec<f64>,
    /// Sum of particle predictions
    pub predictions: Array<f64, Ix1>,
}

impl PgBartState {
    pub fn new(forest: Vec<f64>, weights: Vec<f64>, predictions: Array<f64, Ix1>) -> Self {
        Self {
            forest: forest,
            weights: weights,
            predictions: predictions,
        }
    }
}
