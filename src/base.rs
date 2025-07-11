use numpy::{
    ndarray::{s, Array, Axis, Ix, Ix1, Ix2},
    IxDyn, PyArrayMethods, PyReadonlyArray,
};

/// Current state for the Particle Gibbs algorithm
#[derive(Debug)]
pub struct PgBartState {
    /// User-provided design matrix
    pub X: Array<f64, Ix2>,
    /// User-provided response (target) vector
    pub y: Array<f64, Ix1>,
    /// Ensemble of selected particles
    pub forest: Vec<f64>,
    /// Log-likelihood
    pub weights: Vec<f64>,
    /// Sum of particle predictions
    pub predictions: Array<f64, Ix1>,
}

impl PgBartState {
    pub fn new(
        X: Array<f64, Ix2>,
        y: Array<f64, Ix1>,
        forest: Vec<f64>,
        weights: Vec<f64>,
        predictions: Array<f64, Ix1>,
    ) -> Self {
        Self {
            X: X,
            y: y,
            forest: forest,
            weights: weights,
            predictions: predictions,
        }
    }
}
