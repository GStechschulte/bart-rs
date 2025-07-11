//! Response strategy implementations for computing leaf (terminal) node values.
//!
//! This module provides different strategies for sampling leaf values in BART trees,
//! including MOTR-BART, TVP-BART, and GP-BART approaches.

use numpy::ndarray::Array1;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal};

use crate::forest::LeafValue;

/// Response method interface for computing leaf (terminal) node values using
/// various methods such as MOTR-BART, TVP-BART, and GP-BART.
pub trait ResponseStrategy {
    /// Sample a leaf value given the data indices that belong to this leaf node
    fn sample_leaf_value(
        &self,
        y: &Array1<f64>,
        data_indices: &[usize],
        rng: &mut SmallRng,
    ) -> LeafValue;

    /// Update internal state if needed (for stateful strategies)
    fn update_state(&mut self, _y: &Array1<f64>, _predictions: &Array1<f64>) {}
}

/// MOTR-BART (Multinomial logit Ordinal Trees for Regression BART)
///
/// For continuous responses, uses a conjugate normal-inverse-gamma prior
/// for the leaf values, with the posterior being analytically tractable.
pub struct MotrStrategy {
    /// Prior mean for leaf values
    pub prior_mean: f64,
    /// Prior precision (inverse variance) for leaf values
    pub prior_precision: f64,
    /// Prior shape parameter for inverse-gamma distribution
    pub prior_alpha: f64,
    /// Prior rate parameter for inverse-gamma distribution
    pub prior_beta: f64,
    /// Noise precision (inverse variance) - can be learned or fixed
    pub noise_precision: f64,
}

impl MotrStrategy {
    pub fn new(
        prior_mean: f64,
        prior_precision: f64,
        prior_alpha: f64,
        prior_beta: f64,
        noise_precision: f64,
    ) -> Self {
        Self {
            prior_mean,
            prior_precision,
            prior_alpha,
            prior_beta,
            noise_precision,
        }
    }
}

impl ResponseStrategy for MotrStrategy {
    fn sample_leaf_value(
        &self,
        y: &Array1<f64>,
        data_indices: &[usize],
        rng: &mut SmallRng,
    ) -> LeafValue {
        if data_indices.is_empty() {
            // If no data points, sample from prior
            let prior_dist =
                Normal::new(self.prior_mean, 1.0 / self.prior_precision.sqrt()).unwrap();
            return prior_dist.sample(rng);
        }

        // Compute sufficient statistics
        let n = data_indices.len() as f64;
        let sum_y: f64 = data_indices.iter().map(|&i| y[i]).sum();
        let mean_y = sum_y / n;

        // Posterior parameters for normal distribution
        let posterior_precision = self.prior_precision + n * self.noise_precision;
        let posterior_mean = (self.prior_precision * self.prior_mean
            + self.noise_precision * sum_y)
            / posterior_precision;
        let posterior_variance = 1.0 / posterior_precision;

        // Sample from posterior
        let posterior_dist = Normal::new(posterior_mean, posterior_variance.sqrt()).unwrap();
        posterior_dist.sample(rng)
    }
}

/// Simple Gaussian response strategy
///
/// Uses a simple normal distribution centered at the empirical mean
/// of the response values in the leaf, with fixed variance.
pub struct GaussianResponseStrategy {
    /// Fixed variance for leaf value sampling
    pub variance: f64,
}

impl GaussianResponseStrategy {
    pub fn new(variance: f64) -> Self {
        Self { variance }
    }
}

impl ResponseStrategy for GaussianResponseStrategy {
    fn sample_leaf_value(
        &self,
        y: &Array1<f64>,
        data_indices: &[usize],
        rng: &mut SmallRng,
    ) -> LeafValue {
        if data_indices.is_empty() {
            // If no data points, sample from N(0, variance)
            let dist = Normal::new(0.0, self.variance.sqrt()).unwrap();
            return dist.sample(rng);
        }

        // Compute empirical mean
        let sum_y: f64 = data_indices.iter().map(|&i| y[i]).sum();
        let mean_y = sum_y / data_indices.len() as f64;

        // Sample around the empirical mean
        let dist = Normal::new(mean_y, self.variance.sqrt()).unwrap();
        dist.sample(rng)
    }
}

/// Enum for dynamic dispatch over response strategies
pub enum ResponseStrategies {
    Motr(MotrStrategy),
    Gaussian(GaussianResponseStrategy),
}

impl ResponseStrategies {
    pub fn sample_leaf_value(
        &self,
        y: &Array1<f64>,
        data_indices: &[usize],
        rng: &mut SmallRng,
    ) -> LeafValue {
        match self {
            ResponseStrategies::Motr(strategy) => strategy.sample_leaf_value(y, data_indices, rng),
            ResponseStrategies::Gaussian(strategy) => {
                strategy.sample_leaf_value(y, data_indices, rng)
            }
        }
    }
}
