//! Response strategy implementations for computing leaf (terminal) node values.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Response method interface for computing leaf values.
pub trait ResponseStrategy {
    fn sample_leaf_value(
        &self,
        rng: &mut impl Rng,
        residuals: &[f64],
        sigma: f64,
        n_trees: usize,
    ) -> f64;
}

/// Gaussian response strategy.
///
/// Samples from a normal distribution centered at the empirical mean
/// of the residuals in the leaf, with fixed variance.
#[derive(Clone, Copy, Debug)]
pub struct GaussianResponseStrategy;

impl ResponseStrategy for GaussianResponseStrategy {
    fn sample_leaf_value(
        &self,
        rng: &mut impl Rng,
        residuals: &[f64],
        sigma: f64,
        n_trees: usize,
    ) -> f64 {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let noise = dist.sample(rng) * sigma;

        if residuals.is_empty() {
            return noise;
        }

        let sum: f64 = residuals.iter().sum();
        let mean = sum / residuals.len() as f64 / n_trees as f64;
        mean + noise
    }
}

/// MOTR-BART response strategy (placeholder).
#[derive(Clone, Copy, Debug)]
pub struct MotrStrategy;

impl ResponseStrategy for MotrStrategy {
    fn sample_leaf_value(
        &self,
        _rng: &mut impl Rng,
        _residuals: &[f64],
        _sigma: f64,
        _n_trees: usize,
    ) -> f64 {
        todo!("MotrStrategy not yet implemented")
    }
}

/// Enum for dynamic dispatch over response strategies.
#[derive(Clone, Debug)]
pub enum ResponseStrategies {
    Motr(MotrStrategy),
    Gaussian(GaussianResponseStrategy),
}

impl ResponseStrategies {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.to_lowercase().as_str() {
            "gaussian" => Ok(ResponseStrategies::Gaussian(GaussianResponseStrategy)),
            "motr" => Ok(ResponseStrategies::Motr(MotrStrategy)),
            _ => Err(format!(
                "Unknown response strategy: '{}'. Supported: 'gaussian', 'motr'.",
                name
            )),
        }
    }
}

impl ResponseStrategy for ResponseStrategies {
    fn sample_leaf_value(
        &self,
        rng: &mut impl Rng,
        residuals: &[f64],
        sigma: f64,
        n_trees: usize,
    ) -> f64 {
        match self {
            ResponseStrategies::Motr(s) => s.sample_leaf_value(rng, residuals, sigma, n_trees),
            ResponseStrategies::Gaussian(s) => s.sample_leaf_value(rng, residuals, sigma, n_trees),
        }
    }
}
