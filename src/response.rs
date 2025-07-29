//! Response strategy implementations for computing leaf (terminal) node values.
//!
//! This module provides different strategies for sampling leaf values in BART trees,
//! including MOTR-BART, TVP-BART, and GP-BART approaches.

use numpy::ndarray::Array1;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::{particle::LeafVal, update::TreeContext};

/// Response method interface for computing leaf (terminal) node values using
/// various methods such as MOTR-BART, TVP-BART, and GP-BART.
pub trait ResponseStrategy {
    type Context;

    /// Sample a leaf value given the data indices that belong to this leaf node.
    fn sample_leaf_value(
        &self,
        rng: &mut impl Rng,
        y_data: &Array1<f64>,
        data_indices: &[usize],
        n_trees: usize,
    ) -> LeafVal;

    /// Update internal state if needed (for stateful strategies)
    fn update_state(&mut self, _y: &Array1<f64>, _predictions: &Array1<f64>) {}
}

/// MOTR-BART (Multinomial logit Ordinal Trees for Regression BART).
///
/// For continuous responses, uses a conjugate normal-inverse-gamma prior
/// for the leaf values, with the posterior being analytically tractable.
#[derive(Clone, Copy, Debug)]
pub struct MotrStrategy;

impl ResponseStrategy for MotrStrategy {
    type Context = TreeContext;

    fn sample_leaf_value(
        &self,
        rng: &mut impl Rng,
        y_data: &Array1<f64>,
        data_indices: &[usize],
        n_trees: usize,
    ) -> LeafVal {
        todo!("Not implemented")
    }
}

/// Gaussian response strategy.
///
/// Uses a simple normal distribution centered at the empirical mean
/// of the response values in the leaf, with fixed variance.
#[derive(Clone, Copy, Debug)]
pub struct GaussianResponseStrategy;

impl ResponseStrategy for GaussianResponseStrategy {
    type Context = TreeContext;

    fn sample_leaf_value(
        &self,
        rng: &mut impl Rng,
        y_data: &Array1<f64>,
        data_indices: &[usize],
        n_trees: usize,
    ) -> LeafVal {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let norm = dist.sample(rng);

        if data_indices.is_empty() {
            // If no data points, sample from N(0, variance)
            let dist = Normal::new(0.0, 1.0).unwrap();
            return dist.sample(rng);
        }

        // Compute empirical mean
        let sum_y: f64 = data_indices.iter().map(|&i| y_data[i]).sum();
        let mean_y = sum_y / y_data.len() as f64 / n_trees as f64 + norm;

        mean_y
    }
}

/// Enum for dynamic dispatch over response strategies
#[derive(Clone, Debug)]
pub enum ResponseStrategies {
    Motr(MotrStrategy),
    Gaussian(GaussianResponseStrategy),
}

impl ResponseStrategies {
    pub fn from_str(response_name: &str) -> PyResult<Self> {
        match response_name.to_lowercase().as_str() {
            "gaussian" => Ok(ResponseStrategies::Gaussian(GaussianResponseStrategy)),
            "motr" => Ok(ResponseStrategies::Motr(MotrStrategy)),
            _ => Err(PyValueError::new_err(format!(
                "Unknown split rule: '{}'. Supported split rules are 'ContinuousSplit' and 'OneHotSplit'.",
                response_name
            ))),
        }
    }

    pub fn sample_leaf_value(
        &self,
        rng: &mut impl Rng,
        y_data: &Array1<f64>,
        data_indices: &[usize],
        n_trees: usize,
    ) -> LeafVal {
        match self {
            ResponseStrategies::Motr(strategy) => {
                strategy.sample_leaf_value(rng, y_data, data_indices, n_trees)
            }
            ResponseStrategies::Gaussian(strategy) => {
                strategy.sample_leaf_value(rng, y_data, data_indices, n_trees)
            }
        }
    }
}
