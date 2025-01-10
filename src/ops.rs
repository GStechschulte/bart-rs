//! Implements tree sampling operations and response strategies for decision
//! trees.
//!
//! This module provides functionality for:
//! - Sampling decisions in tree construction (split/leaf nodes)
//! - Computing leaf node values using different response strategies
//! - Managing tree sampling operations with configurable parameters
//!
//! The module implements two main response strategies:
//! - Constant: Computes the mean response normalized by group size
//! - Linear: Implements linear response calculations with special handling for different input sizes
//!
//! The `TreeSamplingOps` struct provides methods for:
//! - Sampling expansion decisions for tree nodes
//! - Computing leaf values with Gaussian noise
//! - Selecting features for splitting nodes

use std::str::FromStr;

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::{self, thread_rng, Rng};
use rand_distr::{Distribution, Normal};

/// Variants indicate the types of response strategies to compute leaf node values.
#[derive(Debug)]
pub enum Response {
    /// Constant implements the `ConstantResponse` strategy.
    Constant(ConstantResponse),
    /// Linear implements the `LinearResponse` strategy.
    Linear(LinearResponse),
}

impl FromStr for Response {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "constant" => Ok(Response::Constant(ConstantResponse)),
            "linear" => Ok(Response::Linear(LinearResponse)),
            _ => Err(format!("Unknown response type: {}", s)),
        }
    }
}

trait ResponseStrategy {
    fn compute_leaf_value(
        &self,
        mu: ArrayView2<f64>,
        m: usize,
        norm: &Array1<f64>,
        n_dims: usize,
    ) -> Array1<f64>;
}

impl ResponseStrategy for Response {
    /// Calls the corresponding `compute_leaf_value` for each `Response` variant
    fn compute_leaf_value(
        &self,
        mu: ArrayView2<f64>,
        m: usize,
        norm: &Array1<f64>,
        n_dims: usize,
    ) -> Array1<f64> {
        match self {
            Response::Constant(constant) => constant.compute_leaf_value(mu, m, norm, n_dims),
            Response::Linear(linear) => linear.compute_leaf_value(mu, m, norm, n_dims),
        }
    }
}

// Use unit structs to define an enum variant constructor for which the
// ResponseStrategy trait can be implemented for as traits cannot be
// implemented for enum variants as a variant is not a type

/// Leaf node value is the mean of `mu`.
#[derive(Debug)]
pub struct ConstantResponse;
impl ResponseStrategy for ConstantResponse {
    fn compute_leaf_value(
        &self,
        mu: ArrayView2<f64>,
        m: usize,
        norm: &Array1<f64>,
        n_dims: usize,
    ) -> Array1<f64> {
        // Compute mean across samples for each dimension
        let means = mu.mean_axis(Axis(0)).unwrap();
        means.mapv(|x| x / m as f64) + norm
    }
}

/// Leaf node value is computed using least squares.
#[derive(Debug)]
pub struct LinearResponse;
impl ResponseStrategy for LinearResponse {
    fn compute_leaf_value(
        &self,
        mu: ArrayView2<f64>,
        m: usize,
        norm: &Array1<f64>,
        n_dims: usize,
    ) -> Array1<f64> {
        match mu.nrows() {
            2 => {
                // Sum values for each dimension then normalize
                let sum = mu.sum_axis(Axis(1));
                sum.mapv(|x| x / (2.0 * m as f64)) + norm
            }
            _len @ 3.. => todo!("Implement fast_linear_fit for multi-dimensional case."),
            _ => unreachable!("Linear response requires at least 2 values per dimension."),
        }
    }
}

/// Holds parameters and distributions used for sampling-related operations
/// of BART.
pub struct TreeSamplingOps {
    /// Normal distribution to sample Gaussian distributed leaf values.
    pub normal: Normal<f64>,
    /// Initial prior probability over a feature being used as a splitting variable.
    pub alpha_vec: Vec<f64>,
    /// Prior probability over a feature being used as a splitting variable.
    pub splitting_probs: Vec<f64>,
    /// Parameter contained with [0, 1] used to control node depth during the
    /// growing of trees.
    pub alpha: f64,
    /// Parameter contained with [0, infinity] used to control node depth during the
    /// growing of trees.
    pub beta: f64,
}

impl TreeSamplingOps {
    /// Sample a boolean flag indicating if a node should be split or not.
    ///
    /// The deeper a leaf node, the greater the prior probability it will
    /// remain a leaf node. The probability a node being a leaf node is
    /// given by `(1 - (p(being a split node))`.
    pub fn sample_expand_flag(&self, depth: usize) -> bool {
        if depth == 0 {
            return true;
        }

        let mut rng = rand::thread_rng();
        let leaf_node_probs = 1. - (self.alpha * ((1. + (depth - 1) as f64).powf(-self.beta)));

        leaf_node_probs < rng.gen::<f64>()
    }

    /// Sample a Gaussian distributed value for a leaf node.
    pub fn sample_leaf_value(
        &self,
        mu: &Array2<f64>, // Shape: [n_groups, n_samples]
        _obs: &[f64],     // Will be used for linear sqaures
        m: usize,
        leaf_sd: &f64,
        n_dims: usize,
        response: &Response,
    ) -> Array1<f64> {
        let mut rng = thread_rng();
        let norms = Array1::from_iter((0..n_dims).map(|_| self.normal.sample(&mut rng) * leaf_sd));

        match mu.nrows() {
            0 => Array1::zeros(n_dims),
            1 => {
                let mut result = mu.column(0).to_owned();
                result.mapv_inplace(|x| x / m as f64);
                result + &norms
            }
            _ => response.compute_leaf_value(mu.view(), m, &norms, n_dims),
        }
    }

    /// Sample the index of a feature to split on.
    ///
    /// Sampling of splitting variables is proportional to `alpha_vec`.
    pub fn sample_split_feature(&self) -> usize {
        let mut rng = rand::thread_rng();

        let p = rng.gen::<f64>();
        for (idx, value) in self.splitting_probs.iter().enumerate() {
            if p <= *value {
                return idx;
            }
        }

        self.splitting_probs.len() - 1
    }
}
