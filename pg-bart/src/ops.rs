use std::str::FromStr;

use rand::{self, thread_rng, Rng};
use rand_distr::{Distribution, Normal, Uniform};

#[derive(Debug)]
pub enum Response {
    Constant(ConstantResponse),
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
    fn compute_leaf_value(&self, mu: &[f64], m: usize, norm: f64) -> f64;
}

impl ResponseStrategy for Response {
    /// Calls the corresponding `compute_leaf_value` for each `Response` variant
    fn compute_leaf_value(&self, mu: &[f64], m: usize, norm: f64) -> f64 {
        match self {
            Response::Constant(constant) => constant.compute_leaf_value(mu, m, norm),
            Response::Linear(linear) => linear.compute_leaf_value(mu, m, norm),
        }
    }
}

// Use unit structs to define an enum variant constructor for which the
// ResponseStrategy trait can be implemented for as traits cannot be
// implemented for enum variants as a variant is not a type
#[derive(Debug)]
pub struct ConstantResponse;
impl ResponseStrategy for ConstantResponse {
    fn compute_leaf_value(&self, mu: &[f64], m: usize, norm: f64) -> f64 {
        (mu.iter().sum::<f64>() / mu.len() as f64) / m as f64 + norm
    }
}

#[derive(Debug)]
pub struct LinearResponse;
impl ResponseStrategy for LinearResponse {
    fn compute_leaf_value(&self, mu: &[f64], m: usize, norm: f64) -> f64 {
        match mu.len() {
            2 => mu.iter().sum::<f64>() / (2.0 * m as f64) + norm,
            _len @ 3.. => todo!("Implement fast_linear_fit."),
            _ => unreachable!("Linear response requires at least 2 values."),
        }
    }
}

pub struct TreeSamplingOps {
    pub normal: Normal<f64>,
    pub uniform: Uniform<f64>,
    pub alpha_vec: Vec<f64>,
    pub splitting_probs: Vec<f64>,
    pub alpha: f64,
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
            // println!("depth: {}, probs: {}", depth, 0.0);
            return true;
        }

        let mut rng = rand::thread_rng();
        let leaf_node_probs = 1. - (self.alpha * ((1. + (depth - 1) as f64).powf(-self.beta)));
        // println!("depth: {}, probs: {}", depth, leaf_node_probs);

        leaf_node_probs < rng.gen::<f64>()
    }

    /// Sample a Gaussian distributed value for a leaf node.
    pub fn sample_leaf_value(
        &self,
        mu: &[f64],
        _obs: &[f64],
        m: usize,
        leaf_sd: &f64,
        _shape: usize,
        response: &Response,
    ) -> f64 {
        let mut rng = thread_rng();
        let norm = self.normal.sample(&mut rng);

        // println!("--- sample_leaf_value ---");
        // println!("m: {}", m);
        // println!("mu.len(): {}", mu.len());
        // println!("norm: {}, leaf_sd: {}", norm, leaf_sd);

        let norm = norm * leaf_sd;

        match mu.len() {
            0 => 0.0,
            1 => mu[0] / m as f64 + norm,
            _ => response.compute_leaf_value(mu, m, norm),
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
