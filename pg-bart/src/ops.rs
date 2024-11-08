use rand_distr::{Distribution, Normal, Uniform};

use rand::{self, thread_rng, Rng};

use crate::pgbart::Response;

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
    /// remain a leaf node.
    pub fn sample_expand_flag(&self, depth: usize) -> bool {
        let mut rng = rand::thread_rng();
        let p = 1. - (self.alpha * (1. + depth as f64).powf(-self.beta));

        p < rng.gen::<f64>()
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
        let norm = self.normal.sample(&mut rng) * leaf_sd;

        match (mu.len(), response) {
            (0, _) => 0.0,
            (1, _) => mu[0] / m as f64 + norm,
            (2, Response::Constant) | (2, Response::Linear) => {
                mu.iter().sum::<f64>() / (2.0 * m as f64) + norm
            }
            (len @ 3.., Response::Constant) => {
                mu.iter().sum::<f64>() / (len as f64 * m as f64) + norm
            }
            (_len @ 3.., Response::Linear) => todo!("Implement fast_linear_fit..."),
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
