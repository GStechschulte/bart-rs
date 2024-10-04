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

        let res = p < rng.gen::<f64>();

        res
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
            (1, _) => {
                let mu_mean = mu[0] / m as f64 + norm;
                mu_mean
            }
            (2, Response::Constant) | (2, Response::Linear) => {
                let mu_mean = mu.iter().sum::<f64>() / (2.0 * m as f64) + norm;
                mu_mean
            }
            (len @ 3.., Response::Constant) => {
                let mu_mean = mu.iter().sum::<f64>() / (len as f64 * m as f64) + norm;
                mu_mean
            }
            (len @ 3.., Response::Linear) => todo!("Implement fast_linear_fit..."),
        }
    }

    // pub fn fast_linear_fit(&self, mu: &Vec<f64, obs: &Vec<f64>) {
    //     todo!("Implement!!!")
    // }

    /// Sample the index of a feature to split on.
    ///
    /// Sampling of splitting variables is proportional to `alpha_vec`.
    pub fn sample_split_index(&self) -> usize {
        let mut rng = rand::thread_rng();

        let p = rng.gen::<f64>();
        for (idx, value) in self.splitting_probs.iter().enumerate() {
            if p <= *value {
                return idx;
            }
        }

        self.splitting_probs.len() - 1
    }

    /// Sample a split value from a vector of candidate points.
    ///
    /// Candidate points are sampled by first creating a Uniform distribution
    /// over the indices of the `candidates` vector. Then, a random index is
    /// sampled from this distribution.
    // pub fn sample_split_value(&self, candidates: &Vec<f64>) -> Option<f64> {
    pub fn sample_split_value(&self, candidates: &[f64]) -> Option<f64> {
        if candidates.is_empty() {
            None
        } else {
            let mut rng = rand::thread_rng();
            // let dist = Uniform::<usize>::new(0, candidates.len());
            let dist = Uniform::from(0..candidates.len());
            let idx = dist.sample(&mut rng);

            Some(candidates[idx])
        }
    }
}
