use rand_distr::{Distribution, Normal, Uniform};

use rand::{self, Rng};

pub struct TreeProbabilities {
    pub normal: Normal<f64>,
    pub uniform: Uniform<f64>,
    pub alpha_vec: Vec<f64>,
    pub splitting_probs: Vec<f64>,
    pub alpha: f64,
    pub beta: f64,
}

impl TreeProbabilities {
    /// Sample a boolean flag indicating if a node should be split or not.
    ///
    /// The deeper a leaf node, the larger the prior probability it will
    /// remain a leaf node.
    pub fn sample_expand_flag(&self, depth: usize) -> bool {
        let mut rng = rand::thread_rng();

        let p = 1. - self.alpha * (1. + depth as f64).powf(-self.beta);
        let res = p < rng.gen::<f64>();

        res
    }

    // Sample a new value for a leaf node
    pub fn sample_leaf_value(&self, mu: f64, kfactor: f64) -> f64 {
        let mut rng = rand::thread_rng();

        let norm = self.normal.sample(&mut rng) * kfactor;

        norm + mu
    }

    // Sample the index of a feature to split on
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

    // Sample a boolean flag indicating if a node should be split or not
    pub fn sample_split_value(&self, candidates: &Vec<f64>) -> Option<f64> {
        let mut rng = rand::thread_rng();

        if candidates.len() == 0 {
            None
        } else {
            let dist = Uniform::<usize>::new(0, candidates.len());
            let idx = dist.sample(&mut rng);
            Some(candidates[idx])
        }
    }
}
