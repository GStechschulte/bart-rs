use numpy::ndarray::Array1;

/// Configuration for the BART sampler.
#[derive(Clone, Debug)]
pub struct BartConfig {
    pub n_trees: usize,
    pub n_particles: usize,
    pub max_depth: u8,
    pub alpha: f64,
    pub beta: f64,
    pub sigma: f64,
    pub min_samples_leaf: usize,
    pub splitting_probs: Option<Array1<f64>>,
    /// Fraction of trees updated per `step()` while tuning.
    pub batch_tune: f64,
    /// Fraction of trees updated per `step()` after tuning.
    pub batch_post: f64,
}

impl Default for BartConfig {
    fn default() -> Self {
        Self {
            n_trees: 50,
            n_particles: 10,
            max_depth: 6,
            alpha: 0.95,
            beta: 2.0,
            sigma: 1.0,
            min_samples_leaf: 5,
            splitting_probs: None,
            batch_tune: 0.1,
            batch_post: 0.1,
        }
    }
}
