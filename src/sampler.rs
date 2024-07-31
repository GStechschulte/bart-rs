use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal, Uniform};

// Functions that implement the BART Particle Gibbs initialization and update step.
//
// Functions that do Particle Gibbs steps operate by taking as input a PgBartState
// struct, and output a new BART struct with the new state.
//

// pub trait ExternalData {
//     fn X(&self) -> &Matrix<f64>;
//     fn y(&self) -> &Vec<f64>;
//     fn model_logp(&self, v: &Vec<f64>) -> f64;
// }

// Parameter settings are used to initialize a new PgBartState
pub struct PgBartSettings {
    n_trees: usize,
    n_particles: usize,
    alpha: f64,
    default_kf: f64,
    batch: (f64, f64),
    init_alpha_vec: Vec<f64>,
}

impl PgBartSettings {
    pub fn new(
        n_trees: usize,
        n_particles: usize,
        alpha: f64,
        default_kf: f64,
        batch: (f64, f64),
        init_alpha_vec: Vec<f64>,
    ) -> Self {
        Self {
            n_trees,
            n_particles,
            alpha,
            default_kf,
            batch,
            init_alpha_vec,
        }
    }
}

pub struct Probabilities {
    normal: Normal<f64>,
    uniform: Uniform<f64>,
    alpha_vec: Vec<f64>,
    splitting_probs: Vec<f64>,
    alpha: f64,
}

impl Probabilities {
    pub fn sample_expand_flag(&self, depth: usize) -> bool {
        let mut rng = rand::thread_rng();
        let p = 1. - self.alpha.powi(depth as i32);
        let res = p < rng.gen::<f64>();

        res
    }

    pub fn sample_leaf_values(&self, mu: f64, kfactor: f64) -> f64 {
        let mut rng = thread_rng();
        let norm = self.normal.sample(&mut rng) * kfactor;

        norm + mu
    }

    // Sample the feature index to split on
    pub fn sample_feature(&self) -> usize {
        let mut rng = rand::thread_rng();

        let p = rng.gen::<f64>();
        for (index, value) in self.splitting_probs.iter().enumerate() {
            if p <= *value {
                return index;
            }
        }

        self.splitting_probs.len() - 1
    }

    // Sample a split value over the observed values
    pub fn sample_threshold(&self, candidates: &Vec<f64>) -> Option<f64> {
        let mut rng = rand::thread_rng();

        if candidates.len() == 0 {
            None
        } else {
            let dist = Uniform::<usize>::new(0, candidates.len());
            let index = dist.sample(&mut rng);
            Some(candidates[index])
        }
    }
}

// TODO: Would be nice if the Particle Gibbs sampler was an interface
//       and we could follow a strategy pattern.
// PgBartState is the main entry point of the algorithm.
pub struct PgBartState {
    // data: Box<dyn ExternalData>,
    pub params: PgBartSettings,
    pub probabilities: Probabilities,
    pub predictions: Vec<f64>,
    // particles: Vec<Particle>
    pub variable_inclusion: Vec<u32>,
    pub tune: bool,
}

// impl PgBartState {
//     pub fn new(param: PgBartSettings, data: Box<dyn ExternalData>) -> Self {}
// }
