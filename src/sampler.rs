// Functions that implement the BART posterior MCMC initialization and update step.
//
// Functions that do MCMC steps operate by taking as input a BART state, and
// outputting a new BART struct with the new state.
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

// TODO: Would be nice if the Particle Gibbs sampler was an interface
//       and we could follow a strategy pattern.
// PgBartState is the main entry point of the algorithm.
pub struct PgBartState {
    // data: Box<dyn ExternalData>,
    params: PgBartSettings,
    // probabilities: Probabilities,
    predictions: Vec<f64>,
    // particles: Vec<Particle>
    variable_inclusion: Vec<u32>,
    tune: bool,
}

// impl PgBartState {
//     pub fn new(param: PgBartSettings, data: Box<dyn ExternalData>) -> Self {}
// }
