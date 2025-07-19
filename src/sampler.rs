use std::ffi::c_double;
use std::time::Instant;

use bumpalo::{collections::Vec as BumpVec, Bump};
use numpy::ndarray::{Array, Ix1, Ix2};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::base::PgBartState;
use crate::forest::Forest;
use crate::resampling::{ResamplingStrategies, ResamplingStrategy, SystematicResampling};
use crate::response::{GaussianResponseStrategy, ResponseStrategies};
use crate::split_rules::{ContinuousSplitRule, SplitRules};
use crate::PyBartSettings;

pub type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

pub trait WeightFn {
    fn weight(&self, x: Array<f64, Ix1>) -> f64;
}

/// Calls the LogpFunc callback to compute the log-likelihood
fn weight(logp: LogpFunc, x: Array<f64, Ix1>) -> f64 {
    unsafe { (logp)(x.as_ptr(), x.len()) }
}

/// Weight normalization via the Log-sum-exp trick.
pub fn normalize_weights(weights: &[f64]) -> Vec<f64> {
    // TODO: Do we have to clone here?
    let max_log_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = weights.iter().map(|&w| (w - max_log_weight).exp()).sum();

    weights
        .iter()
        .map(|&w| ((w - max_log_weight).exp() / sum_exp))
        .collect()
}

/// Particle Gibbs sampler for BART
pub struct PgBartSampler {
    /// User-provided design matrix
    data: Array<f64, Ix2>,
    /// User-provided response (target) vector
    targets: Array<f64, Ix1>,
    model: LogpFunc,
    settings: PyBartSettings,
    split_strategy: SplitRules,
    response_strategy: ResponseStrategies,
    resampling_strategy: ResamplingStrategies,
}

impl PgBartSampler {
    pub fn new(
        data: Array<f64, Ix2>,
        targets: Array<f64, Ix1>,
        model: LogpFunc,
        settings: PyBartSettings,
    ) -> Self {
        // Initialize default strategies
        let split_strategy = SplitRules::Continuous(ContinuousSplitRule);
        let response_strategy = ResponseStrategies::Gaussian(GaussianResponseStrategy::new(
            settings.init_leaf_std.powi(2),
        ));
        let resampling_strategy = ResamplingStrategies::Systematic(SystematicResampling);

        Self {
            data: data,
            targets: targets,
            model: model,
            settings: settings,
            split_strategy,
            response_strategy,
            resampling_strategy,
        }
    }

    pub fn step(&mut self, state: &mut PgBartState) -> Vec<f64> {
        println!("Stepping...");
        let start_time = Instant::now();

        // Setup phase

        // Arena lives only as long as this function call
        let arena = Bump::new();

        // let example_preds =
        //     Array::<f64, Ix1>::from_elem(self.y.len(), self.settings.init_leaf_value);
        // let example_weights = vec![0.25; self.settings.n_trees];
        // let example_forest = vec![0.5; self.settings.n_trees];

        // let example_state = PgBartState {
        //     forest: example_forest,
        //     weights: example_weights,
        //     predictions: example_preds,
        // };

        // Dispatch to choose the pre-compiled optimized worker function
        let predictions = match self.settings.max_depth {
            1..=4 => self.tree_step::<5>(state, &arena),
            6..=9 => self.tree_step::<10>(state, &arena),
            11..=14 => self.tree_step::<15>(state, &arena),
            _ => panic!("MAX_DEPTH is too deep"),
        };

        let duration = start_time.elapsed();
        println!("time: {:?}", duration);

        predictions
    }

    /// Worker function to monomorphize the creation of the Forest and Trees.
    fn tree_step<'a, const MAX_DEPTH: usize>(
        &self,
        state: &mut PgBartState,
        arena: &'a Bump,
    ) -> Vec<f64> {
        let mut setup_rng = SmallRng::seed_from_u64(42);
        let mut forest = Forest::<MAX_DEPTH>::new(arena, self.settings.n_particles);

        for _ in 0..self.settings.n_particles {
            forest.plant_random_tree(
                self.settings.init_leaf_value,
                self.targets.len(),
                10,
                self.data.ncols(),
                &mut setup_rng,
            );
            let predictions =
                Array::<f64, Ix1>::from_elem(self.targets.len(), self.settings.init_leaf_value);
            let ll = weight(self.model, predictions);
            forest.add_weight(ll);
        }

        let weights = forest.weights();
        let norm_weights = normalize_weights(weights);
        let indices = SystematicResampling::resample(&mut setup_rng, &norm_weights);
        forest.resample_trees(&indices).unwrap();

        // let new_predictions =
        //     Array::<f64, Ix1>::from_elem(self.y.len(), self.settings.init_leaf_value);
        let new_predictions_vec = vec![self.settings.init_leaf_value; self.targets.len()];

        new_predictions_vec
    }
}

// TODO: Should we remove the tree_step method above and call this? Would need to add constant
// generics and arena lifetimes to monomorphize this step function into a highly-optimized step.
//
// Forest sampling step of BART.
//
// Steps through all of the trees, and for each tree, grows N particle trees, where one is proposed
// to replace the current tree.
fn step_trees<Rng, U, W, R>(
    rng: &mut Rng,
    state: &mut PgBartState,
    update_fn: &U,
    weight_fn: &W,
    resample_fn: &R,
) -> PgBartState
where
    Rng: rand::Rng,
    U: FnMut(&mut Rng) -> PgBartState,
    W: WeightFn,
    R: ResamplingStrategy,
{
    // Initialize a Forest (set of Particle trees) for this step

    // Weight these Particle trees

    // Update step (apply kernel aka grow each Tree sequentially)

    // Weight update step
    // let tree_predictions = Array::from_elem(shape, elem);
    // let weights = weight_fn.weight(x)

    // Resampling step

    // let new_state = PgBartState {
    //     forest: vec![1, 2, 3],
    //     weights: vec![0.5, 0.3, 0.2],
    //     predictions: Array::<f64, Ix1>::from_iter(0.5..1.0),
    // };

    // new_state
    todo!("Not implemented")
}
