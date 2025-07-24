use std::ffi::c_double;
use std::time::Instant;

use bumpalo::Bump;
use numpy::ndarray::{Array, Ix1, Ix2};
use rand::Rng;

use crate::base::PgBartState;
use crate::forest::{Forest, Update};
use crate::resampling::{ResamplingStrategy, SystematicResampling};
use crate::PyBartSettings;

pub type LogpFn = unsafe extern "C" fn(*const f64, usize) -> c_double;

/// Calls the LogpFunc callback to compute the log-likelihood
fn weight(logp: LogpFn, x: Array<f64, Ix1>) -> f64 {
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
pub struct PgBartSampler<const MAX_DEPTH: usize, U, R> {
    /// Design matrix.
    data: Array<f64, Ix2>,
    /// Target (response) vector.
    targets: Array<f64, Ix1>,
    /// Update.
    update: U,
    /// Resample.
    resample: R,
    /// Weight.
    weight_fn: LogpFn,
    /// Defines.
    settings: PyBartSettings,
}

impl<const MAX_DEPTH: usize, U: Update<MAX_DEPTH>, R: ResamplingStrategy>
    PgBartSampler<MAX_DEPTH, U, R>
{
    pub fn new(
        data: Array<f64, Ix2>,
        targets: Array<f64, Ix1>,
        update_fn: U,
        resample_fn: R,
        weight_fn: LogpFn,
        settings: PyBartSettings,
    ) -> Self {
        Self {
            data: data,
            targets: targets,
            update: update_fn,
            resample: resample_fn,
            weight_fn: weight_fn,
            settings: settings,
        }
    }

    pub fn step(&mut self, rng: &mut impl Rng, state: &mut PgBartState) -> Vec<f64> {
        println!("Stepping...");
        let start_time = Instant::now();

        // Arena lives only as long as this function call
        let arena = Bump::new();

        for tree in 0..self.settings.n_trees {
            let mut forest = Forest::<MAX_DEPTH>::new(&arena, self.settings.n_particles);

            // Initialize Particle trees
            for _ in 0..self.settings.n_particles {
                forest.plant_random_tree(
                    self.settings.init_leaf_value,
                    self.targets.len(),
                    10,
                    self.data.ncols(),
                    rng,
                );
                let predictions =
                    Array::<f64, Ix1>::from_elem(self.targets.len(), self.settings.init_leaf_value);
                let ll = weight(self.weight_fn, predictions);
                forest.add_weight(ll);
            }

            // Update (grow) each particle tree
            forest
                .trees
                .iter_mut()
                .for_each(|tree| self.update_fn.update(rng, tree));

            // Normalize weights
            let weights = forest.weights();
            let norm_weights = normalize_weights(weights);

            // Resample particle trees
            let indices = SystematicResampling::resample(rng, &norm_weights);
            forest.resample_trees(&indices).unwrap();

            // Make new predictions (sum of trees)
            // let new_predictions_vec = vec![self.settings.init_leaf_value; self.targets.len()];
        }

        let new_predictions_vec = vec![self.settings.init_leaf_value; self.targets.len()];

        let duration = start_time.elapsed();
        println!("time: {:?}", duration);

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
// fn step_trees<Rng, U, W, R>(
//     rng: &mut Rng,
//     state: &mut PgBartState,
//     update_fn: &U,
//     weight_fn: &W,
//     resample_fn: &R,
// ) -> PgBartState
// where
//     Rng: rand::Rng,
//     U: FnMut(&mut Rng) -> PgBartState,
//     W: WeightFn,
//     R: ResamplingStrategy,
// {
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
//     todo!("Not implemented")
// }
