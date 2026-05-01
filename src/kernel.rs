use numpy::ndarray::Array1;
use rand::Rng;
use rand::rngs::SmallRng;

use crate::config::BartConfig;
use crate::data::OwnedData;
use crate::resampling::ResamplingStrategy;
use crate::smc::smc_step;
use crate::splitting::SplitRules;
use crate::state::{BartInfo, BartState};
use crate::tree::TreeArrays;
use crate::weight::WeightFn;

pub trait SamplingAlgorithm {
    type State;
    type Info;

    fn init(&self, rng: &mut impl Rng) -> Self::State;
    fn step(&self, rng: &mut impl Rng, state: Self::State) -> (Self::State, Self::Info);
}

pub struct BartKernel<R, W> {
    pub split_rules: Vec<SplitRules>,
    pub resampling: R,
    pub weight_fn: W,
    pub config: BartConfig,
    pub data: OwnedData,
}

impl<R, W> SamplingAlgorithm for BartKernel<R, W>
where
    R: ResamplingStrategy,
    W: WeightFn,
{
    type State = BartState;
    type Info = BartInfo;

    fn init(&self, _rng: &mut impl Rng) -> BartState {
        let n_samples = self.data.n_samples();
        let y_mean = self.data.y.mean().unwrap_or(0.0);
        let init_leaf_value = y_mean / self.config.n_trees as f64;

        let forest: Vec<TreeArrays> = (0..self.config.n_trees)
            .map(|_| TreeArrays::new(init_leaf_value, n_samples, self.config.max_depth))
            .collect();

        // Sum of all initial trees = n_trees * init_leaf_value = y_mean.
        let predictions = Array1::from_elem(n_samples, y_mean);
        let variable_inclusion = vec![0u32; self.data.n_features()];

        BartState {
            forest,
            predictions,
            variable_inclusion,
            next_tree_idx: 0,
            tune: true,
        }
    }

    fn step(&self, rng: &mut impl Rng, mut state: BartState) -> (BartState, BartInfo) {
        let data_view = self.data.view();
        let n_samples = self.data.n_samples();
        let n_trees = self.config.n_trees;

        let batch_frac = if state.tune {
            self.config.batch_tune
        } else {
            self.config.batch_post
        };
        let batch_size = ((batch_frac * n_trees as f64).round() as usize)
            .max(1)
            .min(n_trees);

        let mut acceptance_count = 0;
        let mut tree_depths = Vec::with_capacity(batch_size);
        let mut total_log_likelihood = 0.0;

        let mut residuals_buf = Array1::zeros(n_samples);
        let mut tree_pred_buf = Array1::zeros(n_samples);

        for k in 0..batch_size {
            let tree_idx = (state.next_tree_idx + k) % n_trees;

            // Residuals = sum of all OTHER trees = predictions - old_tree.predict()
            state.forest[tree_idx].predict_training_into(&mut tree_pred_buf);
            residuals_buf.assign(&state.predictions);
            residuals_buf -= &tree_pred_buf;

            let (new_tree, step_info) = smc_step(
                rng,
                &residuals_buf,
                &self.config,
                &data_view,
                &self.split_rules,
                &self.resampling,
                &self.weight_fn,
            );

            // predictions = residuals + new_tree.predict()
            new_tree.predict_training_into(&mut tree_pred_buf);
            state.predictions.assign(&residuals_buf);
            state.predictions += &tree_pred_buf;

            total_log_likelihood += step_info.log_likelihood;
            acceptance_count += step_info.acceptance_count;

            let depth = (0..new_tree.size)
                .filter(|&i| new_tree.is_leaf(i))
                .map(|i| new_tree.get_depth(i) as u8)
                .max()
                .unwrap_or(0);
            tree_depths.push(depth);

            state.forest[tree_idx] = new_tree;
        }

        state.next_tree_idx = (state.next_tree_idx + batch_size) % n_trees;

        let info = BartInfo {
            log_likelihood: total_log_likelihood,
            acceptance_count,
            tree_depths,
        };

        (state, info)
    }
}

/// Type-erased kernel for Python bindings.
pub trait ErasedKernel {
    fn init(&self, rng: &mut SmallRng) -> BartState;
    fn step(&self, rng: &mut SmallRng, state: BartState) -> (BartState, BartInfo);
}

impl<R, W> ErasedKernel for BartKernel<R, W>
where
    R: ResamplingStrategy,
    W: WeightFn,
{
    fn init(&self, rng: &mut SmallRng) -> BartState {
        SamplingAlgorithm::init(self, rng)
    }

    fn step(&self, rng: &mut SmallRng, state: BartState) -> (BartState, BartInfo) {
        SamplingAlgorithm::step(self, rng, state)
    }
}
