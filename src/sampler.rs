use std::ffi::c_double;

use rand::rngs::SmallRng;

use crate::base::PgBartState;
use crate::resampling::{ResamplingStrategies, SystematicResampling};
use crate::response::{GaussianResponseStrategy, ResponseStrategies};
use crate::split_rules::{ContinuousSplitRule, SplitRules};
use crate::PyBartSettings;

pub type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

/// Particle Gibbs sampler for BART
pub struct PgBartSampler {
    rng: SmallRng,
    model: LogpFunc,
    settings: PyBartSettings,
    split_strategy: SplitRules,
    response_strategy: ResponseStrategies,
    resampling_strategy: ResamplingStrategies,
}

impl PgBartSampler {
    pub fn new(model: LogpFunc, settings: PyBartSettings) -> Self {
        // Initialize default strategies
        let split_strategy = SplitRules::Continuous(ContinuousSplitRule);
        let response_strategy = ResponseStrategies::Gaussian(GaussianResponseStrategy::new(
            settings.init_leaf_std.powi(2),
        ));
        let resampling_strategy = ResamplingStrategies::Systematic(SystematicResampling);

        Self {
            rng: settings.rng.clone(),
            model: model,
            settings: settings,
            split_strategy,
            response_strategy,
            resampling_strategy,
        }
    }

    pub fn step(&mut self, state: &mut PgBartState) {
        println!("Stepping...");
    }
}
