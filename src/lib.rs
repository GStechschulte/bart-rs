use std::ffi::c_double;

pub mod config;
pub mod data;
pub mod forest;
pub mod kernel;
pub mod particle;
pub mod resampling;
pub mod response;
pub mod smc;
pub mod splitting;
pub mod state;
pub mod tree;
pub mod update;
pub mod weight;

use crate::config::BartConfig;
use crate::data::OwnedData;
use crate::kernel::{BartKernel, ErasedKernel, SamplingAlgorithm};
use crate::resampling::SystematicResampling;
use crate::splitting::{ContinuousSplit, SplitRules};
use crate::weight::PyMCWeightFn;

use numpy::{
    PyArray1, PyReadonlyArray,
    ndarray::{Array1, Ix1, Ix2},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

#[pyclass]
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PyBartSettings {
    n_trees: usize,
    n_particles: usize,
    max_depth: u8,
    alpha: f64,
    beta: f64,
    sigma: f64,
    split_prior: Vec<f64>,
    split_rules: Vec<String>,
    response_rule: String,
    resampling_rule: String,
    batch_tune: f64,
    batch_post: f64,
    seed: u64,
}

#[pymethods]
impl PyBartSettings {
    #[new]
    #[pyo3(signature = (
        n_trees,
        n_particles,
        max_depth,
        alpha,
        beta,
        sigma,
        split_prior,
        split_rules,
        response_rule,
        resampling_rule,
        batch_tune = 0.1,
        batch_post = 0.1,
        seed = 0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_trees: usize,
        n_particles: usize,
        max_depth: u8,
        alpha: f64,
        beta: f64,
        sigma: f64,
        split_prior: Vec<f64>,
        split_rules: Vec<String>,
        response_rule: String,
        resampling_rule: String,
        batch_tune: f64,
        batch_post: f64,
        seed: u64,
    ) -> Self {
        Self {
            n_trees,
            n_particles,
            max_depth,
            alpha,
            beta,
            sigma,
            split_prior,
            split_rules,
            response_rule,
            resampling_rule,
            batch_tune,
            batch_post,
            seed,
        }
    }
}

#[pyclass(unsendable)]
struct PySampler {
    kernel: Box<dyn ErasedKernel>,
    state: Option<crate::state::BartState>,
    rng: SmallRng,
}

#[pymethods]
impl PySampler {
    #[staticmethod]
    fn init(
        x: PyReadonlyArray<f64, Ix2>,
        y: PyReadonlyArray<f64, Ix1>,
        model: usize,
        settings: PyBartSettings,
    ) -> PyResult<PySampler> {
        let x_data = x.as_array().to_owned();
        let y_data = y.as_array().to_owned();

        let logp_func: LogpFunc = unsafe { std::mem::transmute(model as *const ()) };
        let weight_fn = unsafe { PyMCWeightFn::from_raw(logp_func) };

        // Parse split rules
        let split_rules: Vec<SplitRules> = settings
            .split_rules
            .iter()
            .map(|rule| SplitRules::from_name(rule).map_err(PyValueError::new_err))
            .collect::<PyResult<Vec<SplitRules>>>()?;

        // Fill with continuous splits if not enough rules provided
        let n_features = x_data.ncols();
        let split_rules = if split_rules.len() < n_features {
            let mut rules = split_rules;
            rules.resize(n_features, SplitRules::Continuous(ContinuousSplit));
            rules
        } else {
            split_rules
        };

        let config = BartConfig {
            n_trees: settings.n_trees,
            n_particles: settings.n_particles,
            max_depth: settings.max_depth,
            alpha: settings.alpha,
            beta: settings.beta,
            sigma: settings.sigma,
            min_samples_leaf: 2,
            splitting_probs: if settings.split_prior.is_empty() {
                None
            } else {
                Some(Array1::from_vec(settings.split_prior))
            },
            batch_tune: settings.batch_tune,
            batch_post: settings.batch_post,
        };

        let data = OwnedData::new(x_data, y_data);

        let kernel = BartKernel {
            split_rules,
            resampling: SystematicResampling,
            weight_fn,
            config,
            data,
        };

        let mut rng = SmallRng::seed_from_u64(settings.seed);
        let state = SamplingAlgorithm::init(&kernel, &mut rng);

        Ok(PySampler {
            kernel: Box::new(kernel),
            state: Some(state),
            rng,
        })
    }

    #[pyo3(signature = (tune = None))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        tune: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut state = self
            .state
            .take()
            .ok_or_else(|| PyValueError::new_err("Sampler state is missing (internal error)"))?;

        if let Some(t) = tune {
            state.tune = t;
        }

        let (new_state, _info) = self.kernel.step(&mut self.rng, state);

        let result = PyArray1::from_slice(py, new_state.predictions.as_slice().unwrap());
        self.state = Some(new_state);
        Ok(result)
    }
}

#[pymodule]
fn pymc_bart(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBartSettings>()?;
    m.add_class::<PySampler>()?;
    Ok(())
}
