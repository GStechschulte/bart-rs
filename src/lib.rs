use std::ffi::c_double;

pub mod base;
pub mod builder;
pub mod particle;
pub mod resampling;
pub mod response;
pub mod sampler;
pub mod splitting;
pub mod update;

use crate::builder::{BartSampler, BartSamplerBuilder};
use crate::response::ResponseStrategies;
use crate::splitting::SplitRules;
use crate::update::TreeContext;

use numpy::{
    PyArray1, PyReadonlyArray,
    ndarray::{Ix1, Ix2},
};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

pub type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyBartSettings {
    init_leaf_value: f64,
    init_leaf_std: f64,
    n_trees: usize,
    n_particles: usize,
    max_nodes: usize,
    alpha: f64,
    beta: f64,
    split_prior: Vec<f64>,
    /// Key-value pair indicating the split rule for each dimension of the design matrix.
    split_rules: Vec<SplitRules>,
    response_rule: ResponseStrategies,
    resampling_rule: String,
    batch_size: (f64, f64),
}

#[pymethods]
impl PyBartSettings {
    #[new]
    fn new(
        init_leaf_value: f64,
        init_leaf_std: f64,
        n_trees: usize,
        n_particles: usize,
        max_nodes: usize,
        alpha: f64,
        beta: f64,
        split_prior: Vec<f64>,
        split_rules: Vec<String>,
        response_rule: String,
        resampling_rule: String,
        batch_size: (f64, f64),
    ) -> PyResult<Self> {
        let split_rules: Vec<SplitRules> = split_rules
            .iter()
            .map(|rule| SplitRules::from_str(rule))
            .collect::<PyResult<Vec<SplitRules>>>()?;

        let response_rule = ResponseStrategies::from_str(response_rule.as_str())?;

        Ok(Self {
            init_leaf_value,
            init_leaf_std,
            n_trees,
            n_particles,
            max_nodes,
            alpha,
            beta,
            split_prior,
            split_rules,
            response_rule,
            resampling_rule,
            batch_size,
        })
    }
}

#[pyclass(unsendable)]
struct PySampler {
    sampler: BartSampler,
    context: TreeContext,
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

        let sampler = BartSamplerBuilder::new()
            .with_max_nodes(settings.max_nodes)
            .with_n_particles(settings.n_particles)
            .with_init_leaf_value(settings.init_leaf_value)
            .with_split_strategies(settings.split_rules)
            .with_response_strategy(settings.response_rule)
            .with_bart_params(settings.alpha, settings.beta, 1.0)
            .build(&x_data, &y_data, logp_func)?;

        // TODO: mutable state should not really be in the TreeContext???
        let context = TreeContext {
            x_data: x_data,
            y_data: y_data,
            alpha: settings.alpha,
            beta: settings.beta,
            sigma: settings.init_leaf_std,
            n_trees: settings.n_trees,
            splitting_probs: Some(settings.split_prior.into()),
            min_samples_leaf: 2,
            max_nodes: settings.max_nodes,
        };

        Ok(PySampler { sampler, context })
    }

    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let seed = 42;
        let mut rng = SmallRng::seed_from_u64(seed);
        // TODO: Add a predictions buffer to avoid repeated allocations
        let py_ensemble_predictions = self.sampler.step(&mut rng, &self.context);
        Ok(PyArray1::from_slice(
            py,
            py_ensemble_predictions.as_slice().unwrap(),
        ))
        // let vec_ensemble_predictions = py_ensemble_predictions.to_vec();
        // Ok(PyArray1::from_vec(py, vec_ensemble_predictions))
    }
}

#[pymodule]
fn pymc_bart(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBartSettings>()?;
    m.add_class::<PySampler>()?;
    Ok(())
}
