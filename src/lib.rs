use std::collections::HashMap;
use std::ffi::c_double;

pub mod base;
pub mod builder;
pub mod particle;
pub mod resampling;
pub mod response;
pub mod sampler;
pub mod split_rules;
pub mod update;

use crate::builder::{BartSampler, BartSamplerBuilder};
use crate::split_rules::SplitRules;
use crate::update::BARTContext;

use numpy::{
    ndarray::{Ix1, Ix2},
    PyArray1, PyReadonlyArray,
};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;

pub type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

#[pyclass]
#[derive(Clone)] // Clone is useful for passing settings around
pub struct PyBartSettings {
    init_leaf_value: f64,
    init_leaf_std: f64,
    n_trees: usize,
    n_particles: usize,
    max_depth: usize,
    alpha: f64,
    beta: f64,
    split_prior: Vec<f64>,
    /// Key-value pair indicating the split rule for each dimension of the design matrix.
    split_rules: HashMap<usize, SplitRules>,
    response_rule: String,
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
        max_depth: usize,
        alpha: f64,
        beta: f64,
        split_prior: Vec<f64>,
        split_rules_py: HashMap<usize, String>,
        response_rule: String,
        resampling_rule: String,
        batch_size: (f64, f64),
    ) -> PyResult<Self> {
        let mut split_rules = HashMap::new();
        for (dim, rule_name) in split_rules_py {
            let rule = SplitRules::from_str(&rule_name)?;
            split_rules.insert(dim, rule);
        }

        Ok(Self {
            init_leaf_value,
            init_leaf_std,
            n_trees,
            n_particles,
            max_depth,
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
    context: BARTContext,
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
        let data = x.as_array().to_owned();
        let targets = y.as_array().to_owned();

        let sampler = BartSamplerBuilder::new()
            // .max_nodes = settings.max_nodes;
            // .n_particles = settings.n_particles;
            // .init_leaf_value = settings.init_leaf_value;
            // ... set other fields
            .build(&targets)?;

        let context = BARTContext {
            x_data: data.outer_iter().map(|row| row.to_vec()).collect(),
            y_residuals: targets.to_vec(),
            alpha: settings.alpha,
            beta: settings.beta,
            sigma: 1.0,
            min_samples_leaf: 1,
            max_depth: settings.max_depth,
        };

        Ok(PySampler { sampler, context })
    }

    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let seed = 42;
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights = self.sampler.step(&mut rng, &self.context);
        Ok(PyArray1::from_vec(py, weights))
    }

    fn run<'py>(
        &mut self,
        py: Python<'py>,
        n_iterations: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let seed = 42;
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights = self.sampler.run(&mut rng, &self.context, n_iterations);
        Ok(PyArray1::from_vec(py, weights))
    }
}

#[pymodule]
fn pymc_bart(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBartSettings>()?;
    m.add_class::<PySampler>()?;
    Ok(())
}
