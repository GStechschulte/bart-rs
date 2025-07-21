use std::{collections::HashMap, ffi::c_void};

pub mod base;
pub mod forest;
pub mod resampling;
pub mod response;
pub mod sampler;
pub mod split_rules;

use crate::base::PgBartState;
use crate::forest::Grow;
use crate::resampling::SystematicResampling;
use crate::response::GaussianResponseStrategy;
use crate::sampler::{LogpFn, PgBartSampler};
use crate::split_rules::SplitRules;

use numpy::{
    ndarray::{Array1, Ix1, Ix2},
    PyArray1, PyReadonlyArray,
};
use pyo3::prelude::*;
use rand::{rngs::SmallRng, SeedableRng};

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
            batch_size,
        })
    }
}

#[pyclass]
struct PySampler {
    sampler: Sampler,
    state: PgBartState,
}

#[pymethods]
impl PySampler {
    #[staticmethod]
    #[pyo3(text_signature = "(x, y, settings)")]
    fn init(
        x: PyReadonlyArray<f64, Ix2>,
        y: PyReadonlyArray<f64, Ix1>,
        model: usize,
        settings: PyBartSettings,
    ) -> PyResult<PySampler> {
        // to_owned_array performs a deep copy of the underlyng data
        let logp: LogpFn = unsafe { std::mem::transmute(model as *const c_void) };

        let data = x.as_array().to_owned();
        let targets = y.as_array().to_owned();

        let forest = vec![0.0; settings.n_trees];
        let weights = vec![0.0; settings.n_trees];
        let predictions = Array1::from_elem(targets.len(), targets.mean().unwrap());

        let resample_step = SystematicResampling;
        let state = PgBartState::new(forest, weights, predictions);

        // Create the sampler based on max_depth and response strategy
        let sampler = match (settings.max_depth, settings.response_rule.as_str()) {
            (1..=4, "gaussian") => {
                let response_strategy = GaussianResponseStrategy;
                let update_step = Grow {
                    split_strategy: settings.split_rules.clone(),
                    response_strategy,
                };
                Sampler::Depth5Gaussian(PgBartSampler::new(
                    data,
                    targets,
                    update_step,
                    resample_step,
                    logp,
                    settings,
                ))
            }
            (6..=9, "gaussian") => {
                let response_strategy = GaussianResponseStrategy;
                let update_step = Grow {
                    split_strategy: settings.split_rules.clone(),
                    response_strategy,
                };
                Sampler::Depth10Gaussian(PgBartSampler::new(
                    data,
                    targets,
                    update_step,
                    resample_step,
                    logp,
                    settings,
                ))
            }
            (11..=14, "gaussian") => {
                let response_strategy = GaussianResponseStrategy;
                let update_step = Grow {
                    split_strategy: settings.split_rules.clone(),
                    response_strategy,
                };
                Sampler::Depth15Gaussian(PgBartSampler::new(
                    data,
                    targets,
                    update_step,
                    resample_step,
                    logp,
                    settings,
                ))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported combination: max_depth={}, response_rule='{}'",
                    settings.max_depth, settings.response_rule
                )))
            }
        };

        Ok(PySampler { sampler, state })
    }

    /// Runs the Particle Gibbs sampler sequentially for `M` iterations where `M` is the number
    /// of trees.
    ///
    /// A single step will initialize a set of particles `N`, of which one will replace the `m'th` tree. To decide
    /// which particle will replace the current tree, the `N` particles are grown until the probability of a
    /// leaf node expanding is less than a random value in the interval [0, 1].
    ///
    /// The grown particles are resampled according to their log-likelihood, of which
    /// one is selected to replace the `m'th` tree.
    /// Runs the Particle Gibbs sampler sequentially.
    fn step<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // TODO
        let seed = 42;
        let mut rng = SmallRng::seed_from_u64(seed);
        let sum_trees = self.sampler.step(&mut rng, &mut self.state);
        Ok(PyArray1::from_vec(py, sum_trees)) // Zero-copy
    }
}

pub enum Sampler {
    // Gaussian response strategies with different depths
    Depth5Gaussian(PgBartSampler<5, Grow<GaussianResponseStrategy>, SystematicResampling>),
    Depth10Gaussian(PgBartSampler<10, Grow<GaussianResponseStrategy>, SystematicResampling>),
    Depth15Gaussian(PgBartSampler<15, Grow<GaussianResponseStrategy>, SystematicResampling>),
}

impl Sampler {
    /// Dispatch the step method to the appropriate concrete sampler type
    pub fn step(&mut self, rng: &mut SmallRng, state: &mut PgBartState) -> Vec<f64> {
        match self {
            Sampler::Depth5Gaussian(sampler) => sampler.step(rng, state),
            Sampler::Depth10Gaussian(sampler) => sampler.step(rng, state),
            Sampler::Depth15Gaussian(sampler) => sampler.step(rng, state),
        }
    }
}

#[pymodule]
fn pymc_bart(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBartSettings>()?;
    m.add_class::<PySampler>()?;
    Ok(())
}
