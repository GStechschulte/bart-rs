use std::{
    collections::HashMap,
    ffi::{c_double, c_void},
};

pub mod base;
pub mod forest;
pub mod resampling;
pub mod response;
pub mod sampler;
pub mod split_rules;

use crate::base::PgBartState;
use crate::sampler::{LogpFunc, PgBartSampler};
use crate::split_rules::SplitRules;

use numpy::{
    ndarray::{s, Array, Array1, Axis, Ix1, Ix2},
    PyArrayMethods, PyReadonlyArray, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::PyTimeoutError,
    ffi::Py_uintptr_t,
    intern,
    prelude::*,
    types::{PyList, PyTuple},
};
use rand::{
    rng,
    rngs::{SmallRng, StdRng},
    RngCore, SeedableRng,
};

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
    split_rules: HashMap<usize, String>,
    response_rule: String,
    batch_size: (f64, f64),
    rng: SmallRng,
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
        split_rules: HashMap<usize, String>,
        response_rule: String,
        batch_size: (f64, f64),
        seed: u64,
    ) -> Self {
        let rng = SmallRng::seed_from_u64(seed);

        Self {
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
            rng,
        }
    }
}

#[pyclass]
struct PySampler {
    sampler: PgBartSampler,
    state: PgBartState,
}

#[pymethods]
impl PySampler {
    #[staticmethod]
    #[pyo3(text_signature = "(X, y, settings)")]
    fn init(
        X: PyReadonlyArray<f64, Ix2>,
        y: PyReadonlyArray<f64, Ix1>,
        model: usize,
        settings: PyBartSettings,
    ) -> PyResult<PySampler> {
        // to_owned_array performs a deep copy of the underlyng data
        let logp: LogpFunc = unsafe { std::mem::transmute(model as *const c_void) };

        let X_arr = X.to_owned_array();
        let y_arr = y.to_owned_array();

        let forest = vec![0.0; settings.n_trees];
        let weights = vec![0.0; settings.n_trees];
        let predictions = Array1::from_elem(y.len(), y_arr.mean().unwrap());

        let state = PgBartState::new(X_arr, y_arr, forest, weights, predictions);
        let sampler = PgBartSampler::new(logp, settings);

        Ok(PySampler {
            sampler: sampler,
            state: state,
        })
    }

    /// Runs the Particle Gibbs sampler sequentially for `M` iterations where `M` is the number
    /// of trees.
    ///
    /// A single step will initialize a set of particles `N`, of which one will replace the `m'th` tree. To decide which particle will replace the current tree, the `N`
    /// particles are grown until the probability of a leaf node expanding is less than a
    /// random value in the interval [0, 1].
    ///
    /// The grown particles are resampled according to their log-likelihood, of which
    /// one is selected to replace the `m'th` tree.
    fn step(&mut self) {
        self.sampler.step(&mut self.state);
    }
}

#[pymodule]
fn pymc_bart(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBartSettings>()?;
    m.add_class::<PySampler>()?;
    Ok(())
}
