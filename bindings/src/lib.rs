#![allow(non_snake_case)]

mod data;

extern crate pg_bart;

use std::str::FromStr;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pg_bart::ops::Response;
use pg_bart::pgbart::{PgBartSettings, PgBartState};
use pg_bart::split_rules::{ContinuousSplit, OneHotSplit, SplitRuleType};
use pyo3::prelude::*;

use crate::data::ExternalData;

/// `StateWrapper` wraps around `PgBartState` to hold state pertaining to
/// the Particle Gibbs sampler.
///
/// This class is `unsendable`, i.e., it cannot be sent across threads safely.
#[pyclass(unsendable)]
struct StateWrapper {
    state: PgBartState,
}

#[pyfunction]
fn initialize(
    X: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    logp: usize,
    alpha: f64,
    beta: f64,
    split_prior: PyReadonlyArray1<f64>,
    split_rules: Vec<String>,
    response: String,
    n_trees: usize,
    n_particles: usize,
    leaf_sd: f64,
    batch: (f64, f64),
    leaves_shape: usize,
) -> PyResult<StateWrapper> {
    // Heap allocation because size of 'ExternalData' is not known at compile time
    let data = Box::new(ExternalData::new(X, y, logp));
    let response = Response::from_str(&response).unwrap();
    let mut rules: Vec<SplitRuleType> = Vec::new();

    for rule in split_rules {
        let split = match rule.as_str() {
            "ContinuousSplit" => SplitRuleType::Continuous(ContinuousSplit),
            "OneHotSplit" => SplitRuleType::OneHot(OneHotSplit),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown split type: {}",
                    rule
                )))
            }
        };
        rules.push(split);
    }

    let params = PgBartSettings::new(
        n_trees,
        n_particles,
        alpha,
        beta,
        leaf_sd,
        batch,
        split_prior.to_vec().unwrap(),
        response,
        rules,
    );
    let state = PgBartState::new(params, data);

    Ok(StateWrapper { state })
}

#[pyfunction]
fn step<'py>(
    py: Python<'py>,
    wrapper: &mut StateWrapper,
    tune: bool,
) -> (&'py PyArray1<f64>, &'py PyArray1<i32>) {
    // Update whether or not pm.sampler is in tuning phase or not
    wrapper.state.tune = tune;
    // Run the Particle Gibbs sampler
    wrapper.state.step();

    // Get predictions (sum of trees) and convert to PyArray
    let predictions = wrapper.state.predictions();
    let py_preds_array = PyArray1::from_array(py, &predictions.view());

    // Get variable inclusion counter and convert to PyArray
    let variable_inclusion = wrapper.state.variable_inclusion().clone();
    let py_variable_inclusion_array = PyArray1::from_vec(py, variable_inclusion);

    (py_preds_array, py_variable_inclusion_array)
}

#[pymodule]
fn pymc_bart(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(step, m)?)?;

    Ok(())
}
