mod data;

extern crate pg_bart;

use std::str::FromStr;

use crate::data::ExternalData;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pg_bart::pgbart::{PgBartSettings, PgBartState, Response};
use pyo3::prelude::*;

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
    // split_rules: TODO: Implement SplitRules
    response: String,
    n_trees: usize,
    n_particles: usize,
    leaf_sd: f64,
    batch: (f64, f64),
) -> StateWrapper {
    let data = ExternalData::new(X, y, logp);
    let data = Box::new(data);
    let response = Response::from_str(&response).unwrap();
    let params = PgBartSettings::new(
        n_trees,
        n_particles,
        alpha,
        beta,
        leaf_sd,
        batch,
        split_prior.to_vec().unwrap(),
        response,
    );
    let state = PgBartState::new(params, data);

    StateWrapper { state }
}

#[pyfunction]
fn step<'py>(py: Python<'py>, wrapper: &mut StateWrapper, tune: bool) -> &'py PyArray1<f64> {
    wrapper.state.tune = tune;
    wrapper.state.step();

    let predictions = wrapper.state.predictions();
    let py_array = PyArray1::from_array(py, &predictions.view());

    py_array
}

#[pymodule]
fn bart_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(step, m)?)?;
    Ok(())
}
