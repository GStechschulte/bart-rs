mod data;

extern crate pg_bart;

use crate::data::ExternalData;

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pg_bart::pgbart::{PgBartSettings, PgBartState};
use pyo3::prelude::*;

/// Converts a numpy nd array from Python user to Rust
#[pyfunction]
fn shape(py: Python, arr: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
    let arr_view = arr.as_array();
    println!("arr.as_array(): {:?}", arr_view);
    let data = arr_view.as_standard_layout().into_owned().into_raw_vec();
    println!("data: {:?}", data[0 * 3 + 1]);
    Ok(PyArray2::from_array_bound(py, &arr_view).to_object(py))
}

#[pyfunction]
fn initialize(X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) {
    let data = ExternalData::new(X, y);
}

#[pyclass(unsendable)]
struct StateWrapper {
    state: PgBartState,
}

#[pyfunction]
fn initialize_bart(
    X: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    logp: usize,
    alpha: f64,
    n_trees: usize,
    n_particles: usize,
    kfactor: f64,
    batch: (f64, f64),
    split_prior: PyReadonlyArray1<f64>,
) -> StateWrapper {
    let data = ExternalData::new(X, y);
    let data = Box::new(data);
    let params = PgBartSettings::new(
        n_trees,
        n_particles,
        alpha,
        kfactor,
        batch,
        split_prior.to_vec().unwrap(),
    );
    let state = PgBartState::new(params, data);

    StateWrapper { state }
}

/// A Python module implemented in Rust.
#[pymodule]
fn bart_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_bart, m)?)?;
    Ok(())
}
