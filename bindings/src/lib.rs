mod data;

extern crate pg_bart;

use crate::data::PythonData;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pg_bart::data::ExternalData;
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
fn initialize(py: Python, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) {
    let data = PythonData::new(X, y);
}

/// A Python module implemented in Rust.
#[pymodule]
fn bart_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    Ok(())
}
