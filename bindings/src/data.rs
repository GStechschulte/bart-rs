#![allow(non_snake_case)]

extern crate pg_bart;

use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pg_bart::data::PyData;

// TODO: Is this necessary?
// Lifetime annotation ensures PyReadonlyArray refereces
// do not outlive ExternalData. ExternalData lives for the
// lifetime of the PyObject
pub struct ExternalData<'py> {
    X: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
}

impl<'py> ExternalData<'py> {
    pub fn new(X: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray1<'py, f64>) -> Self {
        ExternalData { X, y }
    }
}

impl<'py> PyData for ExternalData<'py> {
    fn X(&self) -> Array2<f64> {
        self.X.to_owned_array()
    }

    fn y(&self) -> Array1<f64> {
        self.y.to_owned_array()
    }

    fn model_logp(&self, v: Array1<f64>) -> f64 {
        todo!("Implement model_logp...")
    }
}
