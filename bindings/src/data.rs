#![allow(non_snake_case)]

extern crate pg_bart;

use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pg_bart::data::PyData;

pub struct ExternalData {
    X: Array2<f64>,
    y: Array1<f64>,
}

impl ExternalData {
    pub fn new(X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> Self {
        ExternalData {
            // `.to_owned_array()` creates a copy of X and y
            X: X.to_owned_array(),
            y: y.to_owned_array(),
        }
    }
}

impl PyData for ExternalData {
    fn X(&self) -> Array2<f64> {
        self.X.clone()
    }

    fn y(&self) -> Array1<f64> {
        self.y.clone()
    }

    fn model_logp(&self, v: Array1<f64>) -> f64 {
        todo!("Implement model_logp")
    }
}
