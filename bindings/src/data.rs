#![allow(non_snake_case)]

extern crate pg_bart;

use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pg_bart::data::{ExternalData, Matrix};

pub struct PythonData {
    X: Matrix<f64>,
    y: Vec<f64>,
}

impl PythonData {
    pub fn new(X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> Self {
        let X = X.as_array();
        let X = Matrix::from_vec(
            X.as_standard_layout().into_owned().into_raw_vec(),
            X.shape()[0],
            X.shape()[1],
        );

        let y = y.to_vec().unwrap();

        Self { X, y }
    }
}

impl ExternalData for PythonData {
    fn X(&self) -> &Matrix<f64> {
        &self.X
    }

    fn y(&self) -> &Vec<f64> {
        &self.y
    }
}
