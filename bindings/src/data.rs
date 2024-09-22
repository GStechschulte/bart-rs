#![allow(non_snake_case)]

extern crate pg_bart;

use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pg_bart::data::PyData;

type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> std::os::raw::c_double;

pub struct ExternalData {
    X: Array2<f64>,
    y: Array1<f64>,
    logp: LogpFunc,
}

impl ExternalData {
    pub fn new(X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, logp: usize) -> Self {
        let logp: LogpFunc = unsafe { std::mem::transmute(logp as *const std::ffi::c_void) };
        
        Self {
            // `.to_owned_array()` creates a copy of X and y
            X: X.to_owned_array(),
            y: y.to_owned_array(),
            logp: logp,
        }
    }
}

// TODO: DO NOT CLONE...Use Rc<T>?
impl PyData for ExternalData {
    fn X(&self) -> Array2<f64> {
        self.X.clone()
    }

    fn y(&self) -> Array1<f64> {
        self.y.clone()
    }

    fn model_logp(&self, v: Array1<f64>) -> f64 {
        // todo!("Implement model_logp")
        let logp = self.logp;
        let value = unsafe { logp(v.as_ptr(), v.len()) };

        value
    }
}
