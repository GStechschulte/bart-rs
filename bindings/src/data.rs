#![allow(non_snake_case)]
use std::os::raw::{c_double, c_uint, c_void};

extern crate pg_bart;

use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pg_bart::data::PyData;

// extern keyword defines the variable (or function) defined in some other program
// that the Rust executable will be linked with.
//
// Use the std::os::raw module to define the Rust type that are guaranteed to
// have the same representation as the C type
type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

pub struct ExternalData {
    X: Array2<f64>,
    y: Array1<f64>,
    logp: LogpFunc,
}

impl ExternalData {
    pub fn new(X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, logp: usize) -> Self {
        let logp: LogpFunc = unsafe { std::mem::transmute(logp as *const c_void) };

        Self {
            // `.to_owned_array()` creates a copy of X and y
            X: X.to_owned_array(),
            y: y.to_owned_array(),
            logp,
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

    fn evaluate_logp(&self, x: Array1<f64>) -> f64 {
        let logp = unsafe { (self.logp)(x.as_ptr(), x.len()) };
        logp
    }
}
