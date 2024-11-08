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
type LogpFunc = unsafe extern "C" fn(
    dim: c_uint,
    x: *const c_double,
    out: *mut c_double,
    logp: *mut c_double,
    user_data: *mut c_void,
) -> i64;

pub struct ExternalData {
    X: Array2<f64>,
    y: Array1<f64>,
    logp: LogpFunc,
    n_dim: usize,
    user_data: *mut c_void,
}

impl ExternalData {
    pub fn new(
        X: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        logp: usize,
        n_dim: usize,
        user_data: usize,
    ) -> Self {
        // let logp: LogpFunc = unsafe { std::mem::transmute(logp as *const std::ffi::c_void) };
        let logp: LogpFunc = unsafe { std::mem::transmute(logp) };
        let user_data = user_data as *mut c_void;

        Self {
            // `.to_owned_array()` creates a copy of X and y
            X: X.to_owned_array(),
            y: y.to_owned_array(),
            logp,
            n_dim,
            user_data,
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

    fn evaluate_logp(&self, x: Array1<f64>) -> Result<(f64, Vec<f64>), &'static str> {
        println!("x.len(): {:?}, self.n_dim: {:?}", x.len(), self.n_dim);
        if x.len() != self.n_dim {
            return Err("Input dimension mismatch");
        }

        let mut logp = 0.0;
        let mut gradient = vec![0.0; self.n_dim];

        unsafe {
            let result = (self.logp)(
                self.n_dim as c_uint,
                x.as_ptr(),
                gradient.as_mut_ptr(),
                &mut logp,
                self.user_data,
            );

            match result {
                0 => Ok((logp, gradient)),
                1 => Err("Unknown exception occurred"),
                3 => Err("Gradient contains non-finite values"),
                4 => Err("Log probability is non-finite"),
                _ => Err("Unknown error occurred"),
            }
        }
    }
}
