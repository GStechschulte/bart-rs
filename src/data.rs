//! Trait implementation to handle data provided by the Python user.

//! Trait implementation to handle data provided by the Python user.

use std::{
    os::raw::{c_double, c_void},
    rc::Rc,
};

use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

/// Interface for interaction with data provided by the Python user.
pub trait PyData {
    #![allow(non_snake_case)]
    /// Covariate matrix
    fn X(&self) -> &Array2<f64>;
    /// Response (target) vector
    fn y(&self) -> &Array1<f64>;
    /// Evaluate log-probability given data `x` where x has shape [n_groups, n_samples]
    fn evaluate_logp(&self, x: &Array2<f64>) -> f64;
}

// extern keyword defines the variable (or function) defined in some other program
// that the Rust executable will be linked with
type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

/// Container used to store external data passed by the Python user.
pub struct ExternalData {
    // X: Rc<Array2<f64>>,
    // y: Rc<Array1<f64>>,
    X: Array2<f64>,
    y: Array1<f64>,
    logp: LogpFunc,
}

impl ExternalData {
    /// Creates a new `ExternalData` struct.
    pub fn new(X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, logp: usize) -> Self {
        let logp: LogpFunc = unsafe { std::mem::transmute(logp as *const c_void) };
        Self {
            // X: Rc::new(X.to_owned_array()),
            // y: Rc::new(y.to_owned_array()),
            X: X.to_owned_array(),
            y: y.to_owned_array(),
            logp,
        }
    }
}

impl PyData for ExternalData {
    fn X(&self) -> &Array2<f64> {
        &self.X
    }

    fn y(&self) -> &Array1<f64> {
        &self.y
    }

    // fn X(&self) -> Rc<Array2<f64>> {
    //     Rc::clone(&self.X)
    // }

    // fn y(&self) -> Rc<Array1<f64>> {
    //     Rc::clone(&self.y)
    // }

    fn evaluate_logp(&self, x: &Array2<f64>) -> f64 {
        // We assume x has shape [n_groups, n_samples]
        // Convert to contiguous memory layout for C function
        let x_flat = x.t().as_standard_layout().view();

        unsafe { (self.logp)(x_flat.as_ptr(), x_flat.len()) }
    }
}
