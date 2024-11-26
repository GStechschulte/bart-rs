extern crate pg_bart;

use pg_bart::data::PyData;

use std::{
    os::raw::{c_double, c_void},
    rc::Rc,
};

use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

// extern keyword defines the variable (or function) defined in some other program
// that the Rust executable will be linked with
type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

pub struct ExternalData {
    X: Rc<Array2<f64>>,
    y: Rc<Array1<f64>>,
    logp: LogpFunc,
}

impl ExternalData {
    pub fn new(X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, logp: usize) -> Self {
        let logp: LogpFunc = unsafe { std::mem::transmute(logp as *const c_void) };

        Self {
            X: Rc::new(X.to_owned_array()),
            y: Rc::new(y.to_owned_array()),
            logp,
        }
    }
}

impl PyData for ExternalData {
    fn X(&self) -> Rc<Array2<f64>> {
        Rc::clone(&self.X)
    }

    fn y(&self) -> Rc<Array1<f64>> {
        Rc::clone(&self.y)
    }

    fn evaluate_logp(&self, x: Array1<f64>) -> f64 {
        let logp = unsafe { (self.logp)(x.as_ptr(), x.len()) };
        logp
    }
}
