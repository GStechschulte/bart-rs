use std::ffi::c_double;

use numpy::ndarray::{Array, Ix1};

pub trait WeightFn {
    fn log_weight(&self, predictions: &Array<f64, Ix1>) -> f64;
}

/// Weight function backed by a C function pointer from PyMC.
pub struct PyMCWeightFn {
    func_ptr: unsafe extern "C" fn(*const f64, usize) -> c_double,
}

impl PyMCWeightFn {
    /// Create a new PyMCWeightFn from a raw function pointer.
    ///
    /// # Safety
    /// The caller must ensure the function pointer remains valid for
    /// the lifetime of this struct and that it correctly interprets
    /// a (pointer, length) pair as a slice of f64 values.
    pub unsafe fn from_raw(ptr: unsafe extern "C" fn(*const f64, usize) -> c_double) -> Self {
        Self { func_ptr: ptr }
    }
}

impl WeightFn for PyMCWeightFn {
    fn log_weight(&self, predictions: &Array<f64, Ix1>) -> f64 {
        unsafe { (self.func_ptr)(predictions.as_ptr(), predictions.len()) }
    }
}
