//! Trait implementation to handle data provided by the Python user.

use std::rc::Rc;

use ndarray::{Array1, Array2};

/// Handles data provided by the Python user.
pub trait PyData {
    #![allow(non_snake_case)]
    fn X(&self) -> Rc<Array2<f64>>;
    fn y(&self) -> Rc<Array1<f64>>;
    fn evaluate_logp(&self, x: Array1<f64>) -> f64;
}
