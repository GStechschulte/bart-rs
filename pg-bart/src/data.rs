use ndarray::{Array1, Array2};

pub trait PyData {
    fn X(&self) -> Array2<f64>;
    fn y(&self) -> Array1<f64>;
    fn model_logp(&self, v: Array1<f64>) -> f64;
}
