use ndarray::{Array1, Array2};

pub trait PyData {
    fn X(&self) -> Array2<f64>;
    fn y(&self) -> Array1<f64>;
    fn evaluate_logp(&self, x: Array1<f64>) -> Result<(f64, Vec<f64>), &'static str>;
}
