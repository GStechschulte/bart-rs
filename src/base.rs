use numpy::{Ix1, ndarray::Array};

use crate::particle::Tree;

#[derive(Clone, Debug)]
pub struct BartState<const MAX_NODES: usize> {
    pub ensemble_trees: Vec<Tree<MAX_NODES>>,
    pub ensemble_predictions: Array<f64, Ix1>,
}

impl<const MAX_NODES: usize> Default for BartState<MAX_NODES> {
    fn default() -> Self {
        Self {
            ensemble_trees: Vec::new(),
            ensemble_predictions: Array::zeros(0),
        }
    }
}

impl<const MAX_NODES: usize> BartState<MAX_NODES> {
    pub fn new(n_trees: usize, init_predictions: Array<f64, Ix1>) -> Self {
        Self {
            ensemble_trees: Vec::with_capacity(n_trees),
            ensemble_predictions: init_predictions,
        }
    }
}
