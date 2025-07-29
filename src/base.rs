use numpy::{ndarray::Array, Ix1};

use crate::particle::Tree;

#[derive(Clone, Debug)]
pub struct BartState<const MAX_NODES: usize> {
    pub ensemble_trees: Vec<Tree<MAX_NODES>>,
    pub ensemble_predictions: Array<f64, Ix1>,
}

impl<const MAX_NODES: usize> BartState<MAX_NODES> {
    pub fn new(trees: Vec<Tree<MAX_NODES>>, init_predictions: Array<f64, Ix1>) -> Self {
        Self {
            ensemble_trees: trees,
            ensemble_predictions: init_predictions,
        }
    }
}
