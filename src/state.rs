use numpy::ndarray::Array1;

use crate::tree::TreeArrays;

/// BART sampler state
#[derive(Clone)]
pub struct BartState {
    pub forest: Vec<TreeArrays>,
    pub predictions: Array1<f64>,
    pub variable_inclusion: Vec<u32>,
    pub next_tree_idx: usize,
    pub tune: bool,
}

/// Diagnostic information from a single sampling step.
pub struct BartInfo {
    pub log_likelihood: f64,
    pub acceptance_count: usize,
    pub tree_depths: Vec<u8>,
}
