use numpy::ndarray::Array1;

use crate::tree::TreeArrays;

/// Complete state of the BART sampler, consumed and reproduced by each step.
#[derive(Clone)]
pub struct BartState {
    pub forest: Vec<TreeArrays>,
    pub predictions: Array1<f64>,
    pub variable_inclusion: Vec<u32>,
    /// Round-robin index of the next tree to update.
    pub next_tree_idx: usize,
    /// Whether the sampler is in tune mode (selects which batch fraction to use).
    pub tune: bool,
}

/// Diagnostic information from a single sampling step.
pub struct BartInfo {
    pub log_likelihood: f64,
    pub acceptance_count: usize,
    pub tree_depths: Vec<u8>,
}
