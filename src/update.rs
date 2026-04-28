/// Represents the decision outcome for mutation proposals.
#[derive(Clone, Debug)]
pub enum MutationDecision {
    Accept(TreeProposal),
    Reject,
}

/// BART tree proposal containing all information needed for a growth mutation.
///
/// Sample partitioning is performed by the particle from its `leaf_to_samples`
/// cache, so the proposal does not carry the affected sample list.
#[derive(Clone, Debug)]
pub struct TreeProposal {
    pub node_idx: usize,
    pub split_var: u32,
    pub split_val: f64,
    pub left_value: f64,
    pub right_value: f64,
}
