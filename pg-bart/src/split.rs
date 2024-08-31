// Trait for sampling leaf values
pub trait LeafValueSampler {
    fn sample_leaf_value(&self, mu: f64, kfactor: f64) -> f64;
}

// Trait for sampling split probabilities
pub trait SplitProbabilitySampler {
    fn sample_expand_flag(&self, depth: usize) -> bool;
    fn sample_split_index(&self) -> usize;
    fn sample_split_value(&self, candidates: &[f64]) -> Option<f64>;
}

pub trait SplitRule {
    fn get_split_value(&self, values: &[f64]) -> Option<f64>;
    fn divide(&self, values: &[f64], split_value: f64) -> Vec<bool>;
}

pub struct ContinuousSplitRule;

impl SplitRule for ContinuousSplitRule {
    fn get_split_value(&self, values: &[f64]) -> Option<f64> {
        todo!("Implement continuous split value selection");
    }

    fn divide(&self, values: &[f64], split_value: f64) -> Vec<bool> {
        todo!("Implement continuous split")
    }
}
