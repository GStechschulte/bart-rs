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
