// TODO:    Implement different split rule traits for the DecisionTree
//          as the user can pass a `split_rules` argument
//          `split_rules` is either `ContinuousSplitRule`, `OneHotSplitRule`
//          or `SubsetSplitRule`.
//
//          The code below is simply a template!!!

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
