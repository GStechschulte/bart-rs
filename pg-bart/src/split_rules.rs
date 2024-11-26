use std::f64;
use std::iter::Iterator;

use rand::Rng;

pub trait SplitRule {
    type Value;

    fn sample_split_value(&self, candidates: &[Self::Value]) -> Option<Self::Value>;
    fn divide(
        &self,
        candidates: &[Self::Value],
        split_value: &Self::Value,
    ) -> (Vec<usize>, Vec<usize>);
}

/// Standard continuous split rule. Pick a pivot value and split
/// depending on if variable is smaller or greater than the value picked.
pub struct ContinuousSplit;

impl SplitRule for ContinuousSplit {
    type Value = f64;

    fn sample_split_value(&self, candidates: &[f64]) -> Option<f64> {
        if candidates.len() > 1 {
            let idx = rand::thread_rng().gen_range(0..candidates.len());
            Some(candidates[idx])
        } else {
            None
        }
    }

    fn divide(&self, candidates: &[f64], split_value: &f64) -> (Vec<usize>, Vec<usize>) {
        let (left, right): (Vec<usize>, Vec<usize>) =
            (0..candidates.len()).partition(|&idx| candidates[idx] <= *split_value);
        (left, right)
    }
}

/// Choose a single categorical value and branch on it if the variable is that value or not.
pub struct OneHotSplit;

impl SplitRule for OneHotSplit {
    type Value = i32;

    fn sample_split_value(&self, candidates: &[i32]) -> Option<i32> {
        if candidates.len() > 1 && !candidates.iter().all(|&x| x == candidates[0]) {
            let idx = rand::thread_rng().gen_range(0..candidates.len());
            Some(candidates[idx])
        } else {
            None
        }
    }

    fn divide(&self, candidates: &[i32], split_value: &i32) -> (Vec<usize>, Vec<usize>) {
        let (left, right): (Vec<usize>, Vec<usize>) =
            (0..candidates.len()).partition(|&idx| candidates[idx] == *split_value);
        (left, right)
    }
}

/// Choose a random subset of the categorical values and branch on belonging to that set.
///
/// This is the approach taken by Sameer K. Deshpande.
/// flexBART: Flexible Bayesian regression trees with categorical predictors. arXiv,
/// `link <https://arxiv.org/abs/2211.04459>`__
pub struct SubsetSplit;

pub enum SplitRuleType {
    Continuous(ContinuousSplit),
    OneHot(OneHotSplit),
}
