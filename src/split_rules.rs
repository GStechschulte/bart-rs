//! Split rule trait definitions and implementations for decision trees. The module
//! supports sampling split values from a set of candidates and dividing data points based on //! the chosen split value.
//!
//! Includes:
//! - `SplitRule`: Trait defining the interface for split rules.
//! - `ContinuousSplit`: Splitting strategy for continuous variables.
//! - `OneHotSplit`: Splitting strategy for categorical variables using one-hot encoding.
//! - `SubsetSplit`: Splitting strategy for subset-based splitting of categorical variables.
//! - `SplitRuleType`: Enum encapsulating the different split rule types.

use std::f64;
use std::iter::Iterator;

use rand::Rng;

/// Split rule interface for defining split rule strategies.
pub trait SplitRule {
    /// The data type associated with the split rule strategy.
    type Value;

    /// Samples a split value from the candidate points.
    fn sample_split_value(&self, candidates: &[Self::Value]) -> Option<Self::Value>;
    /// Divides the candidates left and right according to the split value.
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

/// Holds the split rule strategies as enum variants.
pub enum SplitRuleType {
    /// Continuous implements the `ContinuousSplit` strategy.
    Continuous(ContinuousSplit),
    /// OneHot implements the `OneHotSplit` strategy.
    OneHot(OneHotSplit),
}
