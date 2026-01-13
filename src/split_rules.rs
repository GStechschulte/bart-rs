//! Split rule trait definitions and implementations for decision trees. The module
//! supports sampling split values from a set of candidates and dividing data points based on //! the chosen split value.

use std::collections::HashSet;
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
        let mut iter = candidates.iter().copied().filter(|v| v.is_finite());
        let first = iter.next()?;
        let (min_val, max_val) = iter.fold((first, first), |(min_v, max_v), val| {
            (min_v.min(val), max_v.max(val))
        });

        if !min_val.is_finite() || !max_val.is_finite() || min_val >= max_val {
            return None;
        }

        Some(rand::thread_rng().gen_range(min_val..max_val))
    }

    fn divide(&self, candidates: &[f64], split_value: &f64) -> (Vec<usize>, Vec<usize>) {
        let (left, right): (Vec<usize>, Vec<usize>) =
            (0..candidates.len()).partition(|&idx| candidates[idx] < *split_value);
        (left, right)
    }
}

/// Choose a single categorical value and branch on it if the variable is that value or not.
pub struct OneHotSplit;

impl SplitRule for OneHotSplit {
    type Value = i32;

    fn sample_split_value(&self, candidates: &[i32]) -> Option<i32> {
        let mut iter = candidates.iter().copied();
        let first = match iter.next() {
            Some(v) => v,
            None => return None,
        };

        if iter.clone().all(|v| v == first) {
            return None;
        }

        let unique: std::collections::HashSet<i32> = iter.chain(std::iter::once(first)).collect();
        let mut unique_vals: Vec<i32> = unique.into_iter().collect();
        if unique_vals.is_empty() {
            return None;
        }

        Some(unique_vals[rand::thread_rng().gen_range(0..unique_vals.len())])
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
