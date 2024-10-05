use rand::Rng;
use std::{collections::HashSet, f64};

/// Interface for split strategies.
pub trait SplitRule {
    type T;

    fn get_split_value(&self, candidates: &[Self::T]) -> Option<Self::T>;
    fn divide(&self, candidates: &[Self::T], split_value: Self::T) -> Vec<bool>;
}

/// Standard continuous split rule. Pick a pivot value and split
/// depending on if variable is smaller or greater than the value picked.
pub struct ContinuousSplit;

impl SplitRule for ContinuousSplit {
    type T = f64;

    fn get_split_value(&self, candidates: &[f64]) -> Option<f64> {
        if candidates.len() > 1 {
            let idx = rand::thread_rng().gen_range(0..candidates.len());
            Some(candidates[idx])
        } else {
            None
        }
    }

    fn divide(&self, candidates: &[f64], split_value: f64) -> Vec<bool> {
        candidates.iter().map(|&x| x <= split_value).collect()
    }
}

/// Choose a single categorical value and branch on it if the variable is that value or not.
pub struct OneHotSplit;

impl SplitRule for OneHotSplit {
    // TODO: macro to generate an implementation for multple integer types?
    type T = i32;

    fn get_split_value(&self, candidates: &[i32]) -> Option<i32> {
        if candidates.len() > 1 && !candidates.iter().all(|&x| x == candidates[0]) {
            let idx = rand::thread_rng().gen_range(0..candidates.len());
            Some(candidates[idx])
        } else {
            None
        }
    }

    fn divide(&self, candidates: &[i32], split_value: i32) -> Vec<bool> {
        candidates.iter().map(|&x| x == split_value).collect()
    }
}

/// Choose a random subset of the categorical values and branch on belonging to that set.
///
/// This is the approach taken by Sameer K. Deshpande.
/// flexBART: Flexible Bayesian regression trees with categorical predictors. arXiv,
/// `link <https://arxiv.org/abs/2211.04459>`__
pub struct SubsetSplit;

impl SplitRule for SubsetSplit {
    type T = i32;

    fn get_split_value(&self, candidates: &[i32]) -> Option<i32> {
        todo!("Implement")
    }

    fn divide(&self, candidates: &[i32], split_value: i32) -> Vec<bool> {
        todo!("Implement")
    }
}
