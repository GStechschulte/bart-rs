use rand::Rng;
use rand_distr::Uniform;
use std::any::Any;
use std::f64;
use std::iter::Iterator;

use crate::tree::SplitValue;

/// Interface for split strategies.
pub trait SplitRule: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn get_split_value_dyn(&self, candidates: &dyn Any) -> Option<SplitValue>;
    fn divide_dyn(
        &self,
        candidates: &dyn Any,
        split_value: &SplitValue,
    ) -> (Vec<usize>, Vec<usize>);
}

/// Standard continuous split rule. Pick a pivot value and split
/// depending on if variable is smaller or greater than the value picked.
pub struct ContinuousSplit;

impl SplitRule for ContinuousSplit {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_split_value_dyn(&self, candidates: &dyn Any) -> Option<SplitValue> {
        if let Some(candidates) = candidates.downcast_ref::<Vec<f64>>() {
            if candidates.len() > 1 {
                let idx = rand::thread_rng().gen_range(0..candidates.len());
                Some(SplitValue::Float(candidates[idx]))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn divide_dyn(
        &self,
        candidates: &dyn Any,
        split_value: &SplitValue,
    ) -> (Vec<usize>, Vec<usize>) {
        if let Some(candidates) = candidates.downcast_ref::<Vec<f64>>() {
            match split_value {
                SplitValue::Float(threshold) => {
                    let (left, right): (Vec<_>, Vec<_>) =
                        (0..candidates.len()).partition(|&idx| candidates[idx] <= *threshold);
                    (left, right)
                }
                SplitValue::Integer(threshold) => {
                    let threshold = *threshold as f64;
                    let (left, right): (Vec<_>, Vec<_>) =
                        (0..candidates.len()).partition(|&idx| candidates[idx] <= threshold);
                    (left, right)
                }
            }
        } else {
            (vec![], vec![])
        }
    }
}

/// Choose a single categorical value and branch on it if the variable is that value or not.
pub struct OneHotSplit;

impl SplitRule for OneHotSplit {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_split_value_dyn(&self, candidates: &dyn Any) -> Option<SplitValue> {
        if let Some(candidates) = candidates.downcast_ref::<Vec<i32>>() {
            if candidates.len() > 1 && !candidates.iter().all(|&x| x == candidates[0]) {
                let idx = rand::thread_rng().gen_range(0..candidates.len());
                Some(SplitValue::Integer(candidates[idx]))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn divide_dyn(
        &self,
        candidates: &dyn Any,
        split_value: &SplitValue,
    ) -> (Vec<usize>, Vec<usize>) {
        if let Some(candidates) = candidates.downcast_ref::<Vec<i32>>() {
            match split_value {
                SplitValue::Integer(threshold) => {
                    let (left, right): (Vec<_>, Vec<_>) =
                        (0..candidates.len()).partition(|&idx| candidates[idx] == *threshold);
                    (left, right)
                }
                SplitValue::Float(threshold) => {
                    let threshold = *threshold as i32;
                    let (left, right): (Vec<_>, Vec<_>) =
                        (0..candidates.len()).partition(|&idx| candidates[idx] == threshold);
                    (left, right)
                }
            }
        } else {
            (vec![], vec![])
        }
    }
}

/// Choose a random subset of the categorical values and branch on belonging to that set.
///
/// This is the approach taken by Sameer K. Deshpande.
/// flexBART: Flexible Bayesian regression trees with categorical predictors. arXiv,
/// `link <https://arxiv.org/abs/2211.04459>`__
pub struct SubsetSplit;
