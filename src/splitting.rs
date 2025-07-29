//! Split rule trait definitions and implementations for decision trees. The module
//! supports sampling split values from a set of candidates and dividing data points based on
//! the chosen split value.
use std::collections::HashSet;

use numpy::Ix2;
use numpy::ndarray::Array;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use rand::Rng;

pub trait SplitRule {
    // The data type associated with the split rule strategy
    type Value: Copy;

    /// Samples a split value from the candidate points
    fn sample_split_value<I>(&self, rng: &mut impl Rng, candidates: I) -> Option<Self::Value>
    where
        I: Iterator<Item = Self::Value>;

    /// Splits data indices based on feature values and threshold
    fn split_data_indices<I>(
        &self,
        data: &Array<f64, Ix2>,
        feature_idx: usize,
        threshold: Self::Value,
        data_indices: I,
    ) -> (Vec<usize>, Vec<usize>)
    where
        I: Iterator<Item = usize>;
}

/// Continuous split rule.
///
/// Pick a pivot value and split depending on if the variable value is smaller or
/// greater than the value picked.
#[derive(Clone, Copy, Debug)]
pub struct ContinuousSplit;

impl SplitRule for ContinuousSplit {
    type Value = f64;

    fn sample_split_value<I>(&self, rng: &mut impl Rng, candidates: I) -> Option<Self::Value>
    where
        I: Iterator<Item = Self::Value>,
    {
        let mut candidates = candidates.peekable();
        if candidates.peek().is_none() {
            return None;
        }

        let initial = candidates.next().unwrap();
        let (min_val, max_val) = candidates.fold((initial, initial), |(min, max), val| {
            (min.min(val), max.max(val))
        });

        if min_val >= max_val {
            return None;
        }

        Some(rng.random_range(min_val..max_val))
    }

    fn split_data_indices<I>(
        &self,
        data: &Array<f64, Ix2>,
        feature_idx: usize,
        threshold: Self::Value,
        data_indices: I,
    ) -> (Vec<usize>, Vec<usize>)
    where
        I: Iterator<Item = usize>,
    {
        data_indices.partition(|&idx| data[[idx, feature_idx]] < threshold)
    }
}

/// Choose a single categorical value and branch on it if the variable value is
/// that value or not.
#[derive(Clone, Copy, Debug)]
pub struct OneHotSplit;

impl SplitRule for OneHotSplit {
    type Value = i32;

    fn sample_split_value<I>(&self, rng: &mut impl Rng, candidates: I) -> Option<Self::Value>
    where
        I: Iterator<Item = Self::Value>,
    {
        let mut candidates = candidates.peekable();
        if candidates.peek().is_none() {
            return None;
        }

        // Collect into a HashSet to get unique values efficiently.
        let unique_vals: Vec<i32> = candidates.collect::<HashSet<_>>().into_iter().collect();

        if unique_vals.len() <= 1 {
            return None;
        }

        Some(unique_vals[rng.random_range(0..unique_vals.len())])
    }

    fn split_data_indices<I>(
        &self,
        data: &Array<f64, Ix2>,
        feature_idx: usize,
        threshold: Self::Value,
        data_indices: I,
    ) -> (Vec<usize>, Vec<usize>)
    where
        I: Iterator<Item = usize>,
    {
        data_indices.partition(|&idx| (data[[idx, feature_idx]] as i32) == threshold)
    }
}

#[derive(Clone, Debug)]
pub enum SplitRules {
    Continuous(ContinuousSplit),
    OneHot(OneHotSplit),
}

impl SplitRules {
    pub fn from_str(rule_name: &str) -> PyResult<Self> {
        match rule_name {
            "ContinuousSplit" => Ok(SplitRules::Continuous(ContinuousSplit)),
            "OneHotSplit" => Ok(SplitRules::OneHot(OneHotSplit)),
            _ => Err(PyValueError::new_err(format!(
                "Unknown split rule: '{}'. Supported split rules are 'ContinuousSplit' and 'OneHotSplit'.",
                rule_name
            ))),
        }
    }

    pub fn sample_split_value<I>(&self, rng: &mut impl Rng, candidates: I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
    {
        match self {
            SplitRules::Continuous(rule) => rule.sample_split_value(rng, candidates),
            SplitRules::OneHot(rule) => {
                let int_candidates: Vec<i32> = candidates.map(|x| x as i32).collect();
                rule.sample_split_value(rng, int_candidates.into_iter())
                    .map(|x| x as f64)
            }
        }
    }

    pub fn split_data_indices<I>(
        &self,
        data: &Array<f64, Ix2>,
        feature_idx: usize,
        threshold: f64,
        data_indices: I,
    ) -> (Vec<usize>, Vec<usize>)
    where
        I: Iterator<Item = usize>,
    {
        match self {
            SplitRules::Continuous(rule) => {
                rule.split_data_indices(data, feature_idx, threshold, data_indices)
            }
            SplitRules::OneHot(rule) => {
                rule.split_data_indices(data, feature_idx, threshold as i32, data_indices)
            }
        }
    }
}
