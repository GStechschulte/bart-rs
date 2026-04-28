//! Split rule trait definitions and implementations for decision trees.
//!
//! Supports sampling split values from candidates and dividing data points
//! based on the chosen split value.
use std::collections::HashSet;

use numpy::Ix2;
use numpy::ndarray::Array;
use rand::Rng;

/// Trait for split rule strategies.
pub trait SplitRule {
    type Value: Copy;

    /// Sample a split value from the candidate points.
    fn sample_split_value<I>(&self, rng: &mut impl Rng, candidates: I) -> Option<Self::Value>
    where
        I: Iterator<Item = Self::Value>;

    /// Split data indices based on feature values and threshold.
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

/// Continuous split rule: pick a pivot and split on < vs >=.
#[derive(Clone, Copy, Debug)]
pub struct ContinuousSplit;

impl SplitRule for ContinuousSplit {
    type Value = f64;

    fn sample_split_value<I>(&self, rng: &mut impl Rng, candidates: I) -> Option<Self::Value>
    where
        I: Iterator<Item = Self::Value>,
    {
        let mut candidates = candidates.peekable();
        candidates.peek()?;

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

/// One-hot split rule: branch on equality with a single categorical value.
#[derive(Clone, Copy, Debug)]
pub struct OneHotSplit;

impl SplitRule for OneHotSplit {
    type Value = i32;

    fn sample_split_value<I>(&self, rng: &mut impl Rng, candidates: I) -> Option<Self::Value>
    where
        I: Iterator<Item = Self::Value>,
    {
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

/// Enum for dynamic dispatch over split rules.
#[derive(Clone, Debug)]
pub enum SplitRules {
    Continuous(ContinuousSplit),
    OneHot(OneHotSplit),
}

impl SplitRules {
    pub fn from_name(rule_name: &str) -> Result<Self, String> {
        match rule_name {
            "ContinuousSplit" => Ok(SplitRules::Continuous(ContinuousSplit)),
            "OneHotSplit" => Ok(SplitRules::OneHot(OneHotSplit)),
            _ => Err(format!(
                "Unknown split rule: '{}'. Supported: 'ContinuousSplit', 'OneHotSplit'.",
                rule_name
            )),
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
