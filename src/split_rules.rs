//! Split rule trait definitions and implementations for decision trees. The module
//! supports sampling split values from a set of candidates and dividing data points based on
//! the chosen split value.

use numpy::ndarray::Array2;
use rand::rngs::SmallRng;
use rand::Rng;

pub trait SplitRule {
    // The data type associated with the split rule strategy
    type Value;

    /// Samples a split value from the candidate points
    fn sample_split_value(
        &self,
        candidates: &[Self::Value],
        rng: &mut SmallRng,
    ) -> Option<Self::Value>;

    /// Splits data indices based on feature values and threshold
    fn split_data_indices(
        &self,
        data: &Array2<f64>,
        feature_idx: usize,
        threshold: Self::Value,
        data_indices: &[usize],
    ) -> (Vec<usize>, Vec<usize>);
}

/// Continuous split rule.
///
/// Pick a pivot value and split depending on if the variable value is smaller or
/// greater than the value picked.
pub struct ContinuousSplitRule;

impl SplitRule for ContinuousSplitRule {
    type Value = f64;

    fn sample_split_value(
        &self,
        candidates: &[Self::Value],
        rng: &mut SmallRng,
    ) -> Option<Self::Value> {
        if candidates.is_empty() {
            return None;
        }

        // For continuous variables, we can sample any value between min and max
        let min_val = candidates.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = candidates.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if min_val >= max_val {
            return None;
        }

        // Sample uniformly between min and max
        Some(rng.gen_range(min_val..max_val))
    }

    fn split_data_indices(
        &self,
        data: &Array2<f64>,
        feature_idx: usize,
        threshold: Self::Value,
        data_indices: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in data_indices {
            if data[[idx, feature_idx]] < threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        (left_indices, right_indices)
    }
}

/// Choose a single categorical value and branch on it if the variable value is
/// that value or not.
pub struct OneHotSplit;

impl SplitRule for OneHotSplit {
    type Value = i32;

    fn sample_split_value(
        &self,
        candidates: &[Self::Value],
        rng: &mut SmallRng,
    ) -> Option<Self::Value> {
        if candidates.is_empty() {
            return None;
        }

        // For categorical variables, randomly select one of the unique values
        let unique_vals: std::collections::HashSet<_> = candidates.iter().copied().collect();
        let unique_vec: Vec<_> = unique_vals.into_iter().collect();

        if unique_vec.len() <= 1 {
            return None;
        }

        Some(unique_vec[rng.gen_range(0..unique_vec.len())])
    }

    fn split_data_indices(
        &self,
        data: &Array2<f64>,
        feature_idx: usize,
        threshold: Self::Value,
        data_indices: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in data_indices {
            if data[[idx, feature_idx]] as i32 == threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        (left_indices, right_indices)
    }
}

pub enum SplitRules {
    Continuous(ContinuousSplitRule),
    OneHot(OneHotSplit),
}

impl SplitRules {
    pub fn sample_split_value(&self, candidates: &[f64], rng: &mut SmallRng) -> Option<f64> {
        match self {
            SplitRules::Continuous(rule) => rule.sample_split_value(candidates, rng),
            SplitRules::OneHot(rule) => {
                let int_candidates: Vec<i32> = candidates.iter().map(|&x| x as i32).collect();
                rule.sample_split_value(&int_candidates, rng)
                    .map(|x| x as f64)
            }
        }
    }

    pub fn split_data_indices(
        &self,
        data: &Array2<f64>,
        feature_idx: usize,
        threshold: f64,
        data_indices: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
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
