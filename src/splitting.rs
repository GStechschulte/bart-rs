//! Split rule trait definitions and implementations for decision trees. The module
//! supports sampling split values from a set of candidates and dividing data points based on
//! the chosen split value.
use numpy::ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::Rng;

pub trait SplitRule {
    // The data type associated with the split rule strategy
    type Value;

    /// Samples a split value from the candidate points
    fn sample_split_value(
        &self,
        rng: &mut impl Rng,
        candidates: &[Self::Value],
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
#[derive(Clone, Copy, Debug)]
pub struct ContinuousSplit;

impl SplitRule for ContinuousSplit {
    type Value = f64;

    fn sample_split_value(
        &self,
        rng: &mut impl Rng,
        candidates: &[Self::Value],
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
        Some(rng.random_range(min_val..max_val))
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
#[derive(Clone, Copy, Debug)]
pub struct OneHotSplit;

impl SplitRule for OneHotSplit {
    type Value = i32;

    fn sample_split_value(
        &self,
        rng: &mut impl Rng,
        candidates: &[Self::Value],
    ) -> Option<Self::Value> {
        if candidates.is_empty() {
            return None;
        }

        // For categorical variables, randomly select one of the unique values
        // Better approach - avoid double collection
        let mut unique_vals: Vec<Self::Value> = candidates.iter().copied().collect();
        unique_vals.sort_unstable();
        unique_vals.dedup();

        if unique_vals.len() <= 1 {
            return None;
        }

        Some(unique_vals[rng.random_range(0..unique_vals.len())])
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

    pub fn sample_split_value(&self, rng: &mut impl Rng, candidates: &[f64]) -> Option<f64> {
        match self {
            SplitRules::Continuous(rule) => rule.sample_split_value(rng, candidates),
            SplitRules::OneHot(rule) => {
                let int_candidates: Vec<i32> = candidates.iter().map(|&x| x as i32).collect();
                rule.sample_split_value(rng, &int_candidates)
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
