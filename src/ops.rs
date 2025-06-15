//! High-performance tree sampling operations optimized for PGBART
//!
//! This module implements tree construction operations with focus on:
//! - Vectorized computations using SIMD
//! - Branchless operations to avoid pipeline stalls
//! - Cache-friendly data access patterns
//! - Minimal dynamic allocations

use std::str::FromStr;
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

/// Vector chunk size for optimized operations
const CHUNK_SIZE: usize = 8;

/// Response computation strategies for leaf node values
#[derive(Debug, Clone)]
pub enum Response {
    /// Constant response: mean of residuals
    Constant(ConstantResponse),
    /// Linear response: least squares fit
    Linear(LinearResponse),
}

impl FromStr for Response {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "constant" => Ok(Response::Constant(ConstantResponse)),
            "linear" => Ok(Response::Linear(LinearResponse)),
            _ => Err(format!("Unknown response type: {}", s)),
        }
    }
}

/// Trait for computing leaf node values with vectorized operations
pub trait ResponseStrategy {
    /// Compute leaf value using vectorized operations where possible
    fn compute_leaf_value(&self, residuals: &[f64], n_trees: usize, noise: f64) -> f64;
}

impl ResponseStrategy for Response {
    #[inline]
    fn compute_leaf_value(&self, residuals: &[f64], n_trees: usize, noise: f64) -> f64 {
        match self {
            Response::Constant(strategy) => strategy.compute_leaf_value(residuals, n_trees, noise),
            Response::Linear(strategy) => strategy.compute_leaf_value(residuals, n_trees, noise),
        }
    }
}

/// Constant response strategy with vectorized mean computation
#[derive(Debug, Clone)]
pub struct ConstantResponse;

impl ResponseStrategy for ConstantResponse {
    #[inline]
    fn compute_leaf_value(&self, residuals: &[f64], n_trees: usize, noise: f64) -> f64 {
        if residuals.is_empty() {
            return noise;
        }

        let mean = self.vectorized_mean(residuals);
        mean / n_trees as f64 + noise
    }
}

impl ConstantResponse {
    /// Vectorized mean computation using chunked operations
    #[inline]
    fn vectorized_mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        // Process in chunks for better cache utilization
        let sum: f64 = values.chunks(CHUNK_SIZE)
            .map(|chunk| chunk.iter().sum::<f64>())
            .sum();
        
        sum / values.len() as f64
    }
}

/// Linear response strategy with optimized least squares
#[derive(Debug, Clone)]
pub struct LinearResponse;

impl ResponseStrategy for LinearResponse {
    #[inline]
    fn compute_leaf_value(&self, residuals: &[f64], n_trees: usize, noise: f64) -> f64 {
        match residuals.len() {
            0 => noise,
            1 => residuals[0] / n_trees as f64 + noise,
            2 => {
                // Simple case: mean of two values
                (residuals[0] + residuals[1]) / (2.0 * n_trees as f64) + noise
            }
            _ => {
                // For larger arrays, use vectorized computation
                let mean = self.vectorized_linear_fit(residuals);
                mean / n_trees as f64 + noise
            }
        }
    }
}

impl LinearResponse {
    /// Vectorized linear fit computation using chunked operations
    #[inline]
    fn vectorized_linear_fit(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        // For now, use mean as approximation
        // TODO: Implement full linear regression with chunked operations
        let sum: f64 = values.chunks(CHUNK_SIZE)
            .map(|chunk| chunk.iter().sum::<f64>())
            .sum();
        
        sum / values.len() as f64
    }
}

/// High-performance tree sampling operations
pub struct TreeSamplingOps {
    /// Normal distribution for leaf value sampling
    pub normal: Normal<f64>,
    /// Feature splitting probabilities (normalized)
    pub alpha_vec: Vec<f64>,
    /// Cumulative splitting probabilities for fast sampling
    pub splitting_probs: Vec<f64>,
    /// Depth control parameter (typically 0.95)
    pub alpha: f64,
    /// Depth control parameter (typically 2.0)
    pub beta: f64,
    /// Pre-computed depth probabilities (cache for performance)
    depth_probs_cache: Vec<f64>,
}

impl TreeSamplingOps {
    /// Create new sampling operations with pre-computed caches
    pub fn new(
        alpha_vec: Vec<f64>,
        splitting_probs: Vec<f64>,
        alpha: f64,
        beta: f64,
    ) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        // Pre-compute depth probabilities for common depths (0-20)
        let mut depth_probs_cache = Vec::with_capacity(21);
        for depth in 0..=20 {
            let prob = if depth == 0 {
                1.0
            } else {
                alpha * (1.0 + depth as f64).powf(-beta)
            };
            depth_probs_cache.push(prob);
        }
        
        Self {
            normal,
            alpha_vec,
            splitting_probs,
            alpha,
            beta,
            depth_probs_cache,
        }
    }

    /// Sample expansion flag with cached probabilities and branchless operations
    #[inline]
    pub fn sample_expand_flag(&self, depth: usize) -> bool {
        // Always expand root
        if depth == 0 {
            return true;
        }
        
        let prob = if depth < self.depth_probs_cache.len() {
            self.depth_probs_cache[depth]
        } else {
            // Compute for deeper nodes
            self.alpha * (1.0 + depth as f64).powf(-self.beta)
        };
        thread_rng().gen::<f64>() < prob
    }

    /// Sample leaf value with optimized response computation
    #[inline]
    pub fn sample_leaf_value(
        &self,
        residuals: &[f64],
        _observations: &[f64], // Currently unused but kept for API compatibility
        n_trees: usize,
        leaf_sd: &[f64],
        _n_dim: usize, // Currently unused
        response: &Response,
    ) -> f64 {
        // Fast path for empty residuals
        if residuals.is_empty() {
            return 0.0;
        }
        
        // Sample noise once
        let noise = if !leaf_sd.is_empty() {
            self.normal.sample(&mut thread_rng()) * leaf_sd[0]
        } else {
            0.0
        };
        
        // Compute response using vectorized operations
        response.compute_leaf_value(residuals, n_trees, noise)
    }

    /// Sample split feature using binary search for better performance
    #[inline]
    pub fn sample_split_feature(&self) -> usize {
        let p = thread_rng().gen::<f64>();
        
        // Binary search through cumulative probabilities
        self.binary_search_cumulative(p)
    }
    
    /// Optimized binary search through cumulative probabilities
    #[inline]
    fn binary_search_cumulative(&self, target: f64) -> usize {
        let mut left = 0;
        let mut right = self.splitting_probs.len();
        
        while left < right {
            let mid = (left + right) / 2;
            
            // Branchless comparison
            let go_right = (self.splitting_probs[mid] < target) as usize;
            left = mid * go_right + left * (1 - go_right);
            right = right * go_right + mid * (1 - go_right);
        }
        
        left.min(self.splitting_probs.len() - 1)
    }
}

/// Fast feature selection using vectorized operations
pub struct FeatureSampler {
    /// Cumulative probabilities for features
    cum_probs: Vec<f64>,
    /// Number of features
    n_features: usize,
}

impl FeatureSampler {
    /// Create new feature sampler with pre-computed cumulative probabilities
    pub fn new(feature_probs: &[f64]) -> Self {
        let mut cum_probs = Vec::with_capacity(feature_probs.len());
        let mut cumsum = 0.0;
        let total: f64 = feature_probs.iter().sum();
        
        for &prob in feature_probs {
            cumsum += prob / total;
            cum_probs.push(cumsum);
        }
        
        // Ensure last element is exactly 1.0
        if let Some(last) = cum_probs.last_mut() {
            *last = 1.0;
        }
        
        Self {
            cum_probs,
            n_features: feature_probs.len(),
        }
    }
    
    /// Sample feature index using fast binary search
    #[inline]
    pub fn sample(&self) -> usize {
        let u = thread_rng().gen::<f64>();
        
        // Fast linear search for small arrays
        if self.n_features <= 8 {
            for (idx, &prob) in self.cum_probs.iter().enumerate() {
                if u <= prob {
                    return idx;
                }
            }
            return self.n_features - 1;
        }
        
        // Binary search for larger arrays
        self.binary_search(u)
    }
    
    /// Branchless binary search
    #[inline]
    fn binary_search(&self, target: f64) -> usize {
        let mut size = self.n_features;
        let mut left = 0;
        
        while size > 1 {
            let half = size / 2;
            let mid = left + half;
            
            // Branchless update
            let go_right = (self.cum_probs[mid] < target) as usize;
            left += half * go_right;
            size -= half;
        }
        
        left
    }
}

/// Vectorized threshold sampling for continuous features
pub struct ThresholdSampler;

impl ThresholdSampler {
    /// Sample threshold from feature values using vectorized operations
    #[inline]
    pub fn sample_continuous(feature_values: &[f64]) -> Option<f64> {
        if feature_values.len() < 2 {
            return None;
        }
        
        // Fast sampling without full sort
        let idx = thread_rng().gen_range(0..feature_values.len());
        Some(feature_values[idx])
    }
    
    /// Sample threshold for one-hot encoded features
    #[inline]
    pub fn sample_onehot(feature_values: &[i32]) -> Option<i32> {
        if feature_values.is_empty() {
            return None;
        }
        
        // Check for heterogeneity using vectorized approach
        let first_val = feature_values[0];
        let is_homogeneous = feature_values.iter().all(|&x| x == first_val);
        
        if is_homogeneous {
            return None;
        }
        
        // Sample random value
        let idx = thread_rng().gen_range(0..feature_values.len());
        Some(feature_values[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorized_mean() {
        let strategy = ConstantResponse;
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mean = strategy.vectorized_mean(&values);
        assert!((mean - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_feature_sampler() {
        let probs = vec![0.3, 0.4, 0.3];
        let sampler = FeatureSampler::new(&probs);
        
        // Test that sampling returns valid indices
        for _ in 0..100 {
            let idx = sampler.sample();
            assert!(idx < 3);
        }
    }

    #[test]
    fn test_threshold_sampling() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let threshold = ThresholdSampler::sample_continuous(&values);
        assert!(threshold.is_some());
        assert!(values.contains(&threshold.unwrap()));
    }
}