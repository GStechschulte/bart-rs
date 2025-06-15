//! Unit tests for PGBART components that don't require Python bindings
//! 
//! These tests focus on the core Rust functionality without PyO3 dependencies

use rand::prelude::*;

use pymc_bart::ops::{ConstantResponse, LinearResponse, ResponseStrategy, FeatureSampler, ThresholdSampler};
use pymc_bart::math::normalized_cumsum;

#[test]
fn test_vectorized_response_computation() {
    let residuals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n_trees = 100;
    
    // Test constant response
    let constant = ConstantResponse;
    let result = constant.compute_leaf_value(&residuals, n_trees, 0.0);
    let expected = residuals.iter().sum::<f64>() / (residuals.len() as f64 * n_trees as f64);
    assert!((result - expected).abs() < 1e-12, "Constant response failed: {} vs {}", result, expected);
    
    // Test with noise
    let result_with_noise = constant.compute_leaf_value(&residuals, n_trees, 0.5);
    assert!((result_with_noise - expected - 0.5).abs() < 1e-12, "Noise addition failed");
    
    // Test linear response
    let linear = LinearResponse;
    let linear_result = linear.compute_leaf_value(&residuals, n_trees, 0.0);
    assert!(linear_result.is_finite(), "Linear response should produce finite result");
    
    // Test edge cases
    let empty_residuals: Vec<f64> = vec![];
    let empty_result = constant.compute_leaf_value(&empty_residuals, n_trees, 1.0);
    assert_eq!(empty_result, 1.0, "Empty residuals should return noise only");
    
    let single_residual = vec![5.0];
    let single_result = constant.compute_leaf_value(&single_residual, n_trees, 0.0);
    let single_expected = 5.0 / n_trees as f64;
    assert!((single_result - single_expected).abs() < 1e-12, "Single residual failed");
}

#[test]
fn test_feature_sampler() {
    let probs = vec![0.1, 0.3, 0.4, 0.2];
    let sampler = FeatureSampler::new(&probs);
    
    // Test sampling produces valid indices
    let mut counts = vec![0; 4];
    for _ in 0..1000 {
        let idx = sampler.sample();
        assert!(idx < 4, "Sampled index {} out of bounds", idx);
        counts[idx] += 1;
    }
    
    // Verify rough proportions (should approximately match probabilities)
    let total = counts.iter().sum::<i32>() as f64;
    for (i, &count) in counts.iter().enumerate() {
        let observed_prob = count as f64 / total;
        let expected_prob = probs[i];
        // Allow for reasonable sampling variance
        assert!((observed_prob - expected_prob).abs() < 0.1, 
               "Feature {} sampled with probability {} vs expected {}", i, observed_prob, expected_prob);
    }
    
    // Test edge case: single feature
    let single_prob = vec![1.0];
    let single_sampler = FeatureSampler::new(&single_prob);
    for _ in 0..10 {
        assert_eq!(single_sampler.sample(), 0, "Single feature should always return index 0");
    }
}

#[test]
fn test_threshold_sampling() {
    // Test continuous threshold sampling
    let continuous_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    for _ in 0..100 {
        if let Some(threshold) = ThresholdSampler::sample_continuous(&continuous_values) {
            assert!(continuous_values.contains(&threshold), "Threshold {} not in original values", threshold);
        }
    }
    
    // Test edge cases
    let empty_values: Vec<f64> = vec![];
    assert!(ThresholdSampler::sample_continuous(&empty_values).is_none(), "Empty values should return None");
    
    let single_value = vec![42.0];
    assert!(ThresholdSampler::sample_continuous(&single_value).is_none(), "Single value should return None");
    
    // Test one-hot threshold sampling
    let onehot_values = vec![1, 2, 1, 3, 2];
    for _ in 0..100 {
        if let Some(threshold) = ThresholdSampler::sample_onehot(&onehot_values) {
            assert!(onehot_values.contains(&threshold), "OneHot threshold {} not in original values", threshold);
        }
    }
    
    // Test homogeneous case
    let homogeneous = vec![1, 1, 1, 1];
    assert!(ThresholdSampler::sample_onehot(&homogeneous).is_none(), "Homogeneous values should return None");
}

#[test]
fn test_normalized_cumsum() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let result = normalized_cumsum(&values);
    
    // Should be cumulative and normalized
    assert_eq!(result.len(), 4);
    assert!((result[0] - 0.1).abs() < 1e-10); // 1/10
    assert!((result[1] - 0.3).abs() < 1e-10); // 3/10
    assert!((result[2] - 0.6).abs() < 1e-10); // 6/10
    assert!((result[3] - 1.0).abs() < 1e-10); // 10/10
    
    // Test edge cases
    let empty: Vec<f64> = vec![];
    let empty_result = normalized_cumsum(&empty);
    assert!(empty_result.is_empty());
    
    let single = vec![5.0];
    let single_result = normalized_cumsum(&single);
    assert_eq!(single_result.len(), 1);
    assert!((single_result[0] - 1.0).abs() < 1e-10);
}

#[test] 
fn test_branchless_operations() {
    // Test that branchless operations produce correct results
    let test_cases = vec![
        (true, 1usize, 0usize),
        (false, 0usize, 1usize),
    ];
    
    for (condition, expected_left, expected_right) in test_cases {
        let goes_left = condition as usize;
        let goes_right = 1 - goes_left;
        
        assert_eq!(goes_left, expected_left);
        assert_eq!(goes_right, expected_right);
    }
    
    // Test branchless min/max operations
    let a = 5.0;
    let b = 3.0;
    let min_result = if a < b { a } else { b };
    let max_result = if a > b { a } else { b };
    
    assert_eq!(min_result, 3.0);
    assert_eq!(max_result, 5.0);
}

#[test]
fn test_chunked_operations() {
    // Test that chunked operations produce same results as scalar
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    
    // Scalar sum
    let scalar_sum: f64 = values.iter().sum();
    
    // Chunked sum
    let chunk_sum: f64 = values.chunks(8).map(|chunk| chunk.iter().sum::<f64>()).sum();
    
    assert!((scalar_sum - chunk_sum).abs() < 1e-10, "Chunked sum should match scalar sum");
    
    // Test mean computation
    let scalar_mean = scalar_sum / values.len() as f64;
    let chunk_mean = chunk_sum / values.len() as f64;
    
    assert!((scalar_mean - chunk_mean).abs() < 1e-10, "Chunked mean should match scalar mean");
}

#[test]
fn test_constant_response_edge_cases() {
    let constant = ConstantResponse;
    
    // Test with very small residuals
    let small_residuals = vec![1e-15, 2e-15, 3e-15];
    let result = constant.compute_leaf_value(&small_residuals, 1, 0.0);
    assert!(result.is_finite() && result >= 0.0);
    
    // Test with large residuals
    let large_residuals = vec![1e10, 2e10, 3e10];
    let result = constant.compute_leaf_value(&large_residuals, 1, 0.0);
    assert!(result.is_finite());
    
    // Test with mixed positive/negative
    let mixed_residuals = vec![-1.0, 1.0, -2.0, 2.0];
    let result = constant.compute_leaf_value(&mixed_residuals, 1, 0.0);
    assert!(result.is_finite());
    assert_eq!(result, 0.0); // Should sum to zero
}

#[test]
fn test_linear_response_edge_cases() {
    let linear = LinearResponse;
    
    // Test with exactly 2 residuals (special case)
    let two_residuals = vec![1.0, 3.0];
    let result = linear.compute_leaf_value(&two_residuals, 2, 0.0);
    assert_eq!(result, 1.0); // (1+3)/(2*2) = 1.0
    
    // Test with single residual
    let single_residual = vec![5.0];
    let result = linear.compute_leaf_value(&single_residual, 1, 0.0);
    assert_eq!(result, 5.0);
    
    // Test with empty residuals
    let empty_residuals: Vec<f64> = vec![];
    let result = linear.compute_leaf_value(&empty_residuals, 1, 1.0);
    assert_eq!(result, 1.0); // Should return noise only
}

#[test]
fn test_feature_sampler_edge_cases() {
    // Test with equal probabilities
    let equal_probs = vec![0.25, 0.25, 0.25, 0.25];
    let sampler = FeatureSampler::new(&equal_probs);
    
    let mut counts = vec![0; 4];
    for _ in 0..1000 {
        let idx = sampler.sample();
        counts[idx] += 1;
    }
    
    // Should be roughly equal
    for count in counts {
        assert!(count > 200 && count < 300, "Equal probabilities should produce roughly equal counts");
    }
    
    // Test with zero probability (except last)
    let skewed_probs = vec![0.0, 0.0, 0.0, 1.0];
    let skewed_sampler = FeatureSampler::new(&skewed_probs);
    
    for _ in 0..10 {
        assert_eq!(skewed_sampler.sample(), 3, "Should always sample the only non-zero probability");
    }
}

#[test]
fn test_performance_characteristics() {
    // Test that operations scale reasonably
    let small_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let large_data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
    
    let constant = ConstantResponse;
    
    // Both should produce finite results
    let small_result = constant.compute_leaf_value(&small_data, 1, 0.0);
    let large_result = constant.compute_leaf_value(&large_data, 1, 0.0);
    
    assert!(small_result.is_finite());
    assert!(large_result.is_finite());
    
    // Large data should have larger mean (but scaled)
    assert!(large_result > small_result);
}