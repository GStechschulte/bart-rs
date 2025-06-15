//! Comprehensive tests for performance-optimized PGBART components
//! 
//! This test suite verifies correctness of optimized implementations:
//! - SIMD operations produce correct results
//! - Vectorized algorithms match reference implementations
//! - Memory layouts are cache-friendly
//! - Branchless operations work correctly

use ndarray::{Array1, Array2};
use rand::prelude::*;
use std::f64;

use pymc_bart::forest::{DecisionTree, Forest, Predict, ParticleArrays};
use pymc_bart::ops::{
    ConstantResponse, LinearResponse, ResponseStrategy, TreeSamplingOps, 
    FeatureSampler, ThresholdSampler
};

/// Generate test data for validation
fn generate_test_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    
    let mut X = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        for j in 0..n_features {
            X[[i, j]] = rng.gen_range(-5.0..5.0);
        }
        y[i] = X[[i, 0]] * 2.0 + X[[i, 1]] * 0.5 + rng.gen_range(-0.1..0.1);
    }
    
    (X, y)
}

#[test]
fn test_particle_arrays_simd_weight_normalization() {
    let n_particles = 64;
    let mut particle_arrays = ParticleArrays::new(n_particles);
    
    // Set up test weights (log-space)
    let mut rng = StdRng::seed_from_u64(123);
    for i in 0..n_particles {
        particle_arrays.weights[i] = rng.gen_range(-10.0..0.0);
    }
    
    let original_weights = particle_arrays.weights.clone();
    
    // Normalize using SIMD implementation
    particle_arrays.normalize_weights(n_particles);
    
    // Verify normalization
    let sum: f64 = particle_arrays.weights[..n_particles].iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "Weights should sum to 1.0, got {}", sum);
    
    // Verify all weights are positive
    for i in 0..n_particles {
        assert!(particle_arrays.weights[i] >= 0.0, "Weight {} is negative: {}", i, particle_arrays.weights[i]);
        assert!(particle_arrays.weights[i] <= 1.0, "Weight {} exceeds 1.0: {}", i, particle_arrays.weights[i]);
    }
    
    // Test numerical stability - verify log-sum-exp trick works
    let max_log = original_weights[..n_particles].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let manual_sum: f64 = original_weights[..n_particles]
        .iter()
        .map(|&w| (w - max_log).exp())
        .sum();
    
    for i in 0..n_particles {
        let expected = (original_weights[i] - max_log).exp() / manual_sum;
        assert!((particle_arrays.weights[i] - expected).abs() < 1e-12, 
               "SIMD normalization mismatch at index {}: {} vs {}", i, particle_arrays.weights[i], expected);
    }
}

#[test]
fn test_systematic_resampling_correctness() {
    let n_particles = 32;
    let mut particle_arrays = ParticleArrays::new(n_particles);
    
    // Set up normalized weights with known distribution
    let weights = vec![0.1, 0.2, 0.3, 0.4]; // Should favor later indices
    for i in 0..4 {
        particle_arrays.weights[i] = weights[i];
    }
    
    // Pad remaining weights with zeros
    for i in 4..n_particles {
        particle_arrays.weights[i] = 0.0;
    }
    
    // Perform systematic resampling
    particle_arrays.systematic_resample(4);
    
    // Verify all indices are valid
    for i in 0..4 {
        assert!(particle_arrays.indices[i] < 4, "Resampled index {} is out of bounds", particle_arrays.indices[i]);
    }
    
    // Test with uniform weights
    for i in 0..4 {
        particle_arrays.weights[i] = 0.25;
    }
    
    particle_arrays.systematic_resample(4);
    
    // All indices should be valid
    for i in 0..4 {
        assert!(particle_arrays.indices[i] < 4, "Uniform resampling produced invalid index");
    }
}

#[test]
fn test_decision_tree_vectorized_prediction() {
    let (X, _) = generate_test_data(100, 5);
    let mut tree = DecisionTree::new(1.0, 100);
    
    // Build a simple tree manually
    // Root -> split on feature 0 at threshold 0.0
    let _ = tree.split_node(0, 0, 0.0, -1.0, 1.0, &X);
    
    // Predict
    let predictions = tree.predict(&X);
    
    // Verify predictions
    assert_eq!(predictions.len(), 100);
    
    for (i, &pred) in predictions.iter().enumerate() {
        let expected = if X[[i, 0]] <= 0.0 { -1.0 } else { 1.0 };
        assert_eq!(pred, expected, "Prediction mismatch at sample {}: {} vs {}", i, pred, expected);
    }
}

#[test]
fn test_decision_tree_branchless_traversal() {
    let (X, _) = generate_test_data(50, 3);
    let mut tree = DecisionTree::new(0.0, 50);
    
    // Create a deeper tree
    let _ = tree.split_node(0, 0, 0.0, 0.0, 0.0, &X); // Split root
    let _ = tree.split_node(1, 1, 1.0, -2.0, -1.0, &X); // Split left child
    let _ = tree.split_node(2, 1, -1.0, 1.0, 2.0, &X); // Split right child
    
    let predictions = tree.predict(&X);
    
    // Verify all predictions are reasonable
    for &pred in &predictions {
        assert!(pred >= -2.0 && pred <= 2.0, "Prediction {} out of expected range", pred);
    }
    
    // Test specific samples
    for i in 0..10 {
        let pred = tree.predict_single_branchless(&X, i);
        assert_eq!(pred, predictions[i], "Branchless prediction mismatch at sample {}", i);
    }
}

#[test]
fn test_vectorized_response_computation() {
    let residuals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // Length divisible by SIMD width
    let n_trees = 100;
    
    // Test constant response
    let constant = ConstantResponse;
    let result = constant.compute_leaf_value(&residuals, n_trees, 0.0);
    let expected = residuals.iter().sum::<f64>() / (residuals.len() as f64 * n_trees as f64);
    assert!((result - expected).abs() < 1e-12, "Constant response vectorization failed: {} vs {}", result, expected);
    
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
    assert!((single_result - single_expected).abs() < 1e-12, "Single residual failed: {} vs {}", single_result, single_expected);
}

#[test]
fn test_feature_sampler_correctness() {
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
fn test_tree_sampling_ops_optimization() {
    let alpha_vec = vec![0.3, 0.4, 0.3];
    let splitting_probs = vec![0.3, 0.7, 1.0]; // Cumulative
    let ops = TreeSamplingOps::new(alpha_vec, splitting_probs, 0.95, 2.0);
    
    // Test cached depth probabilities
    assert!(ops.sample_expand_flag(0), "Root should always be expandable");
    
    // Test multiple depths
    let mut expand_counts = vec![0; 10];
    for _ in 0..1000 {
        for depth in 1..10 {
            if ops.sample_expand_flag(depth) {
                expand_counts[depth] += 1;
            }
        }
    }
    
    // Deeper nodes should be less likely to expand
    for depth in 1..9 {
        assert!(expand_counts[depth] >= expand_counts[depth + 1], 
               "Expansion probability should decrease with depth: depth {} = {}, depth {} = {}", 
               depth, expand_counts[depth], depth + 1, expand_counts[depth + 1]);
    }
    
    // Test feature sampling
    let mut feature_counts = vec![0; 3];
    for _ in 0..1000 {
        let feature = ops.sample_split_feature();
        assert!(feature < 3, "Feature index {} out of bounds", feature);
        feature_counts[feature] += 1;
    }
    
    // Should roughly match probabilities
    let total = feature_counts.iter().sum::<i32>() as f64;
    assert!(feature_counts[1] as f64 / total > 0.3, "Middle feature should be most common");
}

#[test]
fn test_forest_operations_integration() {
    let n_particles = 16;
    let n_samples = 50;
    let (X, y) = generate_test_data(n_samples, 5);
    let initial_value = y.mean().unwrap() / 10.0;
    
    let mut forest = Forest::new(n_particles, n_samples, initial_value, 100);
    
    // Initialize with realistic weights
    let mut rng = StdRng::seed_from_u64(456);
    for i in 0..n_particles {
        forest.particle_arrays.weights[i] = rng.gen_range(-5.0..0.0);
    }
    
    // Test weight normalization
    forest.normalize_weights();
    let sum: f64 = forest.weights[..n_particles].iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "Forest weights should sum to 1.0");
    
    // Test resampling doesn't crash
    forest.resample();
    
    // Test prediction
    for tree in &forest.trees {
        let predictions = tree.predict(&X);
        assert_eq!(predictions.len(), n_samples);
        for &pred in &predictions {
            assert!(pred.is_finite(), "All predictions should be finite");
        }
    }
    
    // Test has_expandable_nodes
    let has_expandable = forest.has_expandable_nodes();
    assert!(has_expandable, "New forest should have expandable nodes");
}

#[test]
fn test_memory_layout_efficiency() {
    let n_particles = 128;
    let particle_arrays = ParticleArrays::new(n_particles);
    
    // Verify SIMD alignment
    assert_eq!(particle_arrays.weights.len() % 4, 0, "Weights should be SIMD-aligned");
    assert_eq!(particle_arrays.temp_weights.len() % 4, 0, "Temp weights should be SIMD-aligned");
    
    // Test capacity pre-allocation
    assert!(particle_arrays.weights.capacity() >= n_particles, "Weights should be pre-allocated");
    assert!(particle_arrays.cdf_buffer.capacity() >= n_particles, "CDF buffer should be pre-allocated");
    
    // Test decision tree memory layout
    let tree = DecisionTree::new(1.0, 100);
    assert!(tree.feature.capacity() > 0, "Feature array should be pre-allocated");
    assert!(tree.threshold.capacity() > 0, "Threshold array should be pre-allocated");
    assert!(tree.value.capacity() > 0, "Value array should be pre-allocated");
}

#[test]
fn test_numerical_stability() {
    let n_particles = 64;
    let mut particle_arrays = ParticleArrays::new(n_particles);
    
    // Test with extreme weights
    for i in 0..n_particles {
        particle_arrays.weights[i] = if i % 2 == 0 { -1000.0 } else { -1001.0 };
    }
    
    particle_arrays.normalize_weights(n_particles);
    
    // Should not produce NaN or Inf
    for i in 0..n_particles {
        assert!(particle_arrays.weights[i].is_finite(), "Weight {} should be finite", i);
        assert!(particle_arrays.weights[i] >= 0.0, "Weight {} should be non-negative", i);
    }
    
    let sum: f64 = particle_arrays.weights[..n_particles].iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "Extreme weights should still normalize properly");
    
    // Test with very small differences
    for i in 0..n_particles {
        particle_arrays.weights[i] = -100.0 + (i as f64) * 1e-10;
    }
    
    particle_arrays.normalize_weights(n_particles);
    
    let sum2: f64 = particle_arrays.weights[..n_particles].iter().sum();
    assert!((sum2 - 1.0).abs() < 1e-9, "Small weight differences should be handled correctly");
}

#[test]
fn test_concurrent_safety() {
    use std::sync::Arc;
    use std::thread;
    
    let n_particles = 32;
    let n_samples = 100;
    let (X, y) = generate_test_data(n_samples, 3);
    let initial_value = y.mean().unwrap() / 10.0;
    
    // Test that forest creation is deterministic
    let mut forests = Vec::new();
    for _ in 0..4 {
        forests.push(Forest::new(n_particles, n_samples, initial_value, 50));
    }
    
    // All forests should have the same initial structure
    for i in 1..forests.len() {
        assert_eq!(forests[0].trees.len(), forests[i].trees.len());
        assert_eq!(forests[0].weights.len(), forests[i].weights.len());
    }
    
    // Test that operations are thread-safe (no data races)
    let forest = Arc::new(forests.into_iter().next().unwrap());
    let X = Arc::new(X);
    
    let handles: Vec<_> = (0..4).map(|_| {
        let forest_clone = Arc::clone(&forest);
        let X_clone = Arc::clone(&X);
        thread::spawn(move || {
            // Read-only operations should be safe
            for tree in &forest_clone.trees {
                let _ = tree.predict(&X_clone);
            }
        })
    }).collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_performance_regression_guards() {
    // These tests ensure that optimizations don't break correctness
    
    let n_particles = 64;
    let mut arrays = ParticleArrays::new(n_particles);
    
    // Set up known weights
    let known_weights: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
    for (i, &weight) in known_weights.iter().enumerate() {
        arrays.weights[i] = weight.ln(); // Log space
    }
    
    // Fill remaining with very small values
    for i in 4..n_particles {
        arrays.weights[i] = (-1000.0f64).ln();
    }
    
    arrays.normalize_weights(n_particles);
    
    // First 4 should dominate
    let dominant_sum: f64 = arrays.weights[..4].iter().sum();
    assert!(dominant_sum > 0.99, "Dominant weights should sum to nearly 1.0, got {}", dominant_sum);
    
    // Test that SIMD and scalar versions produce identical results
    let mut scalar_arrays = ParticleArrays::new(4); // Small enough to force scalar path
    for i in 0..4 {
        scalar_arrays.weights[i] = known_weights[i].ln();
    }
    scalar_arrays.normalize_weights(4);
    
    // Compare normalized results
    for i in 0..4 {
        assert!((arrays.weights[i] - scalar_arrays.weights[i]).abs() < 1e-12,
               "SIMD and scalar results should match: {} vs {}", arrays.weights[i], scalar_arrays.weights[i]);
    }
}