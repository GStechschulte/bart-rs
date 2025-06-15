//! Performance benchmarks for PGBART critical operations
//!
//! This benchmark suite focuses on the performance-critical code paths:
//! 1. Tree traversal and prediction
//! 2. Growing particles (tree expansion)
//! 3. Weight normalization
//! 4. Particle resampling
//! 5. Vectorized operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use std::time::Duration;

use pymc_bart::forest::{DecisionTree, Forest, ParticleArrays, Predict};
use pymc_bart::math::normalized_cumsum;
use pymc_bart::ops::{
    ConstantResponse, FeatureSampler, LinearResponse, ResponseStrategy, ThresholdSampler,
};

/// Generate synthetic dataset for benchmarking
fn generate_synthetic_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = thread_rng();

    let mut X = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);

    // Generate random features
    for i in 0..n_samples {
        for j in 0..n_features {
            X[[i, j]] = rng.gen_range(-10.0..10.0);
        }
        // Simple nonlinear relationship
        y[i] = (X[[i, 0]] as f64).powi(2) + 0.5 * X[[i, 1]] + rng.gen_range(-1.0..1.0);
    }

    (X, y)
}

/// Generate random tree for benchmarking
fn generate_random_tree(n_samples: usize, depth: usize) -> DecisionTree {
    let mut tree = DecisionTree::new(0.0, n_samples);
    let mut rng = thread_rng();

    // Build a complete binary tree to specified depth
    let max_nodes = (1 << (depth + 1)) - 1;

    for node_idx in 0..max_nodes {
        if tree.node_depth(node_idx) < depth && tree.is_leaf(node_idx) {
            let feature_idx = rng.gen_range(0..10) as u16;
            let threshold = rng.gen_range(-5.0..5.0) as f32;
            let left_val = rng.gen_range(-2.0..2.0) as f32;
            let right_val = rng.gen_range(-2.0..2.0) as f32;

            // Create dummy X for splitting
            let X = Array2::from_elem((n_samples, 10), 1.0);
            let _ = tree.split_node(node_idx, feature_idx, threshold, left_val, right_val, &X);
        }
    }

    tree
}

/// Benchmark tree prediction performance
fn bench_tree_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_prediction");
    group.measurement_time(Duration::from_secs(10));

    for &n_samples in &[100, 1000, 10000] {
        for &depth in &[3, 5, 7] {
            let (X, _) = generate_synthetic_data(n_samples, 10);
            let tree = generate_random_tree(n_samples, depth);

            group.bench_with_input(
                BenchmarkId::new("vectorized", format!("{}samples_{}depth", n_samples, depth)),
                &(&X, &tree),
                |b, (X, tree)| b.iter(|| black_box(tree.predict(X))),
            );
        }
    }

    group.finish();
}

/// Benchmark weight normalization using SIMD
fn bench_weight_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization");
    group.measurement_time(Duration::from_secs(5));

    for &n_particles in &[32, 64, 128, 256, 512] {
        let mut particle_arrays = ParticleArrays::new(n_particles);

        // Initialize with random weights
        let mut rng = thread_rng();
        for i in 0..n_particles {
            particle_arrays.weights[i] = rng.gen_range(-10.0..0.0); // Log weights
        }

        group.bench_with_input(
            BenchmarkId::new("simd", n_particles),
            &n_particles,
            |b, &n_particles| {
                b.iter(|| {
                    let mut arrays = particle_arrays.clone();
                    black_box(arrays.normalize_weights(n_particles))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark systematic resampling
fn bench_systematic_resampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("systematic_resampling");
    group.measurement_time(Duration::from_secs(5));

    for &n_particles in &[32, 64, 128, 256, 512] {
        let mut particle_arrays = ParticleArrays::new(n_particles);

        // Initialize with normalized weights
        let mut rng = thread_rng();
        let mut total = 0.0;
        for i in 0..n_particles {
            particle_arrays.weights[i] = rng.gen::<f64>();
            total += particle_arrays.weights[i];
        }

        // Normalize
        for i in 0..n_particles {
            particle_arrays.weights[i] /= total;
        }

        group.bench_with_input(
            BenchmarkId::new("optimized", n_particles),
            &n_particles,
            |b, &n_particles| {
                b.iter(|| {
                    let mut arrays = particle_arrays.clone();
                    black_box(arrays.systematic_resample(n_particles))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark vectorized response computation
fn bench_response_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("response_computation");

    for &n_residuals in &[10, 50, 100, 500] {
        let mut rng = thread_rng();
        let residuals: Vec<f64> = (0..n_residuals).map(|_| rng.gen_range(-5.0..5.0)).collect();

        let constant_response = ConstantResponse;
        let linear_response = LinearResponse;

        group.bench_with_input(
            BenchmarkId::new("constant_vectorized", n_residuals),
            &residuals,
            |b, residuals| {
                b.iter(|| black_box(constant_response.compute_leaf_value(residuals, 100, 0.1)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("linear_vectorized", n_residuals),
            &residuals,
            |b, residuals| {
                b.iter(|| black_box(linear_response.compute_leaf_value(residuals, 100, 0.1)))
            },
        );
    }

    group.finish();
}

/// Benchmark feature sampling with binary search
fn bench_feature_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_sampling");

    for &n_features in &[5, 10, 50, 100] {
        let mut rng = thread_rng();
        let probs: Vec<f64> = (0..n_features).map(|_| rng.gen::<f64>()).collect();
        let sampler = FeatureSampler::new(&probs);

        group.bench_with_input(
            BenchmarkId::new("binary_search", n_features),
            &sampler,
            |b, sampler| b.iter(|| black_box(sampler.sample())),
        );
    }

    group.finish();
}

/// Benchmark threshold sampling
fn bench_threshold_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_sampling");

    for &n_values in &[10, 50, 100, 500] {
        let mut rng = thread_rng();
        let continuous_values: Vec<f64> =
            (0..n_values).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let onehot_values: Vec<i32> = (0..n_values).map(|_| rng.gen_range(0..10)).collect();

        group.bench_with_input(
            BenchmarkId::new("continuous", n_values),
            &continuous_values,
            |b, values| b.iter(|| black_box(ThresholdSampler::sample_continuous(values))),
        );

        group.bench_with_input(
            BenchmarkId::new("onehot", n_values),
            &onehot_values,
            |b, values| b.iter(|| black_box(ThresholdSampler::sample_onehot(values))),
        );
    }

    group.finish();
}

/// Benchmark forest operations (tree growing simulation)
fn bench_forest_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_operations");
    group.measurement_time(Duration::from_secs(15));

    for &n_particles in &[16, 32, 64] {
        for &n_samples in &[100, 1000] {
            let (X, y) = generate_synthetic_data(n_samples, 10);
            let initial_value = y.mean().unwrap() / 100.0;
            let mut forest = Forest::new(n_particles, n_samples, initial_value, 100);

            // Initialize weights
            let mut rng = thread_rng();
            for i in 0..n_particles {
                forest.particle_arrays.weights[i] = rng.gen_range(-5.0..0.0);
            }

            group.bench_with_input(
                BenchmarkId::new(
                    "normalize_and_resample",
                    format!("{}p_{}s", n_particles, n_samples),
                ),
                &(n_particles, n_samples, initial_value),
                |b, (n_particles, n_samples, initial_value)| {
                    b.iter(|| {
                        let mut f = Forest::new(*n_particles, *n_samples, *initial_value, 100);
                        // Initialize with random weights
                        let mut rng = thread_rng();
                        for i in 0..*n_particles {
                            f.particle_arrays.weights[i] = rng.gen_range(-5.0..0.0);
                        }
                        f.normalize_weights();
                        black_box(f.resample())
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark normalized cumulative sum computation
fn bench_normalized_cumsum(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalized_cumsum");

    for &n_elements in &[10, 50, 100, 500] {
        let mut rng = thread_rng();
        let values: Vec<f64> = (0..n_elements).map(|_| rng.gen::<f64>()).collect();

        group.bench_with_input(
            BenchmarkId::new("standard", n_elements),
            &values,
            |b, values| b.iter(|| black_box(normalized_cumsum(values))),
        );
    }

    group.finish();
}

/// Benchmark complete PGBART step simulation
fn bench_pgbart_step_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pgbart_step_simulation");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10); // Fewer samples for expensive operations

    for &n_particles in &[16, 32] {
        for &n_samples in &[100, 500] {
            let (X, y) = generate_synthetic_data(n_samples, 5);
            let initial_value = y.mean().unwrap() / 100.0;
            let mut forest = Forest::new(n_particles, n_samples, initial_value, 100);

            // Setup realistic weights
            let mut rng = thread_rng();
            for i in 0..n_particles {
                forest.weights[i] = rng.gen_range(-3.0..0.0);
                forest.particle_arrays.weights[i] = forest.weights[i];
            }

            group.bench_with_input(
                BenchmarkId::new("full_cycle", format!("{}p_{}s", n_particles, n_samples)),
                &(&X, n_particles, n_samples, initial_value),
                |b, (X, n_particles, n_samples, initial_value)| {
                    b.iter(|| {
                        let mut f = Forest::new(*n_particles, *n_samples, *initial_value, 100);
                        let mut rng = thread_rng();
                        
                        // Simulate a complete particle filtering step
                        // 1. Weight update (simulated)
                        for i in 0..*n_particles {
                            f.particle_arrays.weights[i] = rng.gen_range(-3.0..0.0);
                        }

                        // 2. Normalize weights
                        f.normalize_weights();

                        // 3. Resample
                        f.resample();

                        // 4. Predict (simulation of tree evaluation)
                        for tree in &f.trees {
                            black_box(tree.predict(X));
                        }
                    })
                },
            );
        }
    }

    group.finish();
}

/// Memory allocation benchmark
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    for &n_particles in &[32, 64, 128] {
        for &n_samples in &[100, 1000] {
            group.bench_with_input(
                BenchmarkId::new(
                    "forest_creation",
                    format!("{}p_{}s", n_particles, n_samples),
                ),
                &(n_particles, n_samples),
                |b, &(n_particles, n_samples)| {
                    b.iter(|| black_box(Forest::new(n_particles, n_samples, 0.0, 100)))
                },
            );

            group.bench_with_input(
                BenchmarkId::new("particle_arrays", n_particles),
                &n_particles,
                |b, &n_particles| b.iter(|| black_box(ParticleArrays::new(n_particles))),
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tree_prediction,
    bench_weight_normalization,
    bench_systematic_resampling,
    bench_response_computation,
    bench_feature_sampling,
    bench_threshold_sampling,
    bench_forest_operations,
    bench_normalized_cumsum,
    bench_pgbart_step_simulation,
    bench_memory_allocation
);

criterion_main!(benches);
