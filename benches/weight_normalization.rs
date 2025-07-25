use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use pymc_bart::sampler::normalize_weights;

/// Generate random log weights in typical range
fn generate_log_weights(n: usize, rng: &mut SmallRng) -> Vec<f64> {
    (0..n)
        .map(|_| rng.random_range(-10.0..0.0)) // Typical log probability range
        .collect()
}

/// Generate uniform log weights
fn generate_uniform_log_weights(n: usize) -> Vec<f64> {
    vec![-2.0; n] // All weights equal in log space
}

/// Generate extremely skewed log weights (one dominant weight)
fn generate_skewed_log_weights(n: usize) -> Vec<f64> {
    let mut weights = vec![-100.0; n]; // Very small weights
    if n > 0 {
        weights[0] = 0.0; // One dominant weight
    }
    weights
}

/// Generate weights with extreme values for numerical stability testing
fn generate_extreme_log_weights(n: usize, rng: &mut SmallRng) -> Vec<f64> {
    (0..n)
        .map(|_| rng.random_range(-1000.0..-500.0)) // Very extreme log weights
        .collect()
}

/// Generate weights with mixed ranges
fn generate_mixed_range_log_weights(n: usize, rng: &mut SmallRng) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if i % 3 == 0 {
                rng.random_range(-1.0..0.0) // High weights
            } else if i % 3 == 1 {
                rng.random_range(-50.0..-10.0) // Medium weights
            } else {
                rng.random_range(-200.0..-100.0) // Low weights
            }
        })
        .collect()
}

/// Benchmark weight normalization with different vector sizes
fn benchmark_weight_normalization_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization_sizes");
    let mut rng = SmallRng::seed_from_u64(42);

    let sizes = vec![5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];

    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));

        let weights = generate_log_weights(n, &mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(n), &weights, |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        });
    }

    group.finish();
}

/// Benchmark weight normalization with different weight distributions
fn benchmark_weight_normalization_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization_distributions");
    let mut rng = SmallRng::seed_from_u64(42);

    let n = 100; // Fixed size for distribution comparison
    group.throughput(Throughput::Elements(n as u64));

    // Random log weights
    let random_weights = generate_log_weights(n, &mut rng);
    group.bench_with_input(
        BenchmarkId::new("random", "log_weights"),
        &random_weights,
        |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        },
    );

    // Uniform log weights
    let uniform_weights = generate_uniform_log_weights(n);
    group.bench_with_input(
        BenchmarkId::new("uniform", "log_weights"),
        &uniform_weights,
        |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        },
    );

    // Skewed log weights
    let skewed_weights = generate_skewed_log_weights(n);
    group.bench_with_input(
        BenchmarkId::new("skewed", "log_weights"),
        &skewed_weights,
        |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        },
    );

    // Extreme log weights
    let extreme_weights = generate_extreme_log_weights(n, &mut rng);
    group.bench_with_input(
        BenchmarkId::new("extreme", "log_weights"),
        &extreme_weights,
        |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        },
    );

    // Mixed range log weights
    let mixed_weights = generate_mixed_range_log_weights(n, &mut rng);
    group.bench_with_input(
        BenchmarkId::new("mixed_range", "log_weights"),
        &mixed_weights,
        |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        },
    );

    group.finish();
}

/// Benchmark iterator vs collected patterns
fn benchmark_weight_normalization_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization_memory");
    let mut rng = SmallRng::seed_from_u64(42);

    let sizes = vec![50, 200, 1000];

    for n in sizes {
        let weights = generate_log_weights(n, &mut rng);

        group.throughput(Throughput::Elements(n as u64));

        // Collect into Vec
        group.bench_with_input(
            BenchmarkId::new("collect_vec", n),
            &weights,
            |b, weights| {
                b.iter(|| {
                    let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                    black_box(normalized);
                });
            },
        );

        // Stream processing (sum for forcing evaluation)
        group.bench_with_input(BenchmarkId::new("streaming", n), &weights, |b, weights| {
            b.iter(|| {
                let sum: f64 = normalize_weights(black_box(weights)).sum();
                black_box(sum);
            });
        });

        // Take first few elements
        group.bench_with_input(
            BenchmarkId::new("partial_take", n),
            &weights,
            |b, weights| {
                b.iter(|| {
                    let first_five: Vec<f64> =
                        normalize_weights(black_box(weights)).take(5).collect();
                    black_box(first_five);
                });
            },
        );

        // Enumerate and collect
        group.bench_with_input(BenchmarkId::new("enumerate", n), &weights, |b, weights| {
            b.iter(|| {
                let enumerated: Vec<(usize, f64)> =
                    normalize_weights(black_box(weights)).enumerate().collect();
                black_box(enumerated);
            });
        });
    }

    group.finish();
}

/// Benchmark numerical stability edge cases
fn benchmark_weight_normalization_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization_stability");

    let n = 100;
    group.throughput(Throughput::Elements(n as u64));

    // All zeros (log probability of 1.0 for all)
    let zero_weights = vec![0.0; n];
    group.bench_with_input(
        BenchmarkId::new("stability", "all_zeros"),
        &zero_weights,
        |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        },
    );

    // Very negative weights (near underflow)
    let underflow_weights = vec![-700.0; n];
    group.bench_with_input(
        BenchmarkId::new("stability", "near_underflow"),
        &underflow_weights,
        |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        },
    );

    // One very large, others very small
    let mut dominant_weights = vec![-1000.0; n];
    dominant_weights[0] = 0.0;
    group.bench_with_input(
        BenchmarkId::new("stability", "one_dominant"),
        &dominant_weights,
        |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        },
    );

    // Wide range of values
    let wide_range_weights: Vec<f64> = (0..n)
        .map(|i| -(i as f64) * 10.0) // -0, -10, -20, -30, ...
        .collect();
    group.bench_with_input(
        BenchmarkId::new("stability", "wide_range"),
        &wide_range_weights,
        |b, weights| {
            b.iter(|| {
                let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                black_box(normalized);
            });
        },
    );

    group.finish();
}

/// Benchmark performance with repeated normalization
fn benchmark_weight_normalization_repeated(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization_repeated");
    let mut rng = SmallRng::seed_from_u64(42);

    let n = 50;
    let repetitions = vec![1, 5, 10, 20, 50];

    for reps in repetitions {
        let weights = generate_log_weights(n, &mut rng);

        group.throughput(Throughput::Elements((n * reps) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(reps),
            &(weights, reps),
            |b, (weights, reps)| {
                b.iter(|| {
                    for _ in 0..*reps {
                        let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                        black_box(normalized);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cache-friendly vs cache-unfriendly access patterns
fn benchmark_weight_normalization_cache_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization_cache");
    let mut rng = SmallRng::seed_from_u64(42);

    let sizes = vec![100, 1000, 10000];

    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));

        // Sequential access (cache-friendly)
        let sequential_weights = generate_log_weights(n, &mut rng);
        group.bench_with_input(
            BenchmarkId::new("sequential", n),
            &sequential_weights,
            |b, weights| {
                b.iter(|| {
                    let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                    black_box(normalized);
                });
            },
        );

        // Create interleaved pattern (less cache-friendly for some operations)
        let mut interleaved_weights = generate_log_weights(n, &mut rng);
        // Reverse every other pair to create some cache pressure
        for chunk in interleaved_weights.chunks_mut(4) {
            if chunk.len() >= 2 {
                chunk.swap(0, 1);
            }
        }

        group.bench_with_input(
            BenchmarkId::new("interleaved", n),
            &interleaved_weights,
            |b, weights| {
                b.iter(|| {
                    let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                    black_box(normalized);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark with realistic BART workload patterns
fn benchmark_weight_normalization_bart_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization_bart_realistic");
    let mut rng = SmallRng::seed_from_u64(42);

    // Typical BART particle counts
    let particle_counts = vec![5, 10, 20, 50, 100];

    for n_particles in particle_counts {
        group.throughput(Throughput::Elements(n_particles as u64));

        // Simulate realistic BART log-likelihood values
        let realistic_weights: Vec<f64> = (0..n_particles)
            .map(|_| {
                // BART log-likelihoods typically range from -100 to 0
                rng.random_range(-100.0..0.0)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("bart_realistic", n_particles),
            &realistic_weights,
            |b, weights| {
                b.iter(|| {
                    let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                    black_box(normalized);
                });
            },
        );

        // Simulate scenario where one particle is much better
        let mut better_particle_weights = realistic_weights.clone();
        if !better_particle_weights.is_empty() {
            better_particle_weights[0] = -1.0; // Much better likelihood
        }

        group.bench_with_input(
            BenchmarkId::new("bart_one_better", n_particles),
            &better_particle_weights,
            |b, weights| {
                b.iter(|| {
                    let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                    black_box(normalized);
                });
            },
        );

        // Simulate scenario where particles are very similar
        let similar_weights = vec![-50.0 + rng.random_range(-0.1..0.1); n_particles];

        group.bench_with_input(
            BenchmarkId::new("bart_similar", n_particles),
            &similar_weights,
            |b, weights| {
                b.iter(|| {
                    let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                    black_box(normalized);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    weight_normalization_benches,
    benchmark_weight_normalization_sizes,
    benchmark_weight_normalization_distributions,
    benchmark_weight_normalization_memory_patterns,
    benchmark_weight_normalization_stability,
    benchmark_weight_normalization_repeated,
    benchmark_weight_normalization_cache_patterns,
    benchmark_weight_normalization_bart_realistic,
);

criterion_main!(weight_normalization_benches);
