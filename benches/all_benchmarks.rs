//! Comprehensive benchmark suite for the BART library
//!
//! This module runs all benchmarks for the major components:
//! - SMC algorithm (ParticleGibbsSampler)
//! - Resampling strategies
//! - Split rules and splitting strategies
//! - Weight normalization
//!
//! Run with: `cargo bench --bench all_benchmarks`

use std::hint::black_box;
use std::rc::Rc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numpy::ndarray::{Array1, Array2};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use pymc_bart::{
    base::BartState,
    particle::{Particle, Tree},
    resampling::{ResamplingStrategy, SystematicResampling},
    sampler::{normalize_weights, ParticleGibbsSampler},
    splitting::{ContinuousSplit, OneHotSplit, SplitRule, SplitRules},
    update::{BARTContext, BARTWeighter, Moves},
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Generate synthetic regression data for benchmarking
fn generate_synthetic_data(
    n_samples: usize,
    n_features: usize,
    rng: &mut SmallRng,
) -> (Array2<f64>, Array1<f64>) {
    let x_data: Vec<f64> = (0..n_samples * n_features)
        .map(|_| rng.random_range(-5.0..5.0))
        .collect();
    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();

    let y_data: Vec<f64> = (0..n_samples)
        .map(|i| {
            let linear_component: f64 = x.row(i).iter().sum::<f64>() * 0.1;
            let noise = rng.random_range(-0.5..0.5);
            linear_component + noise
        })
        .collect();
    let y = Array1::from_vec(y_data);

    (x, y)
}

/// Create initial particles for benchmarking
fn create_initial_particles<const MAX_NODES: usize>(
    n_particles: usize,
    init_leaf_value: f64,
    n_samples: usize,
) -> Vec<Particle<MAX_NODES>> {
    (0..n_particles)
        .map(|_| Rc::new(Tree::new(init_leaf_value, n_samples)))
        .collect()
}

/// Create BART context for benchmarking
fn create_bart_context(x_data: Array2<f64>) -> BARTContext {
    BARTContext {
        x_data,
        alpha: 0.95,
        beta: 2.0,
        sigma: 1.0,
        min_samples_leaf: 2,
        max_depth: 10,
    }
}

/// Generate normalized weights for resampling
fn generate_normalized_weights(n: usize, rng: &mut SmallRng) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
    let sum: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= sum);
    weights
}

/// Generate log weights for normalization
fn generate_log_weights(n: usize, rng: &mut SmallRng) -> Vec<f64> {
    (0..n).map(|_| rng.random_range(-10.0..0.0)).collect()
}

// ============================================================================
// SMC Algorithm Benchmarks
// ============================================================================

fn benchmark_smc_comprehensive(c: &mut Criterion) {
    const MAX_NODES: usize = 511;
    let mut group = c.benchmark_group("smc_comprehensive");

    let scenarios = vec![
        ("small", 50, 2, 5),    // small dataset, few particles
        ("medium", 100, 5, 10), // medium dataset, moderate particles
        ("large", 200, 10, 20), // large dataset, many particles
    ];

    for (name, n_samples, n_features, n_particles) in scenarios {
        group.throughput(Throughput::Elements(n_particles as u64));

        group.bench_with_input(
            BenchmarkId::new("full_step", name),
            &(n_samples, n_features, n_particles),
            |b, &(n_samples, n_features, n_particles)| {
                b.iter_batched(
                    || {
                        let mut setup_rng = SmallRng::seed_from_u64(42);
                        let (x_data, y_data) =
                            generate_synthetic_data(n_samples, n_features, &mut setup_rng);
                        let init_leaf_value = y_data.mean().unwrap();

                        let initial_particles = create_initial_particles::<MAX_NODES>(
                            n_particles,
                            init_leaf_value,
                            n_samples,
                        );

                        let init_weights = vec![1.0 / n_particles as f64; n_particles];
                        let initial_state = BartState::new(initial_particles, init_weights);
                        let context = create_bart_context(x_data);

                        let sampler = ParticleGibbsSampler::new(
                            Moves::new(),
                            BARTWeighter,
                            SystematicResampling,
                        );

                        let bench_rng = SmallRng::seed_from_u64(42);

                        (sampler, initial_state, context, bench_rng)
                    },
                    |(sampler, initial_state, context, mut rng)| {
                        let result =
                            sampler.step(&mut rng, black_box(initial_state), black_box(&context));
                        black_box(result);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Resampling Benchmarks
// ============================================================================

fn benchmark_resampling_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampling_comprehensive");
    let mut rng = SmallRng::seed_from_u64(42);

    let particle_counts = vec![10, 50, 100, 500, 1000];

    for n_particles in particle_counts {
        group.throughput(Throughput::Elements(n_particles as u64));

        // Test systematic resampling with different weight patterns
        let uniform_weights = vec![1.0 / n_particles as f64; n_particles];
        let random_weights = generate_normalized_weights(n_particles, &mut rng);

        // Uniform weights
        group.bench_with_input(
            BenchmarkId::new("systematic_uniform", n_particles),
            &uniform_weights,
            |b, weights| {
                b.iter_batched(
                    || SmallRng::seed_from_u64(42),
                    |mut batch_rng| {
                        let indices: Vec<usize> = SystematicResampling::resample(
                            &mut batch_rng,
                            black_box(weights.iter().copied()),
                        )
                        .collect();
                        black_box(indices);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Random weights
        group.bench_with_input(
            BenchmarkId::new("systematic_random", n_particles),
            &random_weights,
            |b, weights| {
                b.iter_batched(
                    || SmallRng::seed_from_u64(42),
                    |mut batch_rng| {
                        let indices: Vec<usize> = SystematicResampling::resample(
                            &mut batch_rng,
                            black_box(weights.iter().copied()),
                        )
                        .collect();
                        black_box(indices);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Split Rules Benchmarks
// ============================================================================

fn benchmark_split_rules_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_rules_comprehensive");
    let mut rng = SmallRng::seed_from_u64(42);

    let data_scenarios = vec![("small", 100, 3), ("medium", 500, 5), ("large", 1000, 10)];

    for (name, n_samples, n_features) in data_scenarios {
        let continuous_data = {
            let data: Vec<f64> = (0..n_samples * n_features)
                .map(|_| rng.random_range(-10.0..10.0))
                .collect();
            Array2::from_shape_vec((n_samples, n_features), data).unwrap()
        };

        let categorical_data = {
            let data: Vec<f64> = (0..n_samples * n_features)
                .map(|_| rng.random_range(0..8) as f64)
                .collect();
            Array2::from_shape_vec((n_samples, n_features), data).unwrap()
        };

        let data_indices: Vec<usize> = (0..n_samples).collect();
        let feature_idx = 0;

        group.throughput(Throughput::Elements(n_samples as u64));

        // Continuous splitting
        let continuous_splitter = ContinuousSplit;
        group.bench_with_input(
            BenchmarkId::new("continuous_split", name),
            &(continuous_data, feature_idx, data_indices.clone()),
            |b, (data, feature_idx, data_indices)| {
                b.iter(|| {
                    let threshold = 0.0;
                    let (left, right) = continuous_splitter.split_data_indices(
                        black_box(data),
                        black_box(*feature_idx),
                        black_box(threshold),
                        black_box(data_indices),
                    );
                    black_box((left, right));
                });
            },
        );

        // Categorical splitting
        let categorical_splitter = OneHotSplit;
        group.bench_with_input(
            BenchmarkId::new("categorical_split", name),
            &(categorical_data, feature_idx, data_indices),
            |b, (data, feature_idx, data_indices)| {
                b.iter(|| {
                    let threshold = 2;
                    let (left, right) = categorical_splitter.split_data_indices(
                        black_box(data),
                        black_box(*feature_idx),
                        black_box(threshold),
                        black_box(data_indices),
                    );
                    black_box((left, right));
                });
            },
        );

        // Split value sampling
        let continuous_candidates: Vec<f64> =
            (0..100).map(|_| rng.random_range(-5.0..5.0)).collect();

        group.bench_with_input(
            BenchmarkId::new("continuous_sampling", name),
            &continuous_candidates,
            |b, candidates| {
                b.iter_batched(
                    || SmallRng::seed_from_u64(42),
                    |mut bench_rng| {
                        let result = continuous_splitter
                            .sample_split_value(&mut bench_rng, black_box(candidates));
                        black_box(result);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Weight Normalization Benchmarks
// ============================================================================

fn benchmark_weight_normalization_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization_comprehensive");
    let mut rng = SmallRng::seed_from_u64(42);

    let sizes = vec![10, 50, 100, 500, 1000];

    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));

        // Random log weights
        let random_log_weights = generate_log_weights(n, &mut rng);
        group.bench_with_input(
            BenchmarkId::new("random_weights", n),
            &random_log_weights,
            |b, weights| {
                b.iter(|| {
                    let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                    black_box(normalized);
                });
            },
        );

        // Extreme weights (one dominant)
        let mut extreme_weights = vec![-100.0; n];
        if n > 0 {
            extreme_weights[0] = 0.0;
        }
        group.bench_with_input(
            BenchmarkId::new("extreme_weights", n),
            &extreme_weights,
            |b, weights| {
                b.iter(|| {
                    let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                    black_box(normalized);
                });
            },
        );

        // Uniform weights
        let uniform_weights = vec![-2.0; n];
        group.bench_with_input(
            BenchmarkId::new("uniform_weights", n),
            &uniform_weights,
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

// ============================================================================
// Integration Benchmarks
// ============================================================================

fn benchmark_integration_full_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("integration_full_workflow");

    const MAX_NODES: usize = 511;
    let scenarios = vec![
        ("quick", 50, 2, 5, 1),       // Quick scenario
        ("standard", 100, 5, 10, 2),  // Standard scenario
        ("intensive", 200, 8, 15, 3), // More intensive scenario
    ];

    for (name, n_samples, n_features, n_particles, n_iterations) in scenarios {
        group.throughput(Throughput::Elements((n_particles * n_iterations) as u64));

        group.bench_with_input(
            BenchmarkId::new("complete_bart_workflow", name),
            &(n_samples, n_features, n_particles, n_iterations),
            |b, &(n_samples, n_features, n_particles, n_iterations)| {
                b.iter_batched(
                    || {
                        let mut setup_rng = SmallRng::seed_from_u64(42);
                        let (x_data, y_data) =
                            generate_synthetic_data(n_samples, n_features, &mut setup_rng);
                        let init_leaf_value = y_data.mean().unwrap();

                        let initial_particles = create_initial_particles::<MAX_NODES>(
                            n_particles,
                            init_leaf_value,
                            n_samples,
                        );

                        let init_weights = vec![1.0 / n_particles as f64; n_particles];
                        let initial_state = BartState::new(initial_particles, init_weights);
                        let context = create_bart_context(x_data);

                        let sampler = ParticleGibbsSampler::new(
                            Moves::new(),
                            BARTWeighter,
                            SystematicResampling,
                        );

                        let bench_rng = SmallRng::seed_from_u64(42);

                        (sampler, initial_state, context, bench_rng)
                    },
                    |(sampler, initial_state, context, mut rng)| {
                        let result = sampler.run(
                            &mut rng,
                            black_box(initial_state),
                            black_box(&context),
                            black_box(n_iterations),
                        );
                        black_box(result);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory and Performance Pattern Benchmarks
// ============================================================================

fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Test different MAX_NODES constants
    let node_capacities = vec![("127", 127), ("511", 511), ("1023", 1023)];

    for (name, max_nodes) in node_capacities {
        group.bench_with_input(
            BenchmarkId::new("memory_scaling", name),
            &max_nodes,
            |b, &max_nodes| match max_nodes {
                127 => benchmark_memory_helper::<127>(b),
                511 => benchmark_memory_helper::<511>(b),
                1023 => benchmark_memory_helper::<1023>(b),
                _ => {}
            },
        );
    }

    group.finish();
}

fn benchmark_memory_helper<const MAX_NODES: usize>(b: &mut criterion::Bencher) {
    b.iter_batched(
        || {
            let mut setup_rng = SmallRng::seed_from_u64(42);
            let (x_data, y_data) = generate_synthetic_data(100, 5, &mut setup_rng);
            let init_leaf_value = y_data.mean().unwrap();
            let n_particles = 10;

            let initial_particles =
                create_initial_particles::<MAX_NODES>(n_particles, init_leaf_value, 100);

            let init_weights = vec![1.0 / n_particles as f64; n_particles];
            let initial_state = BartState::new(initial_particles, init_weights);
            let context = create_bart_context(x_data);

            let sampler =
                ParticleGibbsSampler::new(Moves::new(), BARTWeighter, SystematicResampling);

            let bench_rng = SmallRng::seed_from_u64(42);

            (sampler, initial_state, context, bench_rng)
        },
        |(sampler, initial_state, context, mut rng)| {
            let result = sampler.step(&mut rng, black_box(initial_state), black_box(&context));
            black_box(result);
        },
        criterion::BatchSize::SmallInput,
    );
}

// ============================================================================
// Criterion Groups and Main
// ============================================================================

criterion_group!(
    all_benchmarks,
    benchmark_smc_comprehensive,
    benchmark_resampling_comprehensive,
    benchmark_split_rules_comprehensive,
    benchmark_weight_normalization_comprehensive,
    benchmark_integration_full_workflow,
    benchmark_memory_patterns,
);

criterion_main!(all_benchmarks);
