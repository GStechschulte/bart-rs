use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numpy::ndarray::Array2;
use rand::{rngs::SmallRng, Rng, SeedableRng};

use pymc_bart::splitting::{ContinuousSplit, OneHotSplit, SplitRule, SplitRules};

/// Generate synthetic continuous data for benchmarking
fn generate_continuous_data(
    n_samples: usize,
    n_features: usize,
    rng: &mut SmallRng,
) -> Array2<f64> {
    let data: Vec<f64> = (0..n_samples * n_features)
        .map(|_| rng.random_range(-10.0..10.0))
        .collect();
    Array2::from_shape_vec((n_samples, n_features), data).unwrap()
}

/// Generate synthetic categorical data for benchmarking
fn generate_categorical_data(
    n_samples: usize,
    n_features: usize,
    n_categories: usize,
    rng: &mut SmallRng,
) -> Array2<f64> {
    let data: Vec<f64> = (0..n_samples * n_features)
        .map(|_| rng.random_range(0..n_categories) as f64)
        .collect();
    Array2::from_shape_vec((n_samples, n_features), data).unwrap()
}

/// Generate candidate values for continuous splits
fn generate_continuous_candidates(n_candidates: usize, rng: &mut SmallRng) -> Vec<f64> {
    (0..n_candidates)
        .map(|_| rng.random_range(-5.0..5.0))
        .collect()
}

/// Generate candidate values for categorical splits
fn generate_categorical_candidates(n_candidates: usize, n_categories: usize) -> Vec<i32> {
    let mut candidates = Vec::new();
    for _ in 0..n_candidates {
        candidates.push((candidates.len() % n_categories) as i32);
    }
    candidates
}

/// Benchmark continuous split value sampling
fn benchmark_continuous_split_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("continuous_split_sampling");
    let mut rng = SmallRng::seed_from_u64(42);
    let splitter = ContinuousSplit;

    let candidate_counts = vec![10, 50, 100, 500, 1000, 5000];

    for n_candidates in candidate_counts {
        group.throughput(Throughput::Elements(n_candidates as u64));

        let candidates = generate_continuous_candidates(n_candidates, &mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(n_candidates),
            &candidates,
            |b, candidates| {
                b.iter_batched(
                    || SmallRng::seed_from_u64(42),
                    |mut bench_rng| {
                        let result =
                            splitter.sample_split_value(&mut bench_rng, black_box(candidates));
                        black_box(result);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark categorical split value sampling
fn benchmark_categorical_split_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("categorical_split_sampling");
    let mut rng = SmallRng::seed_from_u64(42);
    let splitter = OneHotSplit;

    let test_cases = vec![
        ("few_categories", 100, 5),   // Many samples, few categories
        ("many_categories", 100, 50), // Many samples, many categories
        ("sparse", 1000, 10),         // Very many samples, moderate categories
    ];

    for (name, n_candidates, n_categories) in test_cases {
        group.throughput(Throughput::Elements(n_candidates as u64));

        let candidates = generate_categorical_candidates(n_candidates, n_categories);

        group.bench_with_input(
            BenchmarkId::new("sample", name),
            &candidates,
            |b, candidates| {
                b.iter_batched(
                    || SmallRng::seed_from_u64(42),
                    |mut bench_rng| {
                        let result =
                            splitter.sample_split_value(&mut bench_rng, black_box(candidates));
                        black_box(result);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark continuous data splitting
fn benchmark_continuous_data_splitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("continuous_data_splitting");
    let mut rng = SmallRng::seed_from_u64(42);
    let splitter = ContinuousSplit;

    let data_sizes = vec![
        (50, 2),    // Small dataset
        (200, 5),   // Medium dataset
        (1000, 10), // Large dataset
        (5000, 20), // Very large dataset
    ];

    for (n_samples, n_features) in data_sizes {
        let data = generate_continuous_data(n_samples, n_features, &mut rng);
        let feature_idx = 0;
        let threshold = 0.0;
        let data_indices: Vec<usize> = (0..n_samples).collect();

        let param_str = format!("samples={}_features={}", n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("split", &param_str),
            &(data, feature_idx, threshold, data_indices),
            |b, (data, feature_idx, threshold, data_indices)| {
                b.iter(|| {
                    let (left, right) = splitter.split_data_indices(
                        black_box(data),
                        black_box(*feature_idx),
                        black_box(*threshold),
                        black_box(data_indices),
                    );
                    black_box((left, right));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark categorical data splitting
fn benchmark_categorical_data_splitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("categorical_data_splitting");
    let mut rng = SmallRng::seed_from_u64(42);
    let splitter = OneHotSplit;

    let test_cases = vec![
        (50, 2, 5),     // Small dataset, few categories
        (200, 5, 10),   // Medium dataset, moderate categories
        (1000, 10, 8),  // Large dataset, moderate categories
        (5000, 15, 20), // Very large dataset, many categories
    ];

    for (n_samples, n_features, n_categories) in test_cases {
        let data = generate_categorical_data(n_samples, n_features, n_categories, &mut rng);
        let feature_idx = 0;
        let threshold = 2; // Split on category 2
        let data_indices: Vec<usize> = (0..n_samples).collect();

        let param_str = format!(
            "samples={}_features={}_categories={}",
            n_samples, n_features, n_categories
        );

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("split", &param_str),
            &(data, feature_idx, threshold, data_indices),
            |b, (data, feature_idx, threshold, data_indices)| {
                b.iter(|| {
                    let (left, right) = splitter.split_data_indices(
                        black_box(data),
                        black_box(*feature_idx),
                        black_box(*threshold),
                        black_box(data_indices),
                    );
                    black_box((left, right));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark dynamic dispatch through SplitRules enum
fn benchmark_split_rules_dynamic_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_rules_dynamic_dispatch");
    let mut rng = SmallRng::seed_from_u64(42);

    let data_size = (500, 5);
    let (n_samples, n_features) = data_size;

    // Test both continuous and categorical rules
    let continuous_data = generate_continuous_data(n_samples, n_features, &mut rng);
    let categorical_data = generate_categorical_data(n_samples, n_features, 10, &mut rng);

    let continuous_rule = SplitRules::Continuous(ContinuousSplit);
    let categorical_rule = SplitRules::OneHot(OneHotSplit);

    let feature_idx = 0;
    let data_indices: Vec<usize> = (0..n_samples).collect();

    group.throughput(Throughput::Elements(n_samples as u64));

    // Benchmark continuous rule through enum
    group.bench_with_input(
        BenchmarkId::new("continuous_enum", "split"),
        &(
            continuous_data,
            continuous_rule,
            feature_idx,
            data_indices.clone(),
        ),
        |b, (data, rule, feature_idx, data_indices)| {
            b.iter(|| {
                let threshold = 0.0;
                let (left, right) = rule.split_data_indices(
                    black_box(data),
                    black_box(*feature_idx),
                    black_box(threshold),
                    black_box(data_indices),
                );
                black_box((left, right));
            });
        },
    );

    // Benchmark categorical rule through enum
    group.bench_with_input(
        BenchmarkId::new("categorical_enum", "split"),
        &(
            categorical_data,
            categorical_rule,
            feature_idx,
            data_indices,
        ),
        |b, (data, rule, feature_idx, data_indices)| {
            b.iter(|| {
                let threshold = 2.0;
                let (left, right) = rule.split_data_indices(
                    black_box(data),
                    black_box(*feature_idx),
                    black_box(threshold),
                    black_box(data_indices),
                );
                black_box((left, right));
            });
        },
    );

    group.finish();
}

/// Benchmark split value sampling through SplitRules enum
fn benchmark_split_rules_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_rules_sampling");
    let mut rng = SmallRng::seed_from_u64(42);

    let continuous_candidates = generate_continuous_candidates(100, &mut rng);
    let categorical_candidates: Vec<f64> = generate_categorical_candidates(100, 10)
        .into_iter()
        .map(|x| x as f64)
        .collect();

    let continuous_rule = SplitRules::Continuous(ContinuousSplit);
    let categorical_rule = SplitRules::OneHot(OneHotSplit);

    group.throughput(Throughput::Elements(100));

    // Benchmark continuous sampling through enum
    group.bench_with_input(
        BenchmarkId::new("continuous_enum", "sample"),
        &(continuous_rule, continuous_candidates),
        |b, (rule, candidates)| {
            b.iter_batched(
                || SmallRng::seed_from_u64(42),
                |mut bench_rng| {
                    let result = rule.sample_split_value(&mut bench_rng, black_box(candidates));
                    black_box(result);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    // Benchmark categorical sampling through enum
    group.bench_with_input(
        BenchmarkId::new("categorical_enum", "sample"),
        &(categorical_rule, categorical_candidates),
        |b, (rule, candidates)| {
            b.iter_batched(
                || SmallRng::seed_from_u64(42),
                |mut bench_rng| {
                    let result = rule.sample_split_value(&mut bench_rng, black_box(candidates));
                    black_box(result);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

/// Benchmark memory allocation patterns in splits
fn benchmark_split_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_memory_patterns");
    let mut rng = SmallRng::seed_from_u64(42);
    let splitter = ContinuousSplit;

    let split_ratios = vec![
        ("balanced", 0.5),     // 50/50 split
        ("skewed_left", 0.1),  // 10/90 split
        ("skewed_right", 0.9), // 90/10 split
    ];

    let n_samples = 1000;
    let n_features = 5;

    for (name, target_ratio) in split_ratios {
        let data = generate_continuous_data(n_samples, n_features, &mut rng);
        let feature_idx = 0;

        // Calculate threshold to achieve target split ratio
        let mut feature_values: Vec<f64> = (0..n_samples).map(|i| data[[i, feature_idx]]).collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = (n_samples as f64 * target_ratio) as usize;
        let threshold = feature_values[threshold_idx.min(n_samples - 1)];

        let data_indices: Vec<usize> = (0..n_samples).collect();

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("memory_allocation", name),
            &(data, feature_idx, threshold, data_indices),
            |b, (data, feature_idx, threshold, data_indices)| {
                b.iter(|| {
                    let (left, right) = splitter.split_data_indices(
                        black_box(data),
                        black_box(*feature_idx),
                        black_box(*threshold),
                        black_box(data_indices),
                    );
                    // Force allocation by accessing lengths
                    black_box((left.len(), right.len()));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark edge cases in split rules
fn benchmark_split_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_edge_cases");
    let mut rng = SmallRng::seed_from_u64(42);

    // Test with very few candidates
    let few_candidates = vec![1.0, 2.0];
    let continuous_splitter = ContinuousSplit;

    group.bench_with_input(
        BenchmarkId::new("continuous", "few_candidates"),
        &few_candidates,
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

    // Test with identical candidates
    let identical_candidates = vec![5.0; 100];

    group.bench_with_input(
        BenchmarkId::new("continuous", "identical_candidates"),
        &identical_candidates,
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

    // Test categorical with single category
    let single_category = vec![0];
    let categorical_splitter = OneHotSplit;

    group.bench_with_input(
        BenchmarkId::new("categorical", "single_category"),
        &single_category,
        |b, candidates| {
            b.iter_batched(
                || SmallRng::seed_from_u64(42),
                |mut bench_rng| {
                    let result = categorical_splitter
                        .sample_split_value(&mut bench_rng, black_box(candidates));
                    black_box(result);
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

criterion_group!(
    split_rules_benches,
    benchmark_continuous_split_sampling,
    benchmark_categorical_split_sampling,
    benchmark_continuous_data_splitting,
    benchmark_categorical_data_splitting,
    benchmark_split_rules_dynamic_dispatch,
    benchmark_split_rules_sampling,
    benchmark_split_memory_patterns,
    benchmark_split_edge_cases,
);

criterion_main!(split_rules_benches);
