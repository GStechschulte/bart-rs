use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use pymc_bart::resampling::{ResamplingStrategy, SystematicResampling};
use pymc_bart::sampler::normalize_weights;

fn generate_weights(n: usize, rng: &mut SmallRng) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
    let sum: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= sum);
    weights
}

fn generate_uniform_weights(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

fn generate_skewed_weights(n: usize, rng: &mut SmallRng) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..n)
        .map(|i| {
            if i < n / 10 {
                rng.random_range(5.0..10.0) // High weights for first 10%
            } else {
                rng.random_range(0.1..1.0) // Low weights for the rest
            }
        })
        .collect();
    let sum: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= sum);
    weights
}

fn benchmark_systematic_resample(c: &mut Criterion) {
    let mut group = c.benchmark_group("systematic_resample");
    let mut rng = SmallRng::seed_from_u64(42);

    let sizes = vec![10, 20, 50, 100, 200, 500, 1000, 2000];

    for n in sizes.iter() {
        group.throughput(Throughput::Elements(*n as u64));
        let weights = generate_weights(*n, &mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(n), &weights, |b, w| {
            b.iter_batched(
                || rng.clone(),
                |mut batch_rng| {
                    let indices: Vec<usize> = SystematicResampling::resample(
                        &mut batch_rng,
                        black_box(w.iter().copied()),
                    )
                    .collect();
                    black_box(indices);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_systematic_resample_uniform_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("systematic_resample_uniform");
    let mut rng = SmallRng::seed_from_u64(42);

    let sizes = vec![10, 20, 50, 100, 200, 500, 1000];

    for n in sizes.iter() {
        group.throughput(Throughput::Elements(*n as u64));
        let weights = generate_uniform_weights(*n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &weights, |b, w| {
            b.iter_batched(
                || rng.clone(),
                |mut batch_rng| {
                    let indices: Vec<usize> = SystematicResampling::resample(
                        &mut batch_rng,
                        black_box(w.iter().copied()),
                    )
                    .collect();
                    black_box(indices);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_systematic_resample_skewed_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("systematic_resample_skewed");
    let mut rng = SmallRng::seed_from_u64(42);

    let sizes = vec![10, 20, 50, 100, 200, 500, 1000];

    for n in sizes.iter() {
        group.throughput(Throughput::Elements(*n as u64));
        let weights = generate_skewed_weights(*n, &mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(n), &weights, |b, w| {
            b.iter_batched(
                || rng.clone(),
                |mut batch_rng| {
                    let indices: Vec<usize> = SystematicResampling::resample(
                        &mut batch_rng,
                        black_box(w.iter().copied()),
                    )
                    .collect();
                    black_box(indices);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_weight_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_normalization");
    let mut rng = SmallRng::seed_from_u64(42);

    let sizes = vec![10, 20, 50, 100, 200, 500, 1000, 2000];

    for n in sizes.iter() {
        group.throughput(Throughput::Elements(*n as u64));

        // Test with random log weights (typical use case)
        let log_weights: Vec<f64> = (0..(*n))
            .map(|_| rng.random_range(-10.0..0.0)) // Typical log probability range
            .collect();

        group.bench_with_input(
            BenchmarkId::new("random_log_weights", n),
            &log_weights,
            |b, weights| {
                b.iter(|| {
                    let normalized: Vec<f64> = normalize_weights(black_box(weights)).collect();
                    black_box(normalized);
                });
            },
        );

        // Test with extreme log weights (numerical stability test)
        let extreme_log_weights: Vec<f64> = (0..(*n))
            .map(|i| {
                if i == 0 {
                    0.0 // One very high weight
                } else {
                    -100.0 // Rest very low
                }
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("extreme_log_weights", n),
            &extreme_log_weights,
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

fn benchmark_resampling_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampling_memory_patterns");
    let mut rng = SmallRng::seed_from_u64(42);

    let particle_counts = vec![10, 50, 100, 500, 1000];

    for n_particles in particle_counts {
        group.throughput(Throughput::Elements(n_particles as u64));

        // Benchmark memory allocation patterns
        group.bench_with_input(
            BenchmarkId::new("vector_collection", n_particles),
            &n_particles,
            |b, &n_particles| {
                b.iter_batched(
                    || {
                        let weights = generate_weights(n_particles, &mut rng.clone());
                        let bench_rng = SmallRng::seed_from_u64(42);
                        (weights, bench_rng)
                    },
                    |(weights, mut bench_rng)| {
                        // Collect into Vec to measure full memory allocation
                        let indices: Vec<usize> = SystematicResampling::resample(
                            &mut bench_rng,
                            black_box(weights.iter().copied()),
                        )
                        .collect();
                        black_box(indices);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Benchmark streaming pattern (no collection)
        group.bench_with_input(
            BenchmarkId::new("streaming", n_particles),
            &n_particles,
            |b, &n_particles| {
                b.iter_batched(
                    || {
                        let weights = generate_weights(n_particles, &mut rng.clone());
                        let bench_rng = SmallRng::seed_from_u64(42);
                        (weights, bench_rng)
                    },
                    |(weights, mut bench_rng)| {
                        // Process iterator without collecting
                        let mut sum = 0usize;
                        for idx in SystematicResampling::resample(
                            &mut bench_rng,
                            black_box(weights.iter().copied()),
                        ) {
                            sum += idx;
                        }
                        black_box(sum);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn benchmark_resampling_with_particle_cloning(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampling_with_cloning");
    let mut rng = SmallRng::seed_from_u64(42);

    let particle_counts = vec![10, 20, 50, 100];
    let particle_size_variants = vec![("small", 10), ("medium", 100), ("large", 1000)];

    for n_particles in particle_counts {
        for (size_name, data_size) in &particle_size_variants {
            let param_str = format!("particles={}_size={}", n_particles, size_name);

            group.throughput(Throughput::Elements(n_particles as u64));

            group.bench_with_input(
                BenchmarkId::new("full_workflow", &param_str),
                &(n_particles, *data_size),
                |b, &(n_particles, data_size)| {
                    b.iter_batched(
                        || {
                            // Simulate particle data (e.g., trees)
                            let particles: Vec<Vec<f64>> = (0..n_particles)
                                .map(|_| (0..data_size).map(|_| rng.random::<f64>()).collect())
                                .collect();

                            let weights = generate_weights(n_particles, &mut rng.clone());
                            let bench_rng = SmallRng::seed_from_u64(42);

                            (particles, weights, bench_rng)
                        },
                        |(particles, weights, mut bench_rng)| {
                            // Full resampling workflow: resample + clone particles
                            let indices: Vec<usize> = SystematicResampling::resample(
                                &mut bench_rng,
                                black_box(weights.iter().copied()),
                            )
                            .collect();

                            let resampled_particles: Vec<Vec<f64>> = indices
                                .into_iter()
                                .map(|idx| black_box(particles[idx].clone()))
                                .collect();

                            black_box(resampled_particles);
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    resample_benches,
    benchmark_systematic_resample,
    benchmark_systematic_resample_uniform_weights,
    benchmark_systematic_resample_skewed_weights,
    benchmark_weight_normalization,
    benchmark_resampling_memory_patterns,
    benchmark_resampling_with_particle_cloning,
);

criterion_main!(resample_benches);
