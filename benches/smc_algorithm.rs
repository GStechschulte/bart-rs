use std::hint::black_box;
use std::rc::Rc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numpy::ndarray::{Array1, Array2};
use rand::{rngs::SmallRng, Rng, SeedableRng};

use pymc_bart::{
    base::BartState,
    particle::{Particle, Tree},
    resampling::SystematicResampling,
    sampler::ParticleGibbsSampler,
    update::{BARTContext, BARTWeighter, Moves},
};

/// Generate synthetic regression data for benchmarking
fn generate_synthetic_data(
    n_samples: usize,
    n_features: usize,
    rng: &mut SmallRng,
) -> (Array2<f64>, Array1<f64>) {
    // Generate X data
    let x_data: Vec<f64> = (0..n_samples * n_features)
        .map(|_| rng.random_range(-5.0..5.0))
        .collect();
    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();

    // Generate y data with some pattern
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

/// Benchmark SMC step with MAX_NODES = 127
fn benchmark_smc_step_127(c: &mut Criterion) {
    const MAX_NODES: usize = 127;
    let mut group = c.benchmark_group("smc_step_127");

    let data_sizes = vec![
        (50, 2),   // small dataset
        (100, 5),  // medium dataset
        (200, 10), // large dataset
    ];

    let particle_counts = vec![5, 10, 20];

    for (n_samples, n_features) in data_sizes {
        for n_particles in &particle_counts {
            let param_str = format!(
                "samples={}_features={}_particles={}",
                n_samples, n_features, n_particles
            );

            group.throughput(Throughput::Elements(*n_particles as u64));

            group.bench_with_input(
                BenchmarkId::new("step", &param_str),
                &(n_samples, n_features, *n_particles),
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
                            let result = sampler.step(
                                &mut rng,
                                black_box(initial_state),
                                black_box(&context),
                            );
                            black_box(result);
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

/// Benchmark SMC step with MAX_NODES = 511
fn benchmark_smc_step_511(c: &mut Criterion) {
    const MAX_NODES: usize = 511;
    let mut group = c.benchmark_group("smc_step_511");

    let data_sizes = vec![(50, 2), (100, 5), (200, 10)];

    let particle_counts = vec![5, 10, 20];

    for (n_samples, n_features) in data_sizes {
        for n_particles in &particle_counts {
            let param_str = format!(
                "samples={}_features={}_particles={}",
                n_samples, n_features, n_particles
            );

            group.throughput(Throughput::Elements(*n_particles as u64));

            group.bench_with_input(
                BenchmarkId::new("step", &param_str),
                &(n_samples, n_features, *n_particles),
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
                            let result = sampler.step(
                                &mut rng,
                                black_box(initial_state),
                                black_box(&context),
                            );
                            black_box(result);
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

/// Benchmark SMC step with MAX_NODES = 1023
fn benchmark_smc_step_1023(c: &mut Criterion) {
    const MAX_NODES: usize = 1023;
    let mut group = c.benchmark_group("smc_step_1023");

    let data_sizes = vec![(50, 2), (100, 5), (200, 10)];

    let particle_counts = vec![5, 10, 20];

    for (n_samples, n_features) in data_sizes {
        for n_particles in &particle_counts {
            let param_str = format!(
                "samples={}_features={}_particles={}",
                n_samples, n_features, n_particles
            );

            group.throughput(Throughput::Elements(*n_particles as u64));

            group.bench_with_input(
                BenchmarkId::new("step", &param_str),
                &(n_samples, n_features, *n_particles),
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
                            let result = sampler.step(
                                &mut rng,
                                black_box(initial_state),
                                black_box(&context),
                            );
                            black_box(result);
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

/// Benchmark multiple SMC iterations
fn benchmark_smc_run_multiple_iterations(c: &mut Criterion) {
    const MAX_NODES: usize = 511;
    let mut group = c.benchmark_group("smc_multiple_iterations");

    let n_samples = 100;
    let n_features = 5;
    let n_particles = 10;
    let iteration_counts = vec![1, 2, 5, 10];

    for n_iterations in iteration_counts {
        group.throughput(Throughput::Elements((n_particles * n_iterations) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(n_iterations),
            &n_iterations,
            |b, &n_iterations| {
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

/// Benchmark memory usage patterns with different MAX_NODES
fn benchmark_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");

    let max_nodes_variants = vec![("127", 127), ("255", 255), ("511", 511), ("1023", 1023)];

    let n_samples = 100;
    let n_features = 5;
    let n_particles = 10;

    for (name, max_nodes) in max_nodes_variants {
        group.bench_with_input(
            BenchmarkId::new("node_capacity", name),
            &max_nodes,
            |b, &max_nodes| {
                match max_nodes {
                    127 => {
                        const MAX_NODES: usize = 127;
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
                                let result = sampler.step(
                                    &mut rng,
                                    black_box(initial_state),
                                    black_box(&context),
                                );
                                black_box(result);
                            },
                            criterion::BatchSize::SmallInput,
                        );
                    }
                    511 => {
                        const MAX_NODES: usize = 511;
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
                                let result = sampler.step(
                                    &mut rng,
                                    black_box(initial_state),
                                    black_box(&context),
                                );
                                black_box(result);
                            },
                            criterion::BatchSize::SmallInput,
                        );
                    }
                    1023 => {
                        const MAX_NODES: usize = 1023;
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
                                let result = sampler.step(
                                    &mut rng,
                                    black_box(initial_state),
                                    black_box(&context),
                                );
                                black_box(result);
                            },
                            criterion::BatchSize::SmallInput,
                        );
                    }
                    _ => {} // Handle other cases as needed
                }
            },
        );
    }

    group.finish();
}

criterion_group!(
    smc_benches,
    benchmark_smc_step_127,
    benchmark_smc_step_511,
    benchmark_smc_step_1023,
    benchmark_smc_run_multiple_iterations,
    benchmark_memory_scaling,
);

criterion_main!(smc_benches);
