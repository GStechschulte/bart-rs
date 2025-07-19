use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use bumpalo::Bump;
use pymc_bart::forest::Forest;
use pymc_bart::resampling::{ResamplingStrategy, SystematicResampling};
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn generate_weights(n: usize, rng: &mut SmallRng) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
    let sum: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= sum);
    weights
}

fn benchmark_systematic_resample(c: &mut Criterion) {
    let mut group = c.benchmark_group("systematic_resample");
    let mut rng = SmallRng::seed_from_u64(42);

    let sizes = vec![10, 20, 40, 80, 160, 320, 640, 1280];

    for n in sizes.iter() {
        group.throughput(Throughput::Elements(*n as u64));
        let weights = generate_weights(*n, &mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(n), &weights, |b, w| {
            b.iter_batched(
                || rng.clone(),
                |mut batch_rng| SystematicResampling::resample(&mut batch_rng, black_box(w)),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_tree_resampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_resampling");

    // Test parameters
    let n_particles_sizes = vec![5, 10, 20, 50, 100];
    let n_samples_sizes = vec![100, 200, 400, 800, 1_600];
    let n_splits_sizes = vec![5, 10, 20, 40, 80];

    const N_FEATURES: usize = 10;
    const MAX_DEPTH: usize = 10;

    for &n_particles in &n_particles_sizes {
        for &n_samples in &n_samples_sizes {
            for &n_splits in &n_splits_sizes {
                let param_str = format!(
                    "particles={}, samples={}, splits={}",
                    n_particles, n_samples, n_splits
                );

                group.throughput(Throughput::Elements(n_particles as u64));

                group.bench_with_input(
                    BenchmarkId::new("full_resample_clone", param_str),
                    &(n_particles, n_samples, n_splits),
                    |b, &(n_particles, n_samples, n_splits)| {
                        b.iter_batched(
                            || {
                                let mut setup_rng = SmallRng::seed_from_u64(42);
                                let weights: Vec<f64> = (0..n_particles)
                                    .map(|_| setup_rng.random::<f64>())
                                    .collect();
                                let sum: f64 = weights.iter().sum();
                                let normalized_weights: Vec<f64> =
                                    weights.iter().map(|w| w / sum).collect();

                                let bench_rng = SmallRng::seed_from_u64(42);
                                (normalized_weights, bench_rng)
                            },
                            |(weights, mut bench_rng)| {
                                let arena = Bump::new();
                                let mut forest = Forest::<MAX_DEPTH>::new(&arena, n_particles);
                                let mut setup_rng = SmallRng::seed_from_u64(42);

                                for _ in 0..n_particles {
                                    forest.plant_random_tree(
                                        0.0,
                                        n_samples,
                                        n_splits,
                                        N_FEATURES,
                                        &mut setup_rng,
                                    );
                                    forest.add_weight(setup_rng.random::<f64>());
                                }

                                let indices = SystematicResampling::resample(
                                    &mut bench_rng,
                                    black_box(&weights),
                                );
                                forest.resample_trees(black_box(&indices)).unwrap();
                            },
                            criterion::BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_systematic_resample,
    benchmark_tree_resampling,
);
criterion_main!(benches);
