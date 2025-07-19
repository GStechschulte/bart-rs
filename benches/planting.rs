use std::hint::black_box;

use bumpalo::Bump;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pymc_bart::forest::Forest;

fn benchmark_forest(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_creation");

    // Define parameter ranges for benchmarking
    let particle_counts = [10, 50, 100];
    let sample_counts = [100, 1000, 10000];
    let max_depths = [2, 4, 6];

    for &n_particles in &particle_counts {
        for &n_samples in &sample_counts {
            for &max_depth in &max_depths {
                let benchmark_id = BenchmarkId::new(
                    "ForestCreation",
                    format!(
                        "particles={}, samples={}, depth={}",
                        n_particles, n_samples, max_depth
                    ),
                );

                group.bench_function(benchmark_id, |b| {
                    b.iter(|| {
                        // The arena is created locally for each iteration
                        let arena = Bump::new();

                        match max_depth {
                            2 => {
                                let mut forest = black_box(Forest::<2>::new(&arena, n_particles));
                                for _ in 0..n_particles {
                                    forest.plant_tree(0.0, n_samples);
                                }
                            }
                            4 => {
                                let mut forest = black_box(Forest::<4>::new(&arena, n_particles));
                                for _ in 0..n_particles {
                                    forest.plant_tree(0.0, n_samples);
                                }
                            }
                            6 => {
                                let mut forest = black_box(Forest::<6>::new(&arena, n_particles));
                                for _ in 0..n_particles {
                                    forest.plant_tree(0.0, n_samples);
                                }
                            }
                            // Since the loop controls the values, other arms are not reachable.
                            _ => unreachable!(),
                        }
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_forest);
criterion_main!(benches);
