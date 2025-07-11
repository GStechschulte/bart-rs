use bumpalo::Bump;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use pymc_bart::forest::Forest;

fn benchmark_forest(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_creation");

    // Define parameter ranges when planting the forest
    let particle_counts = [10, 50, 100];
    let sample_counts = [100, 1000, 10000];
    let max_depths = [2, 4, 6];

    for &n_particles in &particle_counts {
        for &n_samples in &sample_counts {
            for &max_depth in &max_depths {
                // Create Bump allocator
                let bump = Bump::new();
                let benchmark_id = BenchmarkId::new(
                    "Bumpalo",
                    format!("p{}_s{}_d{}", n_particles, n_samples, max_depth),
                );

                group.bench_function(benchmark_id, |b| {
                    b.iter(|| {
                        let mut forest = black_box(Forest::new(&bump, n_particles));

                        for _ in 0..n_particles {
                            forest.add_particle(0.0, n_samples, max_depth);
                        }

                        // Deallocate, but reuse the same arena
                        // bump.reset();
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_forest);
criterion_main!(benches);
