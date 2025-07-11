use bumpalo::Bump;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use numpy::ndarray::{Array, Array2};
use rand::prelude::*;
use std::ffi::c_double;
use std::hint::black_box;

use pymc_bart::base::State;
use pymc_bart::forest::ParticleSet;

/// A dummy logp function required to initialize the `State` struct.å
/// In real usage, this would compute the log-probability of the data given the model.
unsafe extern "C" fn dummy_logp(_: *const f64, _: usize) -> c_double {
    0.0
}

/// Simulates the particle growth process by appending random data to the particle vectors.
///
/// This function mimics the tree mutation operations that occur during BART sampling,
/// where particles (candidate trees) are grown by adding split points and leaf values.
///
/// # Arguments
/// * `particles` - Mutable reference to the particle set containing arena-allocated vectors
/// * `n_features` - Number of features available for splitting (bounds feature selection)
/// * `n_growth_steps` - Number of growth operations to perform
/// * `rng` - Random number generator for sampling split points and values
///
/// # Performance considerations
/// - All vectors are allocated in the Bumpalo arena for fast, temporary allocation
/// - Vector growth is tested under realistic workload patterns
/// - Random sampling simulates the decision-making process in real BART mutations
fn grow_particles(
    particles: &mut ParticleSet,
    n_features: usize,
    n_growth_steps: usize,
    rng: &mut impl Rng,
) {
    // Pre-define distributions for sampling split features and values.
    let feature_dist = Uniform::new(0, n_features);
    let split_value_dist = Uniform::new(0, usize::MAX); // Placeholder for split values
    let leaf_value_dist = Uniform::new(0, usize::MAX); // Placeholder for leaf values

    // In a loop, append new data to simulate the growth of tree structures.
    // This directly tests the performance of pushing to the `Vec`s
    // allocated in the Bumpalo arena.
    for _ in 0..n_growth_steps {
        particles.split_indices.push(rng.sample(feature_dist));
        particles.split_values.push(rng.sample(split_value_dist));
        particles.leaf_values.push(rng.sample(leaf_value_dist));
    }
}

/// Benchmark suite for measuring particle growth performance with arena allocation.
///
/// This benchmark evaluates the performance characteristics of the core BART particle
/// mutation operations under different workload sizes. It measures:
///
/// 1. **Arena allocation overhead**: Cost of creating temporary memory arenas
/// 2. **Vector growth performance**: How efficiently particles can be mutated
/// 3. **Scalability**: Performance characteristics across different growth step counts
///
/// # Benchmark parameters
/// - Dataset: 1000 samples × 50 features
/// - Particles: 100 concurrent particle candidates
/// - Growth steps: [10, 50, 100, 200]
///
/// # Expected performance characteristics
/// - Linear scaling with number of growth steps
/// - Consistent per-operation performance due to arena allocation
/// - Minimal allocation overhead due to bump allocation strategy
fn benchmark_particle_growth(c: &mut Criterion) {
    let mut group = c.benchmark_group("ParticleGrowthWithArena");

    // --- Simulation Setup ---
    let n_samples = 1000;
    let n_features = 50;
    let n_particles = 100;

    // Simulate feature matrix `X` and target `y`.
    let mut rng = StdRng::seed_from_u64(42);
    let x_data: Array2<f64> =
        Array::random_using((n_samples, n_features), Uniform::new(0., 1.), &mut rng);
    let x_data = x_data.into_dyn(); // Convert to dynamic dimensionality
    let y_data = Array::random_using(n_samples, Uniform::new(0., 1.), &mut rng);

    // Create the main state object that holds global data.
    let state = State::new(x_data, y_data, dummy_logp, 50, n_particles);

    // Benchmark across a range of growth steps to see how performance scales.
    for n_growth_steps in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("GrowParticles", n_growth_steps),
            n_growth_steps,
            |b, &steps| {
                let mut rng = StdRng::seed_from_u64(42);

                // The `iter` block contains the code to be measured.
                b.iter(|| {
                    // Create a new Bump arena for each iteration, mirroring the
                    // `mcmc_loop` logic where the arena is temporary for each tree update.
                    let bump = Bump::new();
                    let mut particles = ParticleSet::new_in(black_box(&bump), state.n_particles);

                    // The `grow_particles` function simulates mutations.
                    // `black_box` prevents the compiler from optimizing away the calls.
                    grow_particles(
                        black_box(&mut particles),
                        state.X.shape()[1],
                        black_box(steps),
                        black_box(&mut rng),
                    );
                });
            },
        );
    }
    group.finish();
}

// Register the benchmark group with Criterion.
criterion_group!(benches, benchmark_particle_growth);
criterion_main!(benches);
