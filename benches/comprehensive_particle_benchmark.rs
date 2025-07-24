use std::hint::black_box;

use bumpalo::Bump;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
    Throughput,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

// Import arena-based implementation
use pymc_bart::forest::{Forest, Tree};
use pymc_bart::resampling::{ResamplingStrategy, SystematicResampling};
use pymc_bart::sampler::normalize_weights;

// Non-arena based implementations for comparison
mod non_arena {
    use rand::Rng;

    #[derive(Debug, Clone)]
    pub struct Tree {
        split_var: Vec<usize>,
        split_value: Vec<f64>,
        leaf_values: Vec<f64>,
        leaf_indices: Vec<usize>,
        max_depth: usize,
    }

    impl Tree {
        pub fn stump(init_leaf: f64, n_samples: usize, max_depth: usize) -> Self {
            let max_leaf_nodes = 1 << max_depth;
            let max_internal_nodes = max_leaf_nodes - 1;

            let mut leaf_values = Vec::with_capacity(max_leaf_nodes);
            leaf_values.push(init_leaf);

            let leaf_indices = vec![0; n_samples];

            Self {
                split_var: Vec::with_capacity(max_internal_nodes),
                split_value: Vec::with_capacity(max_internal_nodes),
                leaf_values,
                leaf_indices,
                max_depth,
            }
        }

        pub fn add_random_split(&mut self, rng: &mut impl Rng, n_features: usize) {
            if self.leaf_values.is_empty() {
                return;
            }

            let leaf_idx = rng.random_range(0..self.leaf_values.len());
            let split_var = rng.random_range(0..n_features);
            let split_value = rng.random::<f64>();

            self.split_var.push(split_var);
            self.split_value.push(split_value);

            let original_leaf_value = self.leaf_values[leaf_idx];
            self.leaf_values
                .push(original_leaf_value + rng.random::<f64>() * 0.1);
            self.leaf_values
                .push(original_leaf_value - rng.random::<f64>() * 0.1);
        }

        pub fn grow_tree(
            &mut self,
            rng: &mut impl Rng,
            n_features: usize,
            growth_probability: f64,
        ) {
            if rng.random::<f64>() < growth_probability && self.can_grow() {
                self.add_random_split(rng, n_features);
            }
        }

        fn can_grow(&self) -> bool {
            let current_depth = (self.split_var.len() as f64).log2().floor() as usize;
            current_depth < self.max_depth && self.leaf_values.len() < (1 << self.max_depth)
        }

        pub fn num_splits(&self) -> usize {
            self.split_var.len()
        }

        pub fn num_leaves(&self) -> usize {
            self.leaf_values.len()
        }

        pub fn num_samples(&self) -> usize {
            self.leaf_indices.len()
        }

        pub fn compute_complexity_penalty(&self) -> f64 {
            // Simple complexity penalty based on tree size
            (self.num_splits() as f64) * 0.01 + (self.num_leaves() as f64) * 0.005
        }
    }

    #[derive(Debug)]
    pub struct Forest {
        pub trees: Vec<Tree>,
        pub weights: Vec<f64>,
        max_depth: usize,
    }

    impl Forest {
        pub fn new(n_particles: usize, max_depth: usize) -> Self {
            Self {
                trees: Vec::with_capacity(n_particles),
                weights: Vec::with_capacity(n_particles),
                max_depth,
            }
        }

        pub fn plant_tree(&mut self, init_leaf: f64, n_samples: usize) {
            let tree = Tree::stump(init_leaf, n_samples, self.max_depth);
            self.trees.push(tree);
        }

        pub fn add_weight(&mut self, weight: f64) {
            self.weights.push(weight);
        }

        pub fn len(&self) -> usize {
            self.trees.len()
        }

        pub fn is_empty(&self) -> bool {
            self.trees.is_empty()
        }

        pub fn trees(&self) -> &[Tree] {
            &self.trees
        }

        pub fn trees_mut(&mut self) -> &mut [Tree] {
            &mut self.trees
        }

        pub fn weights(&self) -> &[f64] {
            &self.weights
        }

        pub fn weights_mut(&mut self) -> &mut [f64] {
            &mut self.weights
        }

        pub fn resample_trees(&mut self, indices: &[usize]) -> Result<(), &'static str> {
            if indices.len() != self.trees.len() {
                return Err("Number of indices must match number of trees");
            }

            let mut new_trees = Vec::with_capacity(indices.len());
            let mut new_weights = Vec::with_capacity(indices.len());

            for &idx in indices {
                if idx >= self.trees.len() {
                    return Err("Invalid tree index");
                }
                new_trees.push(self.trees[idx].clone());
                new_weights.push(self.weights[idx]);
            }

            self.trees = new_trees;
            self.weights = new_weights;
            Ok(())
        }

        pub fn clear(&mut self) {
            self.trees.clear();
            self.weights.clear();
        }

        pub fn plant_random_tree(
            &mut self,
            init_leaf: f64,
            n_samples: usize,
            n_splits: usize,
            n_features: usize,
            rng: &mut impl Rng,
        ) {
            let mut tree = Tree::stump(init_leaf, n_samples, self.max_depth);

            for _ in 0..n_splits {
                tree.add_random_split(rng, n_features);
            }

            self.trees.push(tree);
        }

        pub fn update_all_trees(
            &mut self,
            rng: &mut impl Rng,
            n_features: usize,
            growth_prob: f64,
        ) {
            for tree in &mut self.trees {
                tree.grow_tree(rng, n_features, growth_prob);
            }
        }

        pub fn compute_weights(&mut self, rng: &mut impl Rng) {
            self.weights.clear();
            for tree in &self.trees {
                // Simulate likelihood computation with complexity penalty
                let base_weight = rng.random::<f64>() * 10.0;
                let penalty = tree.compute_complexity_penalty();
                let weight = base_weight - penalty;
                self.weights.push(weight);
            }
        }
    }
}

// Benchmark configuration structures
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    pub n_particles: usize,
    pub n_samples: usize,
    pub n_features: usize,
    pub n_iterations: usize,
    pub initial_splits: usize,
    pub growth_probability: f64,
    pub max_depth: usize,
}

impl BenchmarkConfig {
    pub fn small() -> Self {
        Self {
            n_particles: 20,
            n_samples: 500,
            n_features: 10,
            n_iterations: 100,
            initial_splits: 3,
            growth_probability: 0.3,
            max_depth: 8,
        }
    }

    pub fn medium() -> Self {
        Self {
            n_particles: 50,
            n_samples: 2000,
            n_features: 25,
            n_iterations: 100,
            initial_splits: 8,
            growth_probability: 0.4,
            max_depth: 10,
        }
    }

    pub fn large() -> Self {
        Self {
            n_particles: 100,
            n_samples: 10000,
            n_features: 50,
            n_iterations: 50,
            initial_splits: 15,
            growth_probability: 0.5,
            max_depth: 12,
        }
    }

    pub fn stress_test() -> Self {
        Self {
            n_particles: 200,
            n_samples: 50000,
            n_features: 100,
            n_iterations: 20,
            initial_splits: 25,
            growth_probability: 0.6,
            max_depth: 15,
        }
    }

    pub fn name(&self) -> String {
        format!(
            "p{}_s{}_f{}_i{}",
            self.n_particles, self.n_samples, self.n_features, self.n_iterations
        )
    }

    pub fn memory_footprint_estimate(&self) -> usize {
        // Rough estimate of memory usage in bytes
        let tree_size = self.n_samples * 8 + (1 << self.max_depth) * 16;
        self.n_particles * tree_size * self.n_iterations
    }
}

// Comprehensive particle sampler simulation with arena allocation
fn simulate_bart_particle_sampler_arena<const MAX_DEPTH: usize>(
    config: &BenchmarkConfig,
    rng: &mut SmallRng,
) -> (f64, usize) {
    let arena = Bump::new();
    let mut total_complexity = 0.0;
    let mut total_resample_operations = 0;

    for iteration in 0..config.n_iterations {
        let mut forest = Forest::<MAX_DEPTH>::new(&arena, config.n_particles);

        // Phase 1: Initialize particles with random trees
        for _ in 0..config.n_particles {
            forest.plant_random_tree(
                0.0, // Mean-centered initialization
                config.n_samples,
                config.initial_splits,
                config.n_features,
                rng,
            );
        }

        // Phase 2: Growth phase - update each particle tree
        for tree in forest.trees.iter_mut() {
            if rng.random::<f64>() < config.growth_probability {
                tree.add_random_split(rng, config.n_features);
            }
        }

        // Phase 3: Weighting phase - compute particle weights
        let weights: Vec<f64> = forest
            .trees()
            .iter()
            .map(|tree| {
                // Simulate log-likelihood computation with realistic complexity
                let base_likelihood = rng.random::<f64>() * 5.0;
                let complexity_penalty = (tree.num_leaves() as f64).ln() * 0.1;
                let tree_contribution = tree.num_splits() as f64 * rng.random::<f64>() * 0.01;
                base_likelihood - complexity_penalty + tree_contribution
            })
            .collect();

        for weight in weights {
            forest.add_weight(weight);
        }

        // Phase 4: Normalize weights
        let normalized_weights = normalize_weights(forest.weights());

        // Phase 5: Resampling phase
        let resample_indices = SystematicResampling::resample(rng, &normalized_weights);
        forest.resample_trees(&resample_indices).unwrap();
        total_resample_operations += 1;

        // Accumulate complexity metrics
        total_complexity += forest
            .trees()
            .iter()
            .map(|t| t.num_leaves() as f64 + t.num_splits() as f64)
            .sum::<f64>();

        // Simulate occasional memory pressure
        if iteration % 10 == 0 {
            // Force some additional allocations to test memory management
            let _temp_forest = Forest::<MAX_DEPTH>::new(&arena, config.n_particles / 4);
        }
    }

    (total_complexity, total_resample_operations)
}

// Comprehensive particle sampler simulation without arena allocation
fn simulate_bart_particle_sampler_non_arena(
    config: &BenchmarkConfig,
    rng: &mut SmallRng,
) -> (f64, usize) {
    let mut total_complexity = 0.0;
    let mut total_resample_operations = 0;

    for iteration in 0..config.n_iterations {
        let mut forest = non_arena::Forest::new(config.n_particles, config.max_depth);

        // Phase 1: Initialize particles with random trees
        for _ in 0..config.n_particles {
            forest.plant_random_tree(
                0.0, // Mean-centered initialization
                config.n_samples,
                config.initial_splits,
                config.n_features,
                rng,
            );
        }

        // Phase 2: Growth phase - update each particle tree
        forest.update_all_trees(rng, config.n_features, config.growth_probability);

        // Phase 3: Weighting phase - compute particle weights
        forest.compute_weights(rng);

        // Phase 4: Normalize weights
        let normalized_weights = normalize_weights(forest.weights());

        // Phase 5: Resampling phase
        let resample_indices = SystematicResampling::resample(rng, &normalized_weights);
        forest.resample_trees(&resample_indices).unwrap();
        total_resample_operations += 1;

        // Accumulate complexity metrics
        total_complexity += forest
            .trees()
            .iter()
            .map(|t| t.num_leaves() as f64 + t.num_splits() as f64)
            .sum::<f64>();

        // Simulate occasional memory pressure
        if iteration % 10 == 0 {
            // Force some additional allocations to test memory management
            let mut _temp_forest = non_arena::Forest::new(config.n_particles / 4, config.max_depth);
            for _ in 0..(config.n_particles / 4) {
                _temp_forest.plant_tree(0.0, config.n_samples);
            }
        }
    }

    (total_complexity, total_resample_operations)
}

// Individual phase benchmarks
fn benchmark_tree_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_initialization");
    let configs = [
        ("small", BenchmarkConfig::small()),
        ("medium", BenchmarkConfig::medium()),
        ("large", BenchmarkConfig::large()),
    ];

    for (name, config) in configs {
        let throughput = Throughput::Elements(config.n_particles as u64);
        group.throughput(throughput);

        // Arena version
        group.bench_function(BenchmarkId::new("arena", name), |b| {
            b.iter(|| {
                let arena = Bump::new();
                let mut forest = Forest::<10>::new(&arena, config.n_particles);
                let mut rng = SmallRng::seed_from_u64(42);

                for _ in 0..config.n_particles {
                    forest.plant_random_tree(
                        0.0,
                        config.n_samples,
                        config.initial_splits,
                        config.n_features,
                        &mut rng,
                    );
                }

                black_box(forest.len())
            });
        });

        // Non-arena version
        group.bench_function(BenchmarkId::new("non_arena", name), |b| {
            b.iter(|| {
                let mut forest = non_arena::Forest::new(config.n_particles, config.max_depth);
                let mut rng = SmallRng::seed_from_u64(42);

                for _ in 0..config.n_particles {
                    forest.plant_random_tree(
                        0.0,
                        config.n_samples,
                        config.initial_splits,
                        config.n_features,
                        &mut rng,
                    );
                }

                black_box(forest.len())
            });
        });
    }

    group.finish();
}

fn benchmark_tree_growth_phase(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_growth_phase");
    let growth_intensities = [
        ("low_growth", 0.2),
        ("medium_growth", 0.5),
        ("high_growth", 0.8),
    ];

    for (intensity_name, growth_prob) in growth_intensities {
        let config = BenchmarkConfig {
            growth_probability: growth_prob,
            ..BenchmarkConfig::medium()
        };

        // Arena version
        group.bench_function(BenchmarkId::new("arena", intensity_name), |b| {
            b.iter(|| {
                let arena = Bump::new();
                let mut forest = Forest::<10>::new(&arena, config.n_particles);
                let mut rng = SmallRng::seed_from_u64(42);

                // Initialize trees
                for _ in 0..config.n_particles {
                    forest.plant_tree(0.0, config.n_samples);
                }

                // Growth phase
                for tree in forest.trees.iter_mut() {
                    if rng.random::<f64>() < config.growth_probability {
                        tree.add_random_split(&mut rng, config.n_features);
                    }
                }

                black_box(forest.trees().iter().map(|t| t.num_splits()).sum::<usize>())
            });
        });

        // Non-arena version
        group.bench_function(BenchmarkId::new("non_arena", intensity_name), |b| {
            b.iter(|| {
                let mut forest = non_arena::Forest::new(config.n_particles, config.max_depth);
                let mut rng = SmallRng::seed_from_u64(42);

                // Initialize trees
                for _ in 0..config.n_particles {
                    forest.plant_tree(0.0, config.n_samples);
                }

                // Growth phase
                forest.update_all_trees(&mut rng, config.n_features, config.growth_probability);

                black_box(forest.trees().iter().map(|t| t.num_splits()).sum::<usize>())
            });
        });
    }

    group.finish();
}

fn benchmark_resampling_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampling_performance");
    let particle_counts = [25, 50, 100, 200];

    for n_particles in particle_counts {
        let config = BenchmarkConfig {
            n_particles,
            ..BenchmarkConfig::medium()
        };

        // Arena resampling
        group.bench_function(BenchmarkId::new("arena", n_particles), |b| {
            b.iter(|| {
                let arena = Bump::new();
                let mut forest = Forest::<10>::new(&arena, config.n_particles);
                let mut rng = SmallRng::seed_from_u64(42);

                // Setup forest with realistic trees
                for _ in 0..config.n_particles {
                    forest.plant_random_tree(
                        0.0,
                        config.n_samples,
                        config.initial_splits,
                        config.n_features,
                        &mut rng,
                    );
                    forest.add_weight(rng.random::<f64>() * 10.0);
                }

                let normalized_weights = normalize_weights(forest.weights());
                let indices = SystematicResampling::resample(&mut rng, &normalized_weights);

                forest.resample_trees(&indices).unwrap();
                black_box(forest.len())
            });
        });

        // Non-arena resampling
        group.bench_function(BenchmarkId::new("non_arena", n_particles), |b| {
            b.iter(|| {
                let mut forest = non_arena::Forest::new(config.n_particles, config.max_depth);
                let mut rng = SmallRng::seed_from_u64(42);

                // Setup forest with realistic trees
                for _ in 0..config.n_particles {
                    forest.plant_random_tree(
                        0.0,
                        config.n_samples,
                        config.initial_splits,
                        config.n_features,
                        &mut rng,
                    );
                }
                forest.compute_weights(&mut rng);

                let normalized_weights = normalize_weights(forest.weights());
                let indices = SystematicResampling::resample(&mut rng, &normalized_weights);

                forest.resample_trees(&indices).unwrap();
                black_box(forest.len())
            });
        });
    }

    group.finish();
}

fn benchmark_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation_patterns");

    // Test different allocation patterns
    let patterns = [
        ("frequent_small", 100, 10),   // Many small allocations
        ("infrequent_large", 10, 100), // Few large allocations
        ("balanced", 50, 50),          // Balanced approach
    ];

    for (pattern_name, iterations, particles_per_iteration) in patterns {
        // Arena pattern
        group.bench_function(BenchmarkId::new("arena", pattern_name), |b| {
            b.iter(|| {
                let arena = Bump::new();
                let mut rng = SmallRng::seed_from_u64(42);
                let mut total_trees = 0;

                for _ in 0..iterations {
                    let mut forest = Forest::<8>::new(&arena, particles_per_iteration);

                    for _ in 0..particles_per_iteration {
                        forest.plant_random_tree(0.0, 1000, 5, 20, &mut rng);
                    }

                    total_trees += forest.len();

                    // Simulate some operations
                    for _ in 0..5 {
                        forest.add_weight(rng.random());
                    }

                    if forest.len() > 5 {
                        let weights = normalize_weights(forest.weights());
                        let indices = SystematicResampling::resample(&mut rng, &weights);
                        forest.resample_trees(&indices).unwrap();
                    }
                }

                black_box(total_trees)
            });
        });

        // Non-arena pattern
        group.bench_function(BenchmarkId::new("non_arena", pattern_name), |b| {
            b.iter(|| {
                let mut rng = SmallRng::seed_from_u64(42);
                let mut total_trees = 0;

                for _ in 0..iterations {
                    let mut forest = non_arena::Forest::new(particles_per_iteration, 8);

                    for _ in 0..particles_per_iteration {
                        forest.plant_random_tree(0.0, 1000, 5, 20, &mut rng);
                    }

                    total_trees += forest.len();

                    // Simulate some operations
                    forest.compute_weights(&mut rng);

                    if forest.len() > 5 {
                        let weights = normalize_weights(forest.weights());
                        let indices = SystematicResampling::resample(&mut rng, &weights);
                        forest.resample_trees(&indices).unwrap();
                    }
                }

                black_box(total_trees)
            });
        });
    }

    group.finish();
}

fn benchmark_full_particle_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_particle_sampler");

    let configs = [
        ("small", BenchmarkConfig::small()),
        ("medium", BenchmarkConfig::medium()),
        ("large", BenchmarkConfig::large()),
    ];

    for (config_name, config) in configs {
        let throughput = Throughput::Elements((config.n_particles * config.n_iterations) as u64);
        group.throughput(throughput);

        // Full arena-based sampler
        group.bench_function(BenchmarkId::new("arena_full", config_name), |b| {
            b.iter(|| {
                let mut rng = SmallRng::seed_from_u64(42);
                black_box(simulate_bart_particle_sampler_arena::<10>(
                    &config, &mut rng,
                ))
            });
        });

        // Full non-arena sampler
        group.bench_function(BenchmarkId::new("non_arena_full", config_name), |b| {
            b.iter(|| {
                let mut rng = SmallRng::seed_from_u64(42);
                black_box(simulate_bart_particle_sampler_non_arena(&config, &mut rng))
            });
        });
    }

    group.finish();
}

fn benchmark_memory_pressure_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure");

    // Simulate different memory pressure scenarios
    let scenarios = [
        ("low_pressure", 50, 1000, 5),
        ("medium_pressure", 100, 5000, 10),
        ("high_pressure", 200, 20000, 20),
    ];

    for (scenario_name, n_particles, n_samples, tree_depth) in scenarios {
        let config = BenchmarkConfig {
            n_particles,
            n_samples,
            max_depth: tree_depth,
            n_iterations: 25,
            ..BenchmarkConfig::medium()
        };

        // Arena under memory pressure
        group.bench_function(BenchmarkId::new("arena", scenario_name), |b| {
            b.iter(|| {
                let arena = Bump::new();
                let mut rng = SmallRng::seed_from_u64(42);
                let mut total_operations = 0;

                // Create multiple forests to simulate memory pressure
                for _ in 0..5 {
                    let mut forest = Forest::<15>::new(&arena, config.n_particles);

                    for _ in 0..config.n_particles {
                        forest.plant_random_tree(
                            0.0,
                            config.n_samples,
                            config.initial_splits,
                            config.n_features,
                            &mut rng,
                        );
                    }

                    // Perform multiple resampling operations
                    for _ in 0..10 {
                        let weights: Vec<f64> = forest
                            .trees()
                            .iter()
                            .map(|tree| rng.random::<f64>() * tree.num_leaves() as f64)
                            .collect();

                        for weight in weights {
                            forest.add_weight(weight);
                        }

                        let weights = normalize_weights(forest.weights());
                        let indices = SystematicResampling::resample(&mut rng, &weights);
                        forest.resample_trees(&indices).unwrap();
                        total_operations += 1;

                        forest.weights.clear(); // Reset for next iteration
                    }
                }

                black_box(total_operations)
            });
        });

        // Non-arena under memory pressure
        group.bench_function(BenchmarkId::new("non_arena", scenario_name), |b| {
            b.iter(|| {
                let mut rng = SmallRng::seed_from_u64(42);
                let mut total_operations = 0;

                // Create multiple forests to simulate memory pressure
                for _ in 0..5 {
                    let mut forest = non_arena::Forest::new(config.n_particles, config.max_depth);

                    for _ in 0..config.n_particles {
                        forest.plant_random_tree(
                            0.0,
                            config.n_samples,
                            config.initial_splits,
                            config.n_features,
                            &mut rng,
                        );
                    }

                    // Perform multiple resampling operations
                    for _ in 0..10 {
                        forest.compute_weights(&mut rng);

                        let weights = normalize_weights(forest.weights());
                        let indices = SystematicResampling::resample(&mut rng, &weights);
                        forest.resample_trees(&indices).unwrap();
                        total_operations += 1;
                    }
                }

                black_box(total_operations)
            });
        });
    }

    group.finish();
}

fn benchmark_tree_cloning_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_cloning_efficiency");

    let tree_sizes = [
        ("tiny", 2, 100, 3),
        ("small", 8, 1000, 5),
        ("medium", 20, 5000, 8),
        ("large", 50, 20000, 12),
    ];

    for (size_name, n_splits, n_samples, max_depth) in tree_sizes {
        // Arena cloning with realistic trees
        group.bench_function(BenchmarkId::new("arena_clone", size_name), |b| {
            let arena = Bump::new();
            let mut rng = SmallRng::seed_from_u64(42);

            // Pre-build a complex tree
            let mut source_tree = Tree::<15>::stump(&arena, 0.0, n_samples);
            for _ in 0..n_splits {
                source_tree.add_random_split(&mut rng, 25);
            }

            b.iter(|| {
                // Clone the tree multiple times to simulate resampling
                let mut cloned_trees = Vec::new();
                for _ in 0..10 {
                    cloned_trees.push(source_tree.clone_into(&arena));
                }
                black_box(cloned_trees.len())
            });
        });

        // Non-arena cloning
        group.bench_function(BenchmarkId::new("non_arena_clone", size_name), |b| {
            let mut rng = SmallRng::seed_from_u64(42);

            // Pre-build a complex tree
            let mut source_tree = non_arena::Tree::stump(0.0, n_samples, max_depth);
            for _ in 0..n_splits {
                source_tree.add_random_split(&mut rng, 25);
            }

            b.iter(|| {
                // Clone the tree multiple times to simulate resampling
                let mut cloned_trees = Vec::new();
                for _ in 0..10 {
                    cloned_trees.push(source_tree.clone());
                }
                black_box(cloned_trees.len())
            });
        });
    }

    group.finish();
}

criterion_group!(
    comprehensive_benches,
    benchmark_tree_initialization,
    benchmark_tree_growth_phase,
    benchmark_resampling_performance,
    benchmark_memory_allocation_patterns,
    benchmark_full_particle_sampler,
    benchmark_memory_pressure_scenarios,
    benchmark_tree_cloning_efficiency
);
criterion_main!(comprehensive_benches);
