use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use bumpalo::Bump;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};

// Import implementations
use pymc_bart::forest::Forest;
use pymc_bart::resampling::{ResamplingStrategy, SystematicResampling};
use pymc_bart::sampler::normalize_weights;

// Memory tracking allocator
struct TrackingAllocator {
    allocated: AtomicUsize,
    deallocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl TrackingAllocator {
    const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            deallocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.allocated.store(0, Ordering::SeqCst);
        self.deallocated.store(0, Ordering::SeqCst);
        self.peak_allocated.store(0, Ordering::SeqCst);
        self.allocation_count.store(0, Ordering::SeqCst);
    }

    fn current_allocated(&self) -> usize {
        self.allocated.load(Ordering::SeqCst) - self.deallocated.load(Ordering::SeqCst)
    }

    fn peak_allocated(&self) -> usize {
        self.peak_allocated.load(Ordering::SeqCst)
    }

    fn total_allocations(&self) -> usize {
        self.allocation_count.load(Ordering::SeqCst)
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            self.allocated.fetch_add(size, Ordering::SeqCst);
            self.allocation_count.fetch_add(1, Ordering::SeqCst);

            // Update peak if necessary
            let current = self.current_allocated();
            let mut peak = self.peak_allocated.load(Ordering::SeqCst);
            while current > peak {
                match self.peak_allocated.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(x) => peak = x,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        self.deallocated.fetch_add(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator::new();

// Non-arena implementation for comparison
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

            let split_var = rng.random_range(0..n_features);
            let split_value = rng.random::<f64>();

            self.split_var.push(split_var);
            self.split_value.push(split_value);

            let original_leaf_value = self.leaf_values[0];
            self.leaf_values
                .push(original_leaf_value + rng.random::<f64>() * 0.1);
            self.leaf_values
                .push(original_leaf_value - rng.random::<f64>() * 0.1);
        }

        pub fn num_splits(&self) -> usize {
            self.split_var.len()
        }

        pub fn num_leaves(&self) -> usize {
            self.leaf_values.len()
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

        pub fn add_weight(&mut self, weight: f64) {
            self.weights.push(weight);
        }

        pub fn len(&self) -> usize {
            self.trees.len()
        }

        pub fn trees(&self) -> &[Tree] {
            &self.trees
        }

        pub fn weights(&self) -> &[f64] {
            &self.weights
        }

        pub fn resample_trees(&mut self, indices: &[usize]) -> Result<(), &'static str> {
            if indices.len() != self.trees.len() {
                return Err("Number of indices must match number of trees");
            }

            let mut new_trees = Vec::with_capacity(indices.len());

            for &idx in indices {
                if idx >= self.trees.len() {
                    return Err("Invalid tree index");
                }
                new_trees.push(self.trees[idx].clone());
            }

            self.trees = new_trees;
            Ok(())
        }

        pub fn clear(&mut self) {
            self.trees.clear();
            self.weights.clear();
        }
    }
}

#[derive(Debug, Clone)]
struct MemoryStats {
    peak_memory: usize,
    total_allocations: usize,
    final_memory: usize,
    duration: Duration,
}

impl MemoryStats {
    fn new() -> Self {
        Self {
            peak_memory: 0,
            total_allocations: 0,
            final_memory: 0,
            duration: Duration::from_nanos(0),
        }
    }
}

// Configuration for memory analysis
#[derive(Clone, Debug)]
struct MemoryBenchConfig {
    n_particles: usize,
    n_samples: usize,
    n_features: usize,
    n_iterations: usize,
    initial_splits: usize,
    max_depth: usize,
}

impl MemoryBenchConfig {
    fn small() -> Self {
        Self {
            n_particles: 50,
            n_samples: 1000,
            n_features: 10,
            n_iterations: 20,
            initial_splits: 5,
            max_depth: 8,
        }
    }

    fn medium() -> Self {
        Self {
            n_particles: 100,
            n_samples: 5000,
            n_features: 25,
            n_iterations: 20,
            initial_splits: 10,
            max_depth: 10,
        }
    }

    fn large() -> Self {
        Self {
            n_particles: 200,
            n_samples: 20000,
            n_features: 50,
            n_iterations: 10,
            initial_splits: 15,
            max_depth: 12,
        }
    }
}

// Run memory analysis for arena implementation
fn analyze_arena_memory<const MAX_DEPTH: usize>(
    config: &MemoryBenchConfig,
    rng: &mut SmallRng,
) -> MemoryStats {
    GLOBAL_ALLOCATOR.reset();
    let start_time = Instant::now();

    let arena = Bump::new();

    for _ in 0..config.n_iterations {
        let mut forest = Forest::<MAX_DEPTH>::new(&arena, config.n_particles);

        // Initialize particle trees
        for _ in 0..config.n_particles {
            forest.plant_random_tree(
                0.0,
                config.n_samples,
                config.initial_splits,
                config.n_features,
                rng,
            );
        }

        // Growth phase
        // Phase 2: Growth phase - update each particle tree
        for tree in forest.trees.iter_mut() {
            if rng.random::<f64>() < 0.4 {
                tree.add_random_split(rng, config.n_features);
            }
        }

        // Phase 3: Weighting phase - compute particle weights
        let weights: Vec<f64> = forest
            .trees()
            .iter()
            .map(|tree| rng.random::<f64>() * 10.0 - tree.num_leaves() as f64 * 0.01)
            .collect();

        for weight in weights {
            forest.add_weight(weight);
        }

        // Resampling phase
        let normalized_weights = normalize_weights(forest.weights());
        let indices = SystematicResampling::resample(rng, &normalized_weights);
        forest.resample_trees(&indices).unwrap();

        // Clear for next iteration
        forest.clear();
    }

    let duration = start_time.elapsed();

    MemoryStats {
        peak_memory: GLOBAL_ALLOCATOR.peak_allocated(),
        total_allocations: GLOBAL_ALLOCATOR.total_allocations(),
        final_memory: GLOBAL_ALLOCATOR.current_allocated(),
        duration,
    }
}

// Run memory analysis for non-arena implementation
fn analyze_non_arena_memory(config: &MemoryBenchConfig, rng: &mut SmallRng) -> MemoryStats {
    GLOBAL_ALLOCATOR.reset();
    let start_time = Instant::now();

    for _ in 0..config.n_iterations {
        let mut forest = non_arena::Forest::new(config.n_particles, config.max_depth);

        // Initialize particle trees
        for _ in 0..config.n_particles {
            forest.plant_random_tree(
                0.0,
                config.n_samples,
                config.initial_splits,
                config.n_features,
                rng,
            );
        }

        // Growth phase
        // Phase 2: Growth phase
        for tree in forest.trees.iter_mut() {
            if rng.random::<f64>() < 0.4 {
                tree.add_random_split(rng, config.n_features);
            }
        }

        // Phase 3: Weighting phase
        let weights: Vec<f64> = forest
            .trees()
            .iter()
            .map(|tree| rng.random::<f64>() * 10.0 - tree.num_leaves() as f64 * 0.01)
            .collect();

        for weight in weights {
            forest.add_weight(weight);
        }

        // Resampling phase
        let normalized_weights = normalize_weights(forest.weights());
        let indices = SystematicResampling::resample(rng, &normalized_weights);
        forest.resample_trees(&indices).unwrap();

        // Clear for next iteration
        forest.clear();
    }

    let duration = start_time.elapsed();

    MemoryStats {
        peak_memory: GLOBAL_ALLOCATOR.peak_allocated(),
        total_allocations: GLOBAL_ALLOCATOR.total_allocations(),
        final_memory: GLOBAL_ALLOCATOR.current_allocated(),
        duration,
    }
}

fn benchmark_memory_usage_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_analysis");
    group.sample_size(10); // Fewer samples for detailed memory analysis

    let configs = [
        ("small", MemoryBenchConfig::small()),
        ("medium", MemoryBenchConfig::medium()),
        ("large", MemoryBenchConfig::large()),
    ];

    for (config_name, config) in configs {
        // Arena memory analysis
        group.bench_function(BenchmarkId::new("arena_memory", config_name), |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::from_nanos(0);
                let mut rng = SmallRng::seed_from_u64(42);

                for _ in 0..iters {
                    let stats = analyze_arena_memory::<10>(&config, &mut rng);
                    total_duration += stats.duration;

                    // Print detailed memory statistics
                    eprintln!(
                        "Arena {} - Peak: {} bytes, Allocations: {}, Final: {} bytes, Duration: {:?}",
                        config_name,
                        stats.peak_memory,
                        stats.total_allocations,
                        stats.final_memory,
                        stats.duration
                    );
                }

                total_duration
            });
        });

        // Non-arena memory analysis
        group.bench_function(BenchmarkId::new("non_arena_memory", config_name), |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::from_nanos(0);
                let mut rng = SmallRng::seed_from_u64(42);

                for _ in 0..iters {
                    let stats = analyze_non_arena_memory(&config, &mut rng);
                    total_duration += stats.duration;

                    // Print detailed memory statistics
                    eprintln!(
                        "Non-Arena {} - Peak: {} bytes, Allocations: {}, Final: {} bytes, Duration: {:?}",
                        config_name,
                        stats.peak_memory,
                        stats.total_allocations,
                        stats.final_memory,
                        stats.duration
                    );
                }

                total_duration
            });
        });
    }

    group.finish();
}

fn benchmark_allocation_fragmentation(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_fragmentation");

    // Test allocation patterns that might cause fragmentation
    let fragmentation_tests = [("sequential", true), ("random_pattern", false)];

    for (pattern_name, sequential) in fragmentation_tests {
        // Arena fragmentation test
        group.bench_function(BenchmarkId::new("arena_frag", pattern_name), |b| {
            b.iter(|| {
                GLOBAL_ALLOCATOR.reset();
                let arena = Bump::new();
                let mut rng = SmallRng::seed_from_u64(if sequential { 42 } else { 123 });

                // Create many small forests to test fragmentation
                let mut forests = Vec::new();
                for i in 0..20 {
                    let mut forest = Forest::<8>::new(&arena, 25);

                    let seed = if sequential { i as u64 } else { rng.random() };
                    let mut local_rng = SmallRng::seed_from_u64(seed);

                    for _ in 0..25 {
                        forest.plant_random_tree(0.0, 1000, 5, 10, &mut local_rng);
                    }

                    forests.push(forest);

                    // Occasionally drop some forests to create holes
                    if i % 3 == 0 && !forests.is_empty() {
                        forests.remove(0);
                    }
                }

                black_box((forests.len(), GLOBAL_ALLOCATOR.peak_allocated()))
            });
        });

        // Non-arena fragmentation test
        group.bench_function(BenchmarkId::new("non_arena_frag", pattern_name), |b| {
            b.iter(|| {
                GLOBAL_ALLOCATOR.reset();
                let mut rng = SmallRng::seed_from_u64(if sequential { 42 } else { 123 });

                // Create many small forests to test fragmentation
                let mut forests = Vec::new();
                for i in 0..20 {
                    let mut forest = non_arena::Forest::new(25, 8);

                    let seed = if sequential { i as u64 } else { rng.random() };
                    let mut local_rng = SmallRng::seed_from_u64(seed);

                    for _ in 0..25 {
                        forest.plant_random_tree(0.0, 1000, 5, 10, &mut local_rng);
                    }

                    forests.push(forest);

                    // Occasionally drop some forests to create holes
                    if i % 3 == 0 && !forests.is_empty() {
                        forests.remove(0);
                    }
                }

                black_box((forests.len(), GLOBAL_ALLOCATOR.peak_allocated()))
            });
        });
    }

    group.finish();
}

fn benchmark_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");

    // Test different access patterns that affect cache performance
    let access_patterns = [("sequential_access", true), ("random_access", false)];

    for (pattern_name, sequential) in access_patterns {
        let config = MemoryBenchConfig::medium();

        // Arena cache test
        group.bench_function(BenchmarkId::new("arena_cache", pattern_name), |b| {
            b.iter(|| {
                let arena = Bump::new();
                let mut forest = Forest::<10>::new(&arena, config.n_particles);
                let mut rng = SmallRng::seed_from_u64(42);

                // Initialize trees
                for _ in 0..config.n_particles {
                    forest.plant_random_tree(
                        0.0,
                        config.n_samples,
                        config.initial_splits,
                        config.n_features,
                        &mut rng,
                    );
                }

                // Access trees in different patterns
                let mut total_operations = 0;
                if sequential {
                    // Sequential access - cache friendly
                    for tree in forest.trees() {
                        total_operations += tree.num_splits() + tree.num_leaves();
                    }
                } else {
                    // Random access - cache unfriendly
                    for _ in 0..config.n_particles * 2 {
                        let idx = rng.random_range(0..forest.trees().len());
                        let tree = &forest.trees()[idx];
                        total_operations += tree.num_splits() + tree.num_leaves();
                    }
                }

                black_box(total_operations)
            });
        });

        // Non-arena cache test
        group.bench_function(BenchmarkId::new("non_arena_cache", pattern_name), |b| {
            b.iter(|| {
                let mut forest = non_arena::Forest::new(config.n_particles, config.max_depth);
                let mut rng = SmallRng::seed_from_u64(42);

                // Initialize trees
                for _ in 0..config.n_particles {
                    forest.plant_random_tree(
                        0.0,
                        config.n_samples,
                        config.initial_splits,
                        config.n_features,
                        &mut rng,
                    );
                }

                // Access trees in different patterns
                let mut total_operations = 0;
                if sequential {
                    // Sequential access - cache friendly
                    for tree in forest.trees() {
                        total_operations += tree.num_splits() + tree.num_leaves();
                    }
                } else {
                    // Random access - cache unfriendly
                    for _ in 0..config.n_particles * 2 {
                        let idx = rng.random_range(0..forest.trees().len());
                        let tree = &forest.trees()[idx];
                        total_operations += tree.num_splits() + tree.num_leaves();
                    }
                }

                black_box(total_operations)
            });
        });
    }

    group.finish();
}

fn benchmark_memory_pressure_recovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure_recovery");

    // Test recovery from high memory pressure scenarios
    group.bench_function("arena_recovery", |b| {
        b.iter(|| {
            GLOBAL_ALLOCATOR.reset();

            // Phase 1: Create high memory pressure
            let arena = Bump::new();
            let mut forests = Vec::new();
            let mut rng = SmallRng::seed_from_u64(42);

            for _ in 0..10 {
                let mut forest = Forest::<12>::new(&arena, 100);
                for _ in 0..100 {
                    forest.plant_random_tree(0.0, 10000, 20, 50, &mut rng);
                }
                forests.push(forest);
            }

            let peak_pressure = GLOBAL_ALLOCATOR.peak_allocated();

            // Phase 2: Release memory by dropping forests
            forests.clear();

            // Phase 3: Test allocation performance after pressure
            let mut recovery_forest = Forest::<10>::new(&arena, 50);
            for _ in 0..50 {
                recovery_forest.plant_random_tree(0.0, 5000, 10, 25, &mut rng);
            }

            black_box((peak_pressure, recovery_forest.len()))
        });
    });

    group.bench_function("non_arena_recovery", |b| {
        b.iter(|| {
            GLOBAL_ALLOCATOR.reset();

            // Phase 1: Create high memory pressure
            let mut forests = Vec::new();
            let mut rng = SmallRng::seed_from_u64(42);

            for _ in 0..10 {
                let mut forest = non_arena::Forest::new(100, 12);
                for _ in 0..100 {
                    forest.plant_random_tree(0.0, 10000, 20, 50, &mut rng);
                }
                forests.push(forest);
            }

            let peak_pressure = GLOBAL_ALLOCATOR.peak_allocated();

            // Phase 2: Release memory by dropping forests
            forests.clear();

            // Phase 3: Test allocation performance after pressure
            let mut recovery_forest = non_arena::Forest::new(50, 10);
            for _ in 0..50 {
                recovery_forest.plant_random_tree(0.0, 5000, 10, 25, &mut rng);
            }

            black_box((peak_pressure, recovery_forest.len()))
        });
    });

    group.finish();
}

criterion_group!(
    memory_benches,
    benchmark_memory_usage_patterns,
    benchmark_allocation_fragmentation,
    benchmark_cache_performance,
    benchmark_memory_pressure_recovery
);
criterion_main!(memory_benches);
