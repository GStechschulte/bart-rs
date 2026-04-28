use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use numpy::ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use pymc_bart::config::BartConfig;
use pymc_bart::data::OwnedData;
use pymc_bart::kernel::{BartKernel, SamplingAlgorithm};
use pymc_bart::resampling::{
    MultinomialResampling, ResamplingStrategy, StratifiedResampling, SystematicResampling,
};
use pymc_bart::smc::{normalize_weights_inplace, smc_step};
use pymc_bart::splitting::SplitRules;
use pymc_bart::tree::TreeArrays;
use pymc_bart::weight::WeightFn;

/// Mock weight function
struct GaussianLogLik {
    targets: Array1<f64>,
}

impl WeightFn for GaussianLogLik {
    fn log_weight(&self, predictions: &Array1<f64>) -> f64 {
        predictions
            .iter()
            .zip(self.targets.iter())
            .map(|(&p, &t)| {
                let r = p - t;
                -0.5 * r * r
            })
            .sum()
    }
}

fn make_synthetic_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = SmallRng::seed_from_u64(123);
    let x = Array2::from_shape_fn((n_samples, n_features), |_| {
        use rand::Rng;
        rng.random::<f64>() * 10.0
    });
    let y = Array1::from_shape_fn(n_samples, |i| {
        x[[i, 0]] * 0.5 + x[[i, 1.min(n_features - 1)]] * 0.3 + 0.1
    });
    (x, y)
}

fn make_split_tree(n_samples: usize, n_features: usize) -> (TreeArrays, Array2<f64>) {
    let (x, _) = make_synthetic_data(n_samples, n_features);
    let mut tree = TreeArrays::new(0.0, n_samples, 6);

    // Split root on feature 0 at 5.0
    tree.split_node(0, 0, 5.0, -1.0, 1.0);
    tree.update_leaf_assignments(0, 0, 5.0, &(0..n_samples).collect::<Vec<_>>(), x.view());

    // Split left child on feature 1 at 3.0
    let left_samples: Vec<usize> = tree.get_leaf_samples(1).collect();
    tree.split_node(1, 1, 3.0, -0.5, 0.5);
    tree.update_leaf_assignments(1, 1, 3.0, &left_samples, x.view());

    // Split right child on feature 0 at 8.0
    let right_samples: Vec<usize> = tree.get_leaf_samples(2).collect();
    tree.split_node(2, 0, 8.0, 0.5, 1.5);
    tree.update_leaf_assignments(2, 0, 8.0, &right_samples, x.view());

    (tree, x)
}

fn make_normalized_weights(n: usize) -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut w: Vec<f64> = (0..n)
        .map(|_| {
            use rand::Rng;
            rng.random::<f64>()
        })
        .collect();
    let sum: f64 = w.iter().sum();
    for v in &mut w {
        *v /= sum;
    }
    w
}

fn make_log_weights(n: usize) -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..n)
        .map(|_| {
            use rand::Rng;
            -rng.random::<f64>() * 100.0
        })
        .collect()
}

fn bench_smc_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("smc_step");
    group.sample_size(20);

    for &n_samples in &[100, 500, 1000] {
        for &n_particles in &[10, 20, 40] {
            let n_features = 5;
            let (x, y) = make_synthetic_data(n_samples, n_features);
            let data = OwnedData::new(x, y.clone());
            let data_view = data.view();
            let residuals = y.clone();
            let weight_fn = GaussianLogLik { targets: y };
            let config = BartConfig {
                n_particles,
                ..Default::default()
            };
            let split_rules: Vec<SplitRules> =
                vec![SplitRules::from_name("ContinuousSplit").unwrap(); n_features];
            let resampling = SystematicResampling;

            let param = format!("s{n_samples}_p{n_particles}");
            group.bench_with_input(BenchmarkId::new("systematic", &param), &param, |b, _| {
                let mut rng = SmallRng::seed_from_u64(42);
                b.iter(|| {
                    smc_step(
                        &mut rng,
                        &residuals,
                        &config,
                        &data_view,
                        &split_rules,
                        &resampling,
                        &weight_fn,
                    )
                });
            });
        }
    }

    group.finish();
}

fn bench_predict_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict_training");

    for &n_samples in &[100, 1000, 10000] {
        let (tree, _x) = make_split_tree(n_samples, 5);

        group.bench_with_input(
            BenchmarkId::new("collect", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| tree.predict_training());
            },
        );

        let mut out = Array1::zeros(n_samples);
        group.bench_with_input(
            BenchmarkId::new("into_buf", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| tree.predict_training_into(&mut out));
            },
        );
    }

    group.finish();
}

fn bench_resampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampling");

    for &n in &[10, 50, 100, 500] {
        let weights = make_normalized_weights(n);

        group.bench_with_input(BenchmarkId::new("systematic", n), &n, |b, _| {
            let mut rng = SmallRng::seed_from_u64(42);
            let r = SystematicResampling;
            b.iter(|| r.resample(&mut rng, &weights));
        });

        group.bench_with_input(BenchmarkId::new("multinomial", n), &n, |b, _| {
            let mut rng = SmallRng::seed_from_u64(42);
            let r = MultinomialResampling;
            b.iter(|| r.resample(&mut rng, &weights));
        });

        group.bench_with_input(BenchmarkId::new("stratified", n), &n, |b, _| {
            let mut rng = SmallRng::seed_from_u64(42);
            let r = StratifiedResampling;
            b.iter(|| r.resample(&mut rng, &weights));
        });
    }

    group.finish();
}

fn bench_normalize_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_weights");

    for &n in &[10, 50, 100, 500] {
        let log_weights = make_log_weights(n);

        group.bench_with_input(BenchmarkId::new("logsumexp", n), &n, |b, _| {
            let mut w = log_weights.clone();
            b.iter(|| {
                w.copy_from_slice(&log_weights);
                normalize_weights_inplace(&mut w);
            });
        });
    }

    group.finish();
}

fn bench_tree_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_operations");

    for &n_samples in &[100, 1000, 10000] {
        let (x, _) = make_synthetic_data(n_samples, 5);

        // Benchmark split_node + update_leaf_assignments
        group.bench_with_input(
            BenchmarkId::new("split_and_assign", n_samples),
            &n_samples,
            |b, _| {
                b.iter_batched(
                    || {
                        let tree = TreeArrays::new(0.0, n_samples, 6);
                        let affected: Vec<usize> = (0..n_samples).collect();
                        (tree, affected)
                    },
                    |(mut tree, affected)| {
                        tree.split_node(0, 0, 5.0, -1.0, 1.0);
                        tree.update_leaf_assignments(0, 0, 5.0, &affected, x.view());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Benchmark predict_batch_test (tree traversal)
        let (tree, test_data) = make_split_tree(n_samples, 5);
        group.bench_with_input(
            BenchmarkId::new("predict_batch_test", n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| tree.predict_batch_test(&test_data));
            },
        );
    }

    group.finish();
}

fn bench_kernel_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_step");
    group.sample_size(10);

    for &n_trees in &[10, 50] {
        for &n_samples in &[100, 500] {
            let n_features = 5;
            let (x, y) = make_synthetic_data(n_samples, n_features);
            let data = OwnedData::new(x, y.clone());
            let weight_fn = GaussianLogLik { targets: y };
            let config = BartConfig {
                n_trees,
                n_particles: 10,
                ..Default::default()
            };
            let split_rules: Vec<SplitRules> =
                vec![SplitRules::from_name("ContinuousSplit").unwrap(); n_features];

            let kernel = BartKernel {
                split_rules,
                resampling: SystematicResampling,
                weight_fn,
                config,
                data,
            };

            let param = format!("t{n_trees}_s{n_samples}");
            group.bench_with_input(BenchmarkId::new("full_step", &param), &param, |b, _| {
                let mut rng = SmallRng::seed_from_u64(42);
                let state = kernel.init(&mut rng);
                b.iter_batched(
                    || state.clone(),
                    |s| kernel.step(&mut rng, s),
                    criterion::BatchSize::LargeInput,
                );
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_smc_step,
    bench_predict_training,
    bench_resampling,
    bench_normalize_weights,
    bench_tree_operations,
    bench_kernel_step,
);
criterion_main!(benches);
