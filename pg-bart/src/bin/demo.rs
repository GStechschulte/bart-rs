use rand::seq::IteratorRandom;
use rand::thread_rng;
use rand::{self, Rng};

use ndarray::Array2;

use pg_bart::particle::Particle;
use pg_bart::{particle::ParticleParams, tree::DecisionTree};

fn main() {
    let m = 20 as f64;
    let leaf_value = 0.5;

    let particles: Vec<Particle> = (0..m as usize)
        .map(|_| {
            let p_params = ParticleParams::new(100 as usize, 2 as usize, 0.5 as f64);
            Particle::new(p_params, leaf_value, 100 as usize)
        })
        .collect();

    println!("Num particles: {:?}", particles.len());

    let tune = true;
    let fraction = if tune { 0.1 } else { 0.1 };

    // The number of trees to update in a step...
    let num_to_update = ((particles.len() as f64) * fraction).floor() as usize;
    println!("Num particles to update at a time: {}", num_to_update);

    // Randomly sample a set of Particle indices that we will be modifying...
    let mut rng = thread_rng();

    let indices: Vec<usize> = (0..m as usize).choose_multiple(&mut rng, num_to_update);
    println!("Indices to update: {:?}", indices);

    let indices_2 = rand::seq::index::sample(&mut rng, m as usize, num_to_update);
    println!("Indices to update: {:?}", indices_2);

    println!("---------------------------");

    // PyMC implementation
    let tune = true;
    let m = 20 as f64;
    let batch_fraction = (0.1, 0.1);

    let batch = ((m * batch_fraction.0).ceil(), (m * batch_fraction.1).ceil());

    println!("batch: {:?}", batch);

    let batch_size = if tune { batch.0 } else { batch.1 };
    let lower = 0 as usize;
    let upper = (lower as f64 + batch_size).floor() as usize;

    let tree_ids = lower..upper;

    let lower = if upper >= m as usize {
        0 as usize
    } else {
        upper
    };

    println!(
        "lower: {}, upper: {}, tree_ids: {:?}",
        lower, upper, tree_ids
    );

    for (iter, tree_id) in tree_ids.enumerate() {
        println!("iter: {}, tree_id: {}", iter, tree_id);
    }

    println!("------------------------");

    let X = Array2::from_shape_vec(
        (10, 3),
        vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
        ],
    )
    .unwrap();

    let data_indices: Vec<Vec<usize>> = vec![Vec::from_iter(0..X.nrows())];
    let samples = &data_indices[0];
    let feature = 2 as usize;

    println!("X = {}", X);
    println!("data_indices: {:?}", data_indices);
    println!("Using samples: {:?}", samples);
    println!("Splitting on feature: {}", feature);

    let feature_values: Vec<f64> = samples.iter().map(|&i| X[[i, feature]]).collect();

    println!("Routed feature values: {:?}", feature_values);

    let mut rng = rand::thread_rng();

    let alpha = 0.95;
    let depth = 1 as f64;
    let beta = 2.0;

    // let p = 1. - (alpha * ((1 + depth).pow(-beta as u32)) as f64);
    let p = 1. - alpha * (1.0 + depth).powf(-beta);
    let res = p < rng.gen::<f64>();
    println!("p: {}, res: {}", p, res);
}

struct SampleIndices {
    indicies: Vec<Vec<usize>>,
}
