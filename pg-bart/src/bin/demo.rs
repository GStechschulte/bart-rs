use rand::seq::IteratorRandom;
use rand::thread_rng;

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

    let mut particles = vec![
        Particles {
            position_: 0.0,
            velocity: 1.0,
        },
        Particles {
            position_: 1.0,
            velocity: 2.0,
        },
    ];

    update_particles(&mut particles);
}

// struct P {
//     val: f64,
// }

// struct State {
//     particles: Vec<P>,
// }

// impl State {
//     fn step(&mut self) {
//         for p in &mut self.particles {
//             p += 1.0
//         }
//     }
// }

struct Particles {
    position_: f64,
    velocity: f64,
}

fn update_particles(particles: &mut Vec<Particles>) {
    for particle in particles.iter_mut() {
        particle.position_ += particle.velocity;
    }
}
