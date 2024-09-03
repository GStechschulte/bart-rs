use pg_bart::particle::Particle;
use pg_bart::{particle::ParticleParams, tree::DecisionTree};

fn main() {
    let m = 10 as f64;
    let leaf_value = 0.5;

    let particles: Vec<Particle> = (0..10)
        .map(|_| {
            let p_params = ParticleParams::new(100 as usize, 2 as usize, 0.5 as f64);
            Particle::new(p_params, leaf_value, 100 as usize)
        })
        .collect();

    println!("{:?}", particles.len());
}
