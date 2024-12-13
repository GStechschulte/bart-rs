use pg_bart::particle::{Particle, SampleIndices, Weight};
use pg_bart::pgbart::{normalize_weights, resample_particles, select_particle};
use pg_bart::tree::DecisionTree;

fn create_test_particles() -> Vec<Particle> {
    vec![
        Particle {
            tree: DecisionTree::new(0.0),
            indices: SampleIndices::new(10),
            weight: Weight {
                log_w: 0.0,
                log_likelihood: 0.0,
            },
        },
        Particle {
            tree: DecisionTree::new(0.0),
            indices: SampleIndices::new(10),
            weight: Weight {
                log_w: 1.0,
                log_likelihood: 1.0,
            },
        },
        Particle {
            tree: DecisionTree::new(0.0),
            indices: SampleIndices::new(10),
            weight: Weight {
                log_w: 2.0,
                log_likelihood: 2.0,
            },
        },
    ]
}

fn main() {
    let mut particles = create_test_particles();
    let normalized_weights = normalize_weights(&particles);

    let selected_particles = select_particle(&mut particles, &normalized_weights);
    // println!("{:#?}", particles);

    // Assert that the selected particle is one of the original particles
    assert!(
        particles.iter().any(|p| p.tree == selected_particles.tree
            && p.indices == selected_particles.indices
            && p.weight == selected_particles.weight),
        "Selected particle is not valid"
    );
}
