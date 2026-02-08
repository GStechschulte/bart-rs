use pymc_bart::particle::{Particle, SampleIndices, Weight};
use pymc_bart::pgbart::{normalize_weights, resample_particles, select_particle};
use pymc_bart::tree::DecisionTree;
use rand::{rngs::StdRng, SeedableRng};

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

#[test]
fn test_normalize_weights() {
    let particles = create_test_particles();
    let normalized_weights = normalize_weights(&particles);

    // Assert that the weights sum to approximately 1
    let sum: f64 = normalized_weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Normalized weights do not sum to 1"
    );
}

#[test]
fn test_select_particle() {
    let mut particles = create_test_particles();
    let original_particles = particles.clone();
    let normalized_weights = normalize_weights(&particles);
    let mut rng = StdRng::seed_from_u64(0);
    let selected_particle = select_particle(&mut rng, &mut particles, &normalized_weights);

    // Assert that the selected particle is one of the original particles
    assert!(
        original_particles.contains(&selected_particle),
        "Selected particle is not valid"
    );
}

#[test]
fn test_resample_particles() {
    let mut particles = create_test_particles();
    let original_particles = particles.clone();
    let original_len = particles.len();
    let normalized_weights = normalize_weights(&particles);
    let mut rng = StdRng::seed_from_u64(0);

    // Resample particles based on normalized weights
    let resampled_particles =
        resample_particles(&mut rng, &mut particles, &normalized_weights[1..]);
    // Assert that the number of resampled particles matches the original count
    assert_eq!(
        resampled_particles.len(),
        original_len,
        "Resampled particle count does not match"
    );

    // Assert that all resampled particles are valid (i.e., they exist in the original list)
    for p in &resampled_particles {
        assert!(original_particles.contains(p), "Resampled particle is not valid");
    }
}
