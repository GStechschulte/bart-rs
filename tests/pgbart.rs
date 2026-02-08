use pymc_bart::particle::{Particle, SampleIndices, Weight};
use pymc_bart::pgbart::{normalize_weights, resample_particles, select_particle};
use pymc_bart::tree::DecisionTree;

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
    let normalized_weights = normalize_weights(&particles);
    let selected_particles = select_particle(&mut particles, &normalized_weights);

    // Assert that the selected particle is one of the original particles
    assert!(
        particles.contains(&selected_particles),
        "Selected particle is not valid"
    );
}

#[test]
fn test_resample_particles() {
    let mut particles = create_test_particles();
    let normalized_weights = normalize_weights(&particles);

    // Resample particles based on normalized weights
    let resampled_particles = resample_particles(&mut particles, &normalized_weights);

    // Assert that the number of resampled particles matches the original count
    assert_eq!(
        resampled_particles.len(),
        particles.len(),
        "Resampled particle count does not match"
    );

    // Assert that all resampled particles are valid (i.e., they exist in the original list)
    for p in &resampled_particles {
        assert!(particles.contains(p), "Resampled particle is not valid");
    }
}
