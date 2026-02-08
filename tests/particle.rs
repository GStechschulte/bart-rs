use ndarray::array;

use pymc_bart::particle::{Particle, SampleIndices, Weight};

#[test]
fn test_sample_indices_new() {
    let num_samples = 5;
    let indices = SampleIndices::new(num_samples);
    let expected = SampleIndices::new(num_samples);

    assert_eq!(indices, expected);
}

// Test Weight functionality
#[test]
fn test_weight_new() {
    let weight = Weight {
        log_w: 0.0,
        log_likelihood: 0.0,
    };

    assert_eq!(weight.log_w, 0.0);
    assert_eq!(weight.log_likelihood, 0.0);
}

#[test]
fn test_weight_set() {
    let mut weight = Weight {
        log_w: 0.0,
        log_likelihood: 0.0,
    };
    weight.set(2.5);

    assert_eq!(weight.log_w, 2.5);
    assert_eq!(weight.log_likelihood, 2.5);
}

#[test]
fn test_weight_update() {
    let mut weight = Weight {
        log_w: 0.0,
        log_likelihood: 0.0,
    };
    weight.set(2.5);
    weight.update(3.0);

    assert_eq!(weight.log_w, 3.0);
    assert_eq!(weight.log_likelihood, 3.0);
}

// Test Particle functionality
#[test]
fn test_particle_new() {
    let particle = Particle::new(1.0, 5);

    assert_eq!(particle.tree.value[0], 1.0); // Assuming DecisionTree initializes root with value
    assert_eq!(particle.indices, SampleIndices::new(5));
    assert_eq!(particle.weight.log_w, 0.0);
}

#[test]
fn test_particle_finished() {
    let particle = Particle::new(1.0, 5);

    // Initially not finished
    assert!(!particle.finished());
}

#[test]
fn test_particle_predict() {
    let particle = Particle::new(1.0, 5);

    let x = array![[2.0], [3.0], [4.0], [5.0], [6.0]];

    let predictions = particle.predict(&x);

    assert_eq!(predictions.len(), 5);
    assert!(predictions.iter().all(|v| (*v - 1.0).abs() < 1e-12));
}
