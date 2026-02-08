use std::collections::{HashSet, VecDeque};

use ndarray::array;

use pymc_bart::particle::{Particle, SampleIndices, Weight};

#[test]
fn test_sample_indices_new() {
    let num_samples = 5;
    let indices = SampleIndices::new(num_samples);

    assert_eq!(indices.leaf_nodes, HashSet::from([0]));
    assert_eq!(indices.expansion_nodes, VecDeque::from([0]));
    assert_eq!(indices.data_indices.len(), 1);
    assert_eq!(
        indices.data_indices[0],
        (0..num_samples).collect::<Vec<_>>()
    );
}

#[test]
fn test_sample_indices_add_index() {
    let mut indices = SampleIndices::new(5);
    indices.add_index(1, vec![2, 3]);

    assert!(indices.leaf_nodes.contains(&1));
    assert_eq!(indices.expansion_nodes.back(), Some(&1));
    assert_eq!(indices.data_indices[1], vec![2, 3]);
}

#[test]
fn test_sample_indices_remove_index() {
    let mut indices = SampleIndices::new(5);
    indices.add_index(1, vec![2, 3]);
    indices.remove_index(1);

    assert!(!indices.leaf_nodes.contains(&1));
    assert!(indices.data_indices[1].is_empty());
}

#[test]
fn test_sample_indices_is_empty() {
    let mut indices = SampleIndices::new(5);
    assert!(!indices.is_empty());

    indices.pop_expansion_index();
    assert!(indices.is_empty());
}

#[test]
fn test_sample_indices_pop_expansion_index() {
    let mut indices = SampleIndices::new(5);
    let popped = indices.pop_expansion_index();

    assert_eq!(popped, Some(0));
    assert!(indices.expansion_nodes.is_empty());
}

// Test Weight functionality
#[test]
fn test_weight_new() {
    let weight = Weight::new();

    assert_eq!(weight.log_w, 0.0);
    assert_eq!(weight.log_likelihood, 0.0);
}

#[test]
fn test_weight_set() {
    let mut weight = Weight::new();
    weight.set(2.5);

    assert_eq!(weight.log_w, 2.5);
    assert_eq!(weight.log_likelihood, 2.5);
}

#[test]
fn test_weight_update() {
    let mut weight = Weight::new();
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
    assert_eq!(particle.indices.leaf_nodes, HashSet::from([0]));
    assert_eq!(particle.weight.log_w, 0.0);
}

#[test]
fn test_particle_finished() {
    let mut particle = Particle::new(1.0, 5);

    // Initially not finished
    assert!(!particle.finished());

    // Pop the only expansion node
    particle.indices.pop_expansion_index();

    // Now it should be finished
    assert!(particle.finished());
}

#[test]
fn test_particle_predict() {
    let mut particle = Particle::new(1.0, 5);

    // Mock tree structure and values
    particle.tree.split_node(0, 0, 2.5, 3.0, -1.0).unwrap();

    particle.indices.add_index(1, vec![1, 2]);
    particle.indices.add_index(2, vec![3, 4]);

    let X = array![
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];

    let predictions = particle.predict;
}
