//! Integration tests for the PGBART implementation
//! These tests verify that the core algorithm functionality works correctly.

use std::rc::Rc;
use ndarray::{Array1, Array2};
use pymc_bart::base::{PgBartSettings, PgBartState};
use pymc_bart::data::PyData;
use pymc_bart::ops::Response;
use pymc_bart::split_rules::{ContinuousSplit, SplitRuleType};

/// Mock data implementation for testing
struct MockData {
    X: Rc<Array2<f64>>,
    y: Rc<Array1<f64>>,
}

impl MockData {
    fn new(X: Array2<f64>, y: Array1<f64>) -> Self {
        Self {
            X: Rc::new(X),
            y: Rc::new(y),
        }
    }
}

impl PyData for MockData {
    fn X(&self) -> Rc<Array2<f64>> {
        Rc::clone(&self.X)
    }

    fn y(&self) -> Rc<Array1<f64>> {
        Rc::clone(&self.y)
    }

    fn evaluate_logp(&self, x: Array1<f64>) -> f64 {
        // Simple Gaussian log-likelihood for testing
        let diff = &x - &*self.y;
        let sse = diff.mapv(|v| v * v).sum();
        -0.5 * sse
    }
}

#[test]
fn test_pgbart_initialization() {
    // Create simple test data
    let X = Array2::from_shape_vec((10, 2), vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
    ]).unwrap();
    
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    
    let data = Box::new(MockData::new(X, y));
    
    // Create settings
    let settings = PgBartSettings::new(
        5,                                          // n_trees
        10,                                         // n_particles
        0.95,                                       // alpha
        2.0,                                        // beta
        vec![1.0],                                  // leaf_sd
        (0.1, 0.1),                                 // batch
        vec![1.0, 1.0],                             // init_alpha_vec
        Response::Constant(pymc_bart::ops::ConstantResponse), // response
        vec![
            SplitRuleType::Continuous(ContinuousSplit),
            SplitRuleType::Continuous(ContinuousSplit),
        ],                                          // split_rules
        1,                                          // n_dim
    );
    
    // Initialize PGBART state
    let state = PgBartState::new(settings, data);
    
    // Check initialization
    assert_eq!(state.forest.trees.len(), 5);
    assert_eq!(state.predictions.len(), 10);
    assert_eq!(state.variable_inclusion.len(), 2);
    assert!(state.tune);
}

#[test]
fn test_pgbart_single_step() {
    // Create simple test data
    let X = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    
    let data = Box::new(MockData::new(X, y));
    
    // Create minimal settings
    let settings = PgBartSettings::new(
        2,                                          // n_trees
        3,                                          // n_particles
        0.95,                                       // alpha
        2.0,                                        // beta
        vec![0.5],                                  // leaf_sd
        (1.0, 1.0),                                 // batch (update all trees)
        vec![1.0],                                  // init_alpha_vec
        Response::Constant(pymc_bart::ops::ConstantResponse), // response
        vec![SplitRuleType::Continuous(ContinuousSplit)], // split_rules
        1,                                          // n_dim
    );
    
    let mut state = PgBartState::new(settings, data);
    
    // Get initial predictions
    let initial_predictions = state.predictions.clone();
    
    // Run one step
    state.step();
    
    // Check that predictions may have changed (though not guaranteed)
    assert_eq!(state.predictions.len(), initial_predictions.len());
    assert!(state.predictions.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_pgbart_multiple_steps() {
    // Create test data with clear linear relationship
    let X = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    
    let data = Box::new(MockData::new(X, y));
    
    let settings = PgBartSettings::new(
        3,                                          // n_trees
        5,                                          // n_particles
        0.95,                                       // alpha
        2.0,                                        // beta
        vec![0.1],                                  // leaf_sd
        (0.5, 0.5),                                 // batch
        vec![1.0],                                  // init_alpha_vec
        Response::Constant(pymc_bart::ops::ConstantResponse), // response
        vec![SplitRuleType::Continuous(ContinuousSplit)], // split_rules
        1,                                          // n_dim
    );
    
    let mut state = PgBartState::new(settings, data);
    
    // Run multiple steps
    for _ in 0..5 {
        state.step();
    }
    
    // Verify state is still valid
    assert_eq!(state.predictions.len(), 8);
    assert!(state.predictions.iter().all(|&x| x.is_finite()));
    assert_eq!(state.variable_inclusion.len(), 1);
}

#[test]
fn test_forest_basic_operations() {
    use pymc_bart::forest::{Forest, Predict};
    
    // Test forest creation
    let forest = Forest::new(3, 5, 1.0, 100);
    
    assert_eq!(forest.trees.len(), 3);
    assert_eq!(forest.weights.len(), 3);
    
    // Test prediction
    let X = Array2::from_shape_vec((5, 2), vec![
        1.0, 2.0, 3.0, 4.0, 5.0,
        0.5, 1.5, 2.5, 3.5, 4.5,
    ]).unwrap();
    
    for tree in &forest.trees {
        let predictions = tree.predict(&X);
        assert_eq!(predictions.len(), 5);
        assert!(predictions.iter().all(|&x| x.is_finite()));
    }
}

#[test]
fn test_particle_arrays_normalization() {
    use pymc_bart::forest::ParticleArrays;
    
    let mut arrays = ParticleArrays::new(4);
    
    // Set some test weights (log scale)
    arrays.weights = vec![0.0, -1.0, -2.0, -0.5];
    
    // Normalize
    arrays.normalize_weights();
    
    // Check that weights sum to 1 (approximately)
    let sum: f64 = arrays.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    
    // Check all weights are positive
    assert!(arrays.weights.iter().all(|&w| w >= 0.0));
}