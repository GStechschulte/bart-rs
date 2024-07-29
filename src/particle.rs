use std::cmp::Ordering;
use std::f64::consts::PI;

use rand::{thread_rng, Rng};

use crate::math::{self, Matrix};

use crate::{sampler::PgBartState, tree::DecisionTree};

// Wraps a decision tree along with indices of leaf nodes to expand
// and a log weight
#[derive(Debug)]
pub struct Particle {
    tree: DecisionTree,
    indices: Vec<usize>,
    log_weight: f64,
}

impl Particle {
    pub fn new(feature: usize, threshold: f64, value: f64) -> Self {
        let mut tree = DecisionTree::new();
        tree.add_node(feature, threshold, value);

        Particle {
            tree,
            indices: (0..1).collect(), // TODO
            log_weight: 0.0,
        }
    }

    // Attempt to grow this particle
    fn grow(&mut self, X: &[Vec<f64>], y: &[f64], prior: f64) {
        let mut rng = thread_rng();

        // Identify leaf nodes eligible to expand..
    }
}

// Sample the value (threshold) a a feature should be split on
fn sample_split_feature() {}

// Sample a boolean flag indicating if a node should be split or not.
fn sample_split_value() {}
