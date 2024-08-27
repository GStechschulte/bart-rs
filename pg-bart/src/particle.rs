use std::cmp::Ordering;
use std::f64::consts::PI;

use rand::{thread_rng, Rng};

use crate::data::Matrix;

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
    fn grow(&mut self, X: &Matrix<f64>, prior: f64, state: &PgBartState) -> bool {
        // 1.) Identify leaf nodes eligible to expand..
        let leaf_nodes = self.tree.get_leaf_nodes();

        for leaf_node in leaf_nodes {
            // 2.) Compute probability that this leaf node will remain a leaf node
            //     and stochastically decide whether or not to grow
            let depth = self.tree.node_depth(leaf_node);
            let expand = state.probabilities.sample_expand_flag(depth);
            if expand {
                let feature = state.probabilities.sample_feature();
                // Get the observed data points that were routed to this leaf_node
                // because we need them in order to sample a threshold (split value)
                // let feature_values = X.select_rows(rows, &feature);
            }
        }

        // Get the observations that belong to this leaf node

        return true;
    }
}
