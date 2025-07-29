use std::f64;

use numpy::ndarray::{Array, Array1, Ix1, Ix2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::{
    LogpFunc,
    particle::{Particle, Tree},
    response::ResponseStrategy,
    splitting::SplitRules,
};

/// Represents the decision outcome for update proposals
#[derive(Clone, Debug)]
pub enum MutationDecision {
    /// Mutation should proceed with the given proposal
    Accept(TreeProposal),
    /// Mutation should be rejected - no clone needed
    Reject,
}

/// BART Tree proposal types. A proposal can be thought of as a unit
/// of work as it contains all the information needed to perform, for
/// example, a growth mutation.
///
/// **NOTE**: If more proposal variants are added in the future, a proposal
/// enum should be introduced.
#[derive(Clone, Debug)]
pub struct TreeProposal {
    pub node_idx: usize,
    pub split_var: usize,
    pub split_val: f64,
    pub left_value: f64,
    pub right_value: f64,
    pub affected_samples: Vec<usize>,
}

/// Contains relevant parameters needed for SMC update steps.
#[derive(Clone, Debug)]
pub struct TreeContext {
    pub x_data: Array<f64, Ix2>,
    pub y_data: Array1<f64>,
    pub alpha: f64,
    pub beta: f64,
    pub sigma: f64,
    pub n_trees: usize,
    pub min_samples_leaf: usize,
    pub max_nodes: usize,
    pub splitting_probs: Option<Array1<f64>>,
}

#[derive(Clone, Debug)]
pub struct TreeUpdater<R> {
    /// Split strategies per feature (index = feature_idx)
    split_strategies: Vec<SplitRules>,
    /// Response strategy for leaf value generation
    response_strategy: R,
}

impl<R: ResponseStrategy> TreeUpdater<R> {
    /// Create a new TreeUpdater with strategies for each feature
    pub fn new(split_strategies: Vec<SplitRules>, response_strategy: R) -> Self {
        Self {
            split_strategies,
            response_strategy,
        }
    }

    /// Create TreeUpdater with the same split strategy for all features
    pub fn with_uniform_split_strategy(
        n_features: usize,
        split_strategy: SplitRules,
        response_strategy: R,
    ) -> Self {
        Self {
            split_strategies: vec![split_strategy; n_features],
            response_strategy,
        }
    }
}

pub trait Update<const MAX_NODES: usize> {
    type Proposal;
    type Context;

    /// Evaluate mutation feasibility for a node of the given particle.
    fn should_update(
        &self,
        rng: &mut impl Rng,
        particle: &Particle<MAX_NODES>,
        node_idx: usize,
        context: &Self::Context,
    ) -> MutationDecision;

    /// Apply a mutation proposal to a particle.
    fn apply_update(
        &self,
        particle: &mut Tree<MAX_NODES>,
        proposal: &Self::Proposal,
        context: &Self::Context,
    );
}

/// Weight calculation trait with const generic support
pub trait Weight<const MAX_NODES: usize> {
    fn log_weight(&self, data: &Array<f64, Ix1>) -> f64;
}

impl<const MAX_NODES: usize, R: ResponseStrategy> Update<MAX_NODES> for TreeUpdater<R> {
    type Proposal = TreeProposal;
    type Context = TreeContext;

    /// Validate mutation feasibility before applying the mutation (update).
    fn should_update(
        &self,
        rng: &mut impl Rng,
        tree: &Particle<MAX_NODES>,
        node_idx: usize,
        context: &Self::Context,
    ) -> MutationDecision {
        let depth = tree.get_depth(node_idx);
        let prob_not_expanding = 1.0 - (context.alpha * (1.0 + depth as f64).powf(-context.beta));

        if prob_not_expanding > rng.random::<f64>() {
            return MutationDecision::Reject;
        }

        let node_samples: Vec<usize> = tree.get_leaf_samples(node_idx).collect();

        // Generate and validate proposal using integrated strategies
        let (split_var, split_val) = match self.propose_split(rng, &node_samples, node_idx, context)
        {
            Some(split) => split,
            None => return MutationDecision::Reject,
        };

        // TODO: Move this to the propose_split?
        // if !self.is_valid_split(tree, node_idx, split_var, split_val, context) {
        //     return MutationDecision::Reject;
        // }

        let (left_val, right_val) =
            self.propose_leaf_values(rng, &node_samples, split_var, split_val, context);

        MutationDecision::Accept(TreeProposal {
            node_idx,
            split_var,
            split_val,
            left_value: left_val,
            right_value: right_val,
            affected_samples: node_samples,
        })
    }

    fn apply_update(
        &self,
        tree: &mut Tree<MAX_NODES>,
        proposal: &Self::Proposal,
        context: &Self::Context,
    ) {
        tree.split_node(
            proposal.node_idx,
            proposal.split_var,
            proposal.split_val,
            proposal.left_value,
            proposal.right_value,
        );
        tree.update_leaf_assignments(
            proposal.node_idx,
            proposal.split_var,
            proposal.split_val,
            &proposal.affected_samples,
            &context.x_data,
        );
    }
}

impl<R: ResponseStrategy> TreeUpdater<R> {
    /// Propose a split using the integrated split strategies
    fn propose_split(
        &self,
        rng: &mut impl Rng,
        node_samples: &[usize],
        node_idx: usize,
        context: &TreeContext,
    ) -> Option<(usize, f64)> {
        // Select split variable based on splitting probabilities or uniform
        let split_var = if let Some(ref probs) = context.splitting_probs {
            self.sample_feature_from_probs(rng, probs)
        } else {
            rng.random_range(0..context.x_data.ncols())
        };

        // Get data indices for this node
        // let mut node_samples = tree.get_leaf_samples(node_idx).peekable();

        // if node_samples.peek().is_none() {
        // return None;
        // }

        // Extract feature values for samples in this node
        let feature_values = node_samples
            .iter()
            .map(|&idx| context.x_data[[idx, split_var]]);

        // Use the appropriate split strategy for this feature
        let split_strategy = &self.split_strategies[split_var];
        let split_val = split_strategy.sample_split_value(rng, feature_values)?;

        Some((split_var, split_val))
    }

    /// Propose leaf values using a response strategy.
    fn propose_leaf_values(
        &self,
        rng: &mut impl Rng,
        node_samples: &[usize],
        split_var: usize,
        split_val: f64,
        context: &TreeContext,
    ) -> (f64, f64) {
        let initial_state = (0.0, 0, 0.0, 0);

        let (left_sum_y, left_n, right_sum_y, right_n) = node_samples.iter().fold(
            initial_state,
            |(mut l_sum, mut l_n, mut r_sum, mut r_n), &idx| {
                if context.x_data[[idx, split_var]] < split_val {
                    l_sum += context.y_data[idx];
                    l_n += 1;
                } else {
                    r_sum += context.y_data[idx];
                    r_n += 1;
                }
                (l_sum, l_n, r_sum, r_n)
            },
        );

        let left_value = {
            let dist = Normal::new(0.0, 1.0).unwrap();
            let norm = dist.sample(rng);
            if left_n == 0 {
                norm // Or whatever the empty case logic is
            } else {
                let mean_y = left_sum_y / context.y_data.len() as f64 / context.n_trees as f64;
                mean_y + norm
            }
        };

        let right_value = {
            let dist = Normal::new(0.0, 1.0).unwrap();
            let norm = dist.sample(rng);
            if right_n == 0 {
                norm // Or whatever the empty case logic is
            } else {
                let mean_y = right_sum_y / context.y_data.len() as f64 / context.n_trees as f64;
                mean_y + norm
            }
        };

        // Split the data to get left and right child samples
        // let split_strategy = &self.split_strategies[split_var];
        // let (left_indices, right_indices) =
        //     split_strategy.split_data_indices(&context.x_data, split_var, split_val, &node_samples);

        // Sample leaf values using the response strategy
        // let left_value = self.response_strategy.sample_leaf_value(
        //     rng,
        //     &context.y_data,
        //     &left_indices,
        //     context.n_trees,
        // );

        // let right_value = self.response_strategy.sample_leaf_value(
        //     rng,
        //     &context.y_data,
        //     &right_indices,
        //     context.n_trees,
        // );

        (left_value, right_value)
    }

    /// Sample feature index based on probability weights
    fn sample_feature_from_probs(&self, rng: &mut impl Rng, probs: &Array1<f64>) -> usize {
        let total: f64 = probs.sum();
        let mut target = rng.random::<f64>() * total;

        for (idx, &prob) in probs.iter().enumerate() {
            target -= prob;
            if target <= 0.0 {
                return idx;
            }
        }

        // Fallback to last index
        probs.len() - 1
    }
}

/// BART weight calculator
pub struct BARTWeighter {
    pub logp_func: LogpFunc,
}

impl<const MAX_NODES: usize> Weight<MAX_NODES> for BARTWeighter {
    fn log_weight(&self, data: &Array<f64, Ix1>) -> f64 {
        unsafe { (self.logp_func)(data.as_ptr(), data.len()) }
    }
}
