use std::{f64, rc::Rc};

use numpy::ndarray::{Array, Array1, Ix2};
use rand::{Rng, SeedableRng};

use crate::{particle::Particle, response::ResponseStrategy, splitting::SplitRules};

/// Represents the decision outcome for mutation proposals
#[derive(Clone, Debug)]
pub enum MutationDecision {
    /// Mutation should proceed with the given proposal
    Accept(TreeProposal),
    /// Mutation should be rejected - no clone needed
    Reject,
}

/// BART Tree proposal types. Currently, only growth proposals are supported.
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
}

/// Contains relevant parameters needed for SMC update steps.
#[derive(Clone, Debug)]
pub struct TreeContext {
    pub x_data: Array<f64, Ix2>,
    pub y_data: Array1<f64>,
    pub alpha: f64,
    pub beta: f64,
    pub sigma: f64,
    pub min_samples_leaf: usize,
    pub max_depth: usize,
    pub splitting_probs: Option<Array1<f64>>,
}

/// Optimized TreeUpdater using Vec for O(1) feature access instead of HashMap
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
        particle: &mut Particle<MAX_NODES>,
        proposal: &Self::Proposal,
        context: &Self::Context,
    );
}

/// Weight calculation trait with const generic support
pub trait Weight<const MAX_NODES: usize> {
    type Context;

    fn log_weight(&self, particle: &Particle<MAX_NODES>, context: &Self::Context) -> f64;
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

        // Generate and validate proposal using integrated strategies
        let (split_var, split_val) = match self.propose_split(rng, tree, node_idx, context) {
            Some(split) => split,
            None => return MutationDecision::Reject,
        };

        if !self.is_valid_split(tree, node_idx, split_var, split_val, context) {
            return MutationDecision::Reject;
        }

        let (left_val, right_val) =
            self.propose_leaf_values(rng, tree, node_idx, split_var, split_val, context);

        MutationDecision::Accept(TreeProposal {
            node_idx,
            split_var,
            split_val,
            left_value: left_val,
            right_value: right_val,
        })
    }

    fn apply_update(
        &self,
        tree: &mut Particle<MAX_NODES>,
        proposal: &Self::Proposal,
        _context: &Self::Context,
    ) {
        // Conditional clone using Rc::make_mut
        let tree_mut = Rc::make_mut(tree);
        tree_mut.split_node(
            proposal.node_idx,
            proposal.split_var,
            proposal.split_val,
            proposal.left_value,
            proposal.right_value,
        );
    }
}

impl<R: ResponseStrategy> TreeUpdater<R> {
    /// Propose a split using the integrated split strategies
    fn propose_split<const MAX_NODES: usize>(
        &self,
        rng: &mut impl Rng,
        tree: &Particle<MAX_NODES>,
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
        let node_samples = tree.get_node_samples(node_idx, &context.x_data);
        if node_samples.is_empty() {
            return None;
        }

        // Extract feature values for samples in this node
        let feature_values: Vec<f64> = node_samples
            .iter()
            .map(|&idx| context.x_data[[idx, split_var]])
            .collect();

        // Use the appropriate split strategy for this feature
        let split_strategy = &self.split_strategies[split_var];
        let split_val = split_strategy.sample_split_value(rng, &feature_values)?;

        Some((split_var, split_val))
    }

    /// Validate split using the integrated split strategies
    fn is_valid_split<const MAX_NODES: usize>(
        &self,
        tree: &Particle<MAX_NODES>,
        node_idx: usize,
        split_var: usize,
        split_val: f64,
        context: &TreeContext,
    ) -> bool {
        let node_samples = tree.get_node_samples(node_idx, &context.x_data);

        if node_samples.len() < context.min_samples_leaf * 2 {
            return false;
        }

        // Use the split strategy to determine the actual split
        let split_strategy = &self.split_strategies[split_var];
        let (left_indices, right_indices) =
            split_strategy.split_data_indices(&context.x_data, split_var, split_val, &node_samples);

        left_indices.len() >= context.min_samples_leaf
            && right_indices.len() >= context.min_samples_leaf
    }

    /// Propose leaf values using the integrated response strategy
    fn propose_leaf_values<const MAX_NODES: usize>(
        &self,
        rng: &mut impl Rng,
        tree: &Particle<MAX_NODES>,
        node_idx: usize,
        split_var: usize,
        split_val: f64,
        context: &TreeContext,
    ) -> (f64, f64) {
        let node_samples = tree.get_node_samples(node_idx, &context.x_data);

        // Split the data to get left and right child samples
        let split_strategy = &self.split_strategies[split_var];
        let (left_indices, right_indices) =
            split_strategy.split_data_indices(&context.x_data, split_var, split_val, &node_samples);

        // Sample leaf values using the response strategy
        let left_value =
            self.response_strategy
                .sample_leaf_value(&context.y_data, &left_indices, rng);

        let right_value =
            self.response_strategy
                .sample_leaf_value(&context.y_data, &right_indices, rng);

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
pub struct BARTWeighter;

impl<const MAX_NODES: usize> Weight<MAX_NODES> for BARTWeighter {
    type Context = TreeContext;

    fn log_weight(&self, tree: &Particle<MAX_NODES>, context: &Self::Context) -> f64 {
        self.compute_log_likelihood(tree, context)
    }
}

impl BARTWeighter {
    fn compute_log_likelihood<const MAX_NODES: usize>(
        &self,
        _tree: &Particle<MAX_NODES>,
        _context: &TreeContext,
    ) -> f64 {
        // TODO: Implement actual log-likelihood computation
        // This should compute the likelihood of the data given the tree predictions
        let mut rng = rand::rng();
        rng.random()
    }
}
