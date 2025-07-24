use std::rc::Rc;

use rand::Rng;

use crate::particle::Particle;

/// Represents the decision outcome for mutation proposals
#[derive(Clone, Debug)]
pub enum MutationDecision {
    /// Mutation should proceed with the given proposal
    Accept(BARTProposal),
    /// Mutation should be rejected - no clone needed
    Reject,
}

/// BART Tree proposal types. Currently, only growth proposals are supported.
#[derive(Clone, Debug)]
pub struct BARTProposal {
    pub node_idx: usize,
    pub split_var: usize,
    pub split_val: f64,
    pub left_value: f64,
    pub right_value: f64,
}

/// Enhanced BARTContext with additional constraints
#[derive(Clone)]
pub struct BARTContext {
    pub x_data: Vec<Vec<f64>>,
    pub y_residuals: Vec<f64>,
    pub alpha: f64,
    pub beta: f64,
    pub sigma: f64,
    pub min_samples_leaf: usize,
    pub max_depth: usize,
}

#[derive(Clone, Debug)]
pub struct Moves;

pub trait Update<const MAX_NODES: usize> {
    type Proposal;
    type Context;

    /// Evaluate mutation feasibility without expensive operations
    fn should_mutate(
        &self,
        rng: &mut impl Rng,
        particle: &Particle<MAX_NODES>,
        context: &Self::Context,
    ) -> MutationDecision;

    /// Execute mutation only when decision is Accept
    fn apply_mutation(
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

impl<const MAX_NODES: usize> Update<MAX_NODES> for Moves {
    type Proposal = BARTProposal;
    type Context = BARTContext;

    fn should_mutate(
        &self,
        rng: &mut impl Rng,
        tree: &Particle<MAX_NODES>,
        context: &Self::Context,
    ) -> MutationDecision {
        // Validate mutation feasibility before expensive cloning
        let available_leaves = tree.get_leaf_indices();
        if available_leaves.is_empty() {
            return MutationDecision::Reject;
        }

        let node_idx = self.select_leaf_node(rng, &available_leaves);
        let depth = tree.get_depth(node_idx);
        let prob_not_expanding = 1.0 - (context.alpha * (1.0 + depth as f64).powf(-context.beta));

        if depth >= context.max_depth || prob_not_expanding > rng.random::<f64>() {
            return MutationDecision::Reject;
        }

        // Generate and validate proposal
        let node_idx = self.select_leaf_node(rng, &available_leaves);
        let (split_var, split_val) = self.propose_split(rng, context);

        if !self.is_valid_split(tree, node_idx, split_var, split_val, context) {
            return MutationDecision::Reject;
        }

        let (left_val, right_val) = self.propose_leaf_values(rng, context);

        MutationDecision::Accept(BARTProposal {
            node_idx,
            split_var,
            split_val,
            left_value: left_val,
            right_value: right_val,
        })
    }

    fn apply_mutation(
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

impl Moves {
    pub fn new() -> Self {
        Self
    }

    fn select_leaf_node(&self, rng: &mut impl Rng, available_leaves: &[usize]) -> usize {
        available_leaves[rng.random_range(0..available_leaves.len())]
    }

    fn propose_split(&self, rng: &mut impl Rng, context: &BARTContext) -> (usize, f64) {
        // TODO. Select the split_var based on a splitting probability prior probability vector
        // from the context structure
        //
        // let p = rng.random::<f64>();
        // let splitting_probs = context.splitting_probability;

        let n_vars = context.x_data.first().map_or(0, |row| row.len());
        let split_var = rng.random_range(0..n_vars);
        let split_val = 0.5; // Should be based on data distribution
        (split_var, split_val)
    }

    fn is_valid_split<const MAX_NODES: usize>(
        &self,
        tree: &Particle<MAX_NODES>,
        node_idx: usize,
        split_var: usize,
        split_val: f64,
        context: &BARTContext,
    ) -> bool {
        let node_samples = tree.get_node_samples(node_idx, &context.x_data);

        if node_samples.len() < context.min_samples_leaf * 2 {
            return false;
        }

        let left_count = node_samples
            .iter()
            .filter(|&&sample_idx| context.x_data[sample_idx][split_var] <= split_val)
            .count();

        let right_count = node_samples.len() - left_count;

        left_count >= context.min_samples_leaf && right_count >= context.min_samples_leaf
    }

    fn propose_leaf_values(&self, rng: &mut impl Rng, _context: &BARTContext) -> (f64, f64) {
        // Should sample from conjugate prior distribution
        (0.1, -0.1)
    }
}

/// BART weight calculator
pub struct BARTWeighter;

impl<const MAX_NODES: usize> Weight<MAX_NODES> for BARTWeighter {
    type Context = BARTContext;

    fn log_weight(&self, tree: &Particle<MAX_NODES>, context: &Self::Context) -> f64 {
        let log_likelihood = self.compute_log_likelihood(tree, context);
        log_likelihood
    }
}

impl BARTWeighter {
    fn compute_log_likelihood<const MAX_NODES: usize>(
        &self,
        tree: &Particle<MAX_NODES>,
        context: &BARTContext,
    ) -> f64 {
        // TODO...
        // let predictions = tree.predict(&context.x_data);
        let ll = 0.0;

        ll
    }
}
