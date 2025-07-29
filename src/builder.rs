//! Builder pattern for constructing BART samplers with static dispatch

use std::rc::Rc;

use numpy::ndarray::{Array, Ix1};
use numpy::Ix2;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::SmallRng;

use crate::base::BartState;
use crate::particle::{Particle, Tree};
use crate::resampling::SystematicResampling;
use crate::response::{GaussianResponseStrategy, ResponseStrategies};
use crate::sampler::ParticleGibbsSampler;
use crate::splitting::{ContinuousSplit, SplitRules};
use crate::update::{BARTWeighter, TreeContext, TreeUpdater};
use crate::LogpFunc;

// Type aliases for different node capacities using concrete ResponseStrategy
type Sampler127 = ParticleGibbsSampler<
    127,
    TreeUpdater<GaussianResponseStrategy>,
    BARTWeighter,
    SystematicResampling,
>;
type Sampler255 = ParticleGibbsSampler<
    255,
    TreeUpdater<GaussianResponseStrategy>,
    BARTWeighter,
    SystematicResampling,
>;
type Sampler511 = ParticleGibbsSampler<
    511,
    TreeUpdater<GaussianResponseStrategy>,
    BARTWeighter,
    SystematicResampling,
>;
type Sampler1023 = ParticleGibbsSampler<
    1023,
    TreeUpdater<GaussianResponseStrategy>,
    BARTWeighter,
    SystematicResampling,
>;
type Sampler2047 = ParticleGibbsSampler<
    2047,
    TreeUpdater<GaussianResponseStrategy>,
    BARTWeighter,
    SystematicResampling,
>;

/// The complete BART sampler with state
pub enum BartSampler {
    Nodes127 {
        sampler: Sampler127,
        state: BartState<127>,
    },
    Nodes255 {
        sampler: Sampler255,
        state: BartState<255>,
    },
    Nodes511 {
        sampler: Sampler511,
        state: BartState<511>,
    },
    Nodes1023 {
        sampler: Sampler1023,
        state: BartState<1023>,
    },
    Nodes2047 {
        sampler: Sampler2047,
        state: BartState<2047>,
    },
}

impl BartSampler {
    /// Execute a single sampling step
    pub fn step(&mut self, rng: &mut SmallRng, context: &TreeContext) -> &Array<f64, Ix1> {
        match self {
            BartSampler::Nodes127 { sampler, state } => {
                // Temporarily move state out, transform it, put it back
                let dummy = BartState::new(Vec::new(), Array::zeros(0));
                let old_state = std::mem::replace(state, dummy);
                *state = sampler.step(rng, old_state, context);
                &state.ensemble_predictions // Return reference - no clone!
            }
            BartSampler::Nodes255 { sampler, state } => {
                let dummy = BartState::new(Vec::new(), Array::zeros(0));
                let old_state = std::mem::replace(state, dummy);
                *state = sampler.step(rng, old_state, context);
                &state.ensemble_predictions
            }
            BartSampler::Nodes511 { sampler, state } => {
                let dummy = BartState::new(Vec::new(), Array::zeros(0));
                let old_state = std::mem::replace(state, dummy);
                *state = sampler.step(rng, old_state, context);
                &state.ensemble_predictions
            }
            BartSampler::Nodes1023 { sampler, state } => {
                let dummy = BartState::new(Vec::new(), Array::zeros(0));
                let old_state = std::mem::replace(state, dummy);
                *state = sampler.step(rng, old_state, context);
                &state.ensemble_predictions
            }
            BartSampler::Nodes2047 { sampler, state } => {
                let dummy = BartState::new(Vec::new(), Array::zeros(0));
                let old_state = std::mem::replace(state, dummy);
                *state = sampler.step(rng, old_state, context);
                &state.ensemble_predictions
            }
        }
    }
}

/// Builder for BART samplers with integrated strategy configuration
#[derive(Debug)]
pub struct BartSamplerBuilder {
    pub max_nodes: usize,
    pub n_particles: usize,
    pub n_trees: usize,
    pub init_leaf_value: f64,
    pub split_strategies: Option<Vec<SplitRules>>,
    pub response_strategy: Option<ResponseStrategies>,
    pub splitting_probs: Option<Array<f64, Ix1>>,
    pub alpha: f64,
    pub beta: f64,
    pub sigma: f64,
    pub min_samples_leaf: usize,
    pub max_depth: usize,
}

impl Default for BartSamplerBuilder {
    fn default() -> Self {
        Self {
            max_nodes: 1023,
            n_particles: 10,
            n_trees: 50,
            init_leaf_value: 0.0,
            split_strategies: None,
            response_strategy: None,
            splitting_probs: None,
            alpha: 0.95,
            beta: 2.0,
            sigma: 1.0,
            min_samples_leaf: 5,
            max_depth: 10,
        }
    }
}

impl BartSamplerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of nodes
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Set the number of particles
    pub fn with_n_particles(mut self, n_particles: usize) -> Self {
        self.n_particles = n_particles;
        self
    }

    pub fn with_n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }

    /// Set the initial leaf value
    pub fn with_init_leaf_value(mut self, init_leaf_value: f64) -> Self {
        self.init_leaf_value = init_leaf_value;
        self
    }

    /// Set split strategies for each feature
    pub fn with_split_strategies(mut self, split_strategies: Vec<SplitRules>) -> Self {
        self.split_strategies = Some(split_strategies);
        self
    }

    /// Set uniform split strategy for all features
    pub fn with_uniform_split_strategy(
        mut self,
        split_strategy: SplitRules,
        n_features: usize,
    ) -> Self {
        self.split_strategies = Some(vec![split_strategy; n_features]);
        self
    }

    /// Set response strategy
    pub fn with_response_strategy(mut self, response_strategy: ResponseStrategies) -> Self {
        self.response_strategy = Some(response_strategy);
        self
    }

    /// Set feature splitting probabilities
    pub fn with_splitting_probs(mut self, probs: Array<f64, Ix1>) -> Self {
        self.splitting_probs = Some(probs);
        self
    }

    /// Set BART hyperparameters
    pub fn with_bart_params(mut self, alpha: f64, beta: f64, sigma: f64) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self.sigma = sigma;
        self
    }

    /// Set tree constraints
    pub fn with_tree_constraints(mut self, min_samples_leaf: usize, max_depth: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self.max_depth = max_depth;
        self
    }

    /// Build the BART sampler with given data
    pub fn build(
        self,
        x_data: &Array<f64, Ix2>,
        y_data: &Array<f64, Ix1>,
        logp_fn: LogpFunc,
    ) -> PyResult<BartSampler> {
        let n_features = x_data.ncols();

        // Use provided split strategies or default to continuous for all features
        let split_strategies = self
            .split_strategies
            .unwrap_or_else(|| vec![SplitRules::Continuous(ContinuousSplit); n_features]);

        // Validate split strategies length
        if split_strategies.len() != n_features {
            return Err(PyValueError::new_err(format!(
                "Number of split strategies ({}) must match number of features ({})",
                split_strategies.len(),
                n_features
            )));
        }

        // For now, we'll use the concrete GaussianResponseStrategy since
        // the enum wrapper doesn't implement ResponseStrategy trait
        let response_strategy = GaussianResponseStrategy;

        // Create the TreeUpdater with the configured strategies
        let tree_updater = TreeUpdater::new(split_strategies, response_strategy);

        // Create the Tree weightor with the callback
        let tree_weighter = BARTWeighter { logp_func: logp_fn };

        // Extract needed fields before consuming self
        let max_nodes = self.max_nodes;
        let n_particles = self.n_particles;
        let n_trees = self.n_trees;
        let init_leaf_value = self.init_leaf_value;

        Self::create_sampler_with_components(
            tree_updater,
            tree_weighter,
            y_data,
            max_nodes,
            n_trees,
            n_particles,
            init_leaf_value,
        )
    }

    /// Internal method to create the sampler with the configured updater
    fn create_sampler_with_components(
        tree_updater: TreeUpdater<GaussianResponseStrategy>,
        tree_weighter: BARTWeighter,
        y_data: &Array<f64, Ix1>,
        max_nodes: usize,
        n_trees: usize,
        n_particles: usize,
        init_leaf_value: f64,
    ) -> PyResult<BartSampler> {
        let sampler_components = (tree_updater, tree_weighter, SystematicResampling);

        let y_mean = y_data.mean().unwrap_or(0.0);
        let init_leaf_value_rs = y_mean / n_trees as f64;
        let init_predictions = Array::from_elem(y_data.len(), y_mean);
        let n_samples = y_data.len();

        match max_nodes {
            127 => {
                let sampler = ParticleGibbsSampler::new(
                    sampler_components.0,
                    sampler_components.1,
                    sampler_components.2,
                );
                let tree_ensemble: Vec<Tree<127>> = (0..n_trees)
                    .map(|_| Tree::new(init_leaf_value_rs, n_samples))
                    .collect();
                let state = BartState::new(tree_ensemble, init_predictions);
                Ok(BartSampler::Nodes127 { sampler, state })
            }
            255 => {
                let sampler = ParticleGibbsSampler::new(
                    sampler_components.0,
                    sampler_components.1,
                    sampler_components.2,
                );
                let tree_ensemble: Vec<Tree<255>> = (0..n_trees)
                    .map(|_| Tree::new(init_leaf_value_rs, n_samples))
                    .collect();
                let state = BartState::new(tree_ensemble, init_predictions);
                Ok(BartSampler::Nodes255 { sampler, state })
            }
            511 => {
                let sampler = ParticleGibbsSampler::new(
                    sampler_components.0,
                    sampler_components.1,
                    sampler_components.2,
                );
                let tree_ensemble: Vec<Tree<511>> = (0..n_trees)
                    .map(|_| Tree::new(init_leaf_value_rs, n_samples))
                    .collect();
                let state = BartState::new(tree_ensemble, init_predictions);
                Ok(BartSampler::Nodes511 { sampler, state })
            }
            1023 => {
                let sampler = ParticleGibbsSampler::new(
                    sampler_components.0,
                    sampler_components.1,
                    sampler_components.2,
                );
                let tree_ensemble: Vec<Tree<1023>> = (0..n_trees)
                    .map(|_| Tree::new(init_leaf_value_rs, n_samples))
                    .collect();
                let state = BartState::new(tree_ensemble, init_predictions);
                Ok(BartSampler::Nodes1023 { sampler, state })
            }
            2047 => {
                let sampler = ParticleGibbsSampler::new(
                    sampler_components.0,
                    sampler_components.1,
                    sampler_components.2,
                );
                let tree_ensemble: Vec<Tree<2047>> = (0..n_trees)
                    .map(|_| Tree::new(init_leaf_value_rs, n_samples))
                    .collect();
                let state = BartState::new(tree_ensemble, init_predictions);
                Ok(BartSampler::Nodes2047 { sampler, state })
            }
            _ => Err(PyValueError::new_err(format!(
                "Unsupported MAX_NODES: {}. Use 127, 255, 511, 1023, or 2047",
                max_nodes
            ))),
        }
    }
}

/// Convenience methods for common configurations
impl BartSamplerBuilder {
    /// Create a builder for regression with continuous features
    pub fn for_regression() -> Self {
        Self::new().with_response_strategy(ResponseStrategies::Gaussian(GaussianResponseStrategy))
    }
}
