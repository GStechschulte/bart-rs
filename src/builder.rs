//! Builder pattern for constructing BART samplers with static dispatch

use std::collections::HashMap;
use std::rc::Rc;

use numpy::ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use rand::rngs::SmallRng;

use crate::base::BartState;
use crate::particle::{Particle, Tree};
use crate::resampling::SystematicResampling;
use crate::response::ResponseStrategies;
use crate::sampler::ParticleGibbsSampler;
use crate::split_rules::SplitRules;
use crate::update::{BARTContext, BARTWeighter, Moves};

type Sampler127 = ParticleGibbsSampler<127, Moves, BARTWeighter, SystematicResampling>;
type Sampler255 = ParticleGibbsSampler<255, Moves, BARTWeighter, SystematicResampling>;
type Sampler511 = ParticleGibbsSampler<511, Moves, BARTWeighter, SystematicResampling>;
type Sampler1023 = ParticleGibbsSampler<1023, Moves, BARTWeighter, SystematicResampling>;
type Sampler2047 = ParticleGibbsSampler<2047, Moves, BARTWeighter, SystematicResampling>;

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
    pub fn step(&mut self, rng: &mut SmallRng, context: &BARTContext) -> Vec<f64> {
        match self {
            BartSampler::Nodes127 { sampler, state } => {
                *state = sampler.step(rng, state.clone(), context);
                state.weights.clone()
            }
            BartSampler::Nodes255 { sampler, state } => {
                *state = sampler.step(rng, state.clone(), context);
                state.weights.clone()
            }
            BartSampler::Nodes511 { sampler, state } => {
                *state = sampler.step(rng, state.clone(), context);
                state.weights.clone()
            }
            BartSampler::Nodes1023 { sampler, state } => {
                *state = sampler.step(rng, state.clone(), context);
                state.weights.clone()
            }
            BartSampler::Nodes2047 { sampler, state } => {
                *state = sampler.step(rng, state.clone(), context);
                state.weights.clone()
            }
        }
    }

    /// Execute multiple sampling steps
    pub fn run(
        &mut self,
        rng: &mut SmallRng,
        context: &BARTContext,
        n_iterations: usize,
    ) -> Vec<f64> {
        match self {
            BartSampler::Nodes127 { sampler, state } => {
                *state = sampler.run(rng, state.clone(), context, n_iterations);
                state.weights.clone()
            }
            BartSampler::Nodes255 { sampler, state } => {
                *state = sampler.run(rng, state.clone(), context, n_iterations);
                state.weights.clone()
            }
            BartSampler::Nodes511 { sampler, state } => {
                *state = sampler.run(rng, state.clone(), context, n_iterations);
                state.weights.clone()
            }
            BartSampler::Nodes1023 { sampler, state } => {
                *state = sampler.run(rng, state.clone(), context, n_iterations);
                state.weights.clone()
            }
            BartSampler::Nodes2047 { sampler, state } => {
                *state = sampler.run(rng, state.clone(), context, n_iterations);
                state.weights.clone()
            }
        }
    }
}

/// Builder for BART samplers
#[derive(Default)]
pub struct BartSamplerBuilder {
    pub max_nodes: usize,
    pub n_particles: usize,
    pub init_leaf_value: f64,
    // Add other fields as needed
}

impl BartSamplerBuilder {
    pub fn new() -> Self {
        Self {
            max_nodes: 1023,
            n_particles: 20,
            init_leaf_value: 0.0,
        }
    }

    pub fn build(self, y_data: &Array1<f64>) -> PyResult<BartSampler> {
        let create_sampler_and_state = |max_nodes: usize| -> PyResult<BartSampler> {
            let sampler_components = (Moves::new(), BARTWeighter, SystematicResampling);

            match max_nodes {
                127 => {
                    let sampler = ParticleGibbsSampler::new(
                        sampler_components.0,
                        sampler_components.1,
                        sampler_components.2,
                    );
                    let particles: Vec<Particle<127>> = (0..self.n_particles)
                        .map(|_| Rc::new(Tree::new(self.init_leaf_value, y_data.len())))
                        .collect();
                    let state = BartState::new(particles, vec![1.0; self.n_particles]);
                    Ok(BartSampler::Nodes127 { sampler, state })
                }
                255 => {
                    let sampler = ParticleGibbsSampler::new(
                        sampler_components.0,
                        sampler_components.1,
                        sampler_components.2,
                    );
                    let particles: Vec<Particle<255>> = (0..self.n_particles)
                        .map(|_| Rc::new(Tree::new(self.init_leaf_value, y_data.len())))
                        .collect();
                    let state = BartState::new(particles, vec![1.0; self.n_particles]);
                    Ok(BartSampler::Nodes255 { sampler, state })
                }
                511 => {
                    let sampler = ParticleGibbsSampler::new(
                        sampler_components.0,
                        sampler_components.1,
                        sampler_components.2,
                    );
                    let particles: Vec<Particle<511>> = (0..self.n_particles)
                        .map(|_| Rc::new(Tree::new(self.init_leaf_value, y_data.len())))
                        .collect();
                    let state = BartState::new(particles, vec![1.0; self.n_particles]);
                    Ok(BartSampler::Nodes511 { sampler, state })
                }
                1023 => {
                    let sampler = ParticleGibbsSampler::new(
                        sampler_components.0,
                        sampler_components.1,
                        sampler_components.2,
                    );
                    let particles: Vec<Particle<1023>> = (0..self.n_particles)
                        .map(|_| Rc::new(Tree::new(self.init_leaf_value, y_data.len())))
                        .collect();
                    let state = BartState::new(particles, vec![1.0; self.n_particles]);
                    Ok(BartSampler::Nodes1023 { sampler, state })
                }
                2047 => {
                    let sampler = ParticleGibbsSampler::new(
                        sampler_components.0,
                        sampler_components.1,
                        sampler_components.2,
                    );
                    let particles: Vec<Particle<2047>> = (0..self.n_particles)
                        .map(|_| Rc::new(Tree::new(self.init_leaf_value, y_data.len())))
                        .collect();
                    let state = BartState::new(particles, vec![1.0; self.n_particles]);
                    Ok(BartSampler::Nodes2047 { sampler, state })
                }
                _ => Err(PyValueError::new_err(format!(
                    "Unsupported MAX_NODES: {}. Use 127, 255, 511, 1023, or 2047",
                    max_nodes
                ))),
            }
        };

        create_sampler_and_state(self.max_nodes)
    }
}
