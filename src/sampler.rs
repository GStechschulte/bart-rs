use std::rc::Rc;

use rand::Rng;

use crate::base::BartState;
use crate::particle::Particle;
use crate::resampling::ResamplingStrategy;
use crate::update::{BARTProposal, MutationDecision, Update, Weight};

pub struct ParticleGibbsSampler<const MAX_NODES: usize, U, W, R> {
    update: U,
    weight: W,
    resample: R,
}

impl<const MAX_NODES: usize, U, W, R> ParticleGibbsSampler<MAX_NODES, U, W, R>
where
    U: Update<MAX_NODES, Proposal = BARTProposal>,
    W: Weight<MAX_NODES, Context = U::Context>,
    R: ResamplingStrategy,
{
    pub fn new(update: U, weight: W, resample: R) -> Self {
        Self {
            update,
            weight,
            resample,
        }
    }

    pub fn step(
        &self,
        rng: &mut impl Rng,
        state: BartState<MAX_NODES>,
        context: &U::Context,
    ) -> BartState<MAX_NODES> {
        let mut new_particles = Vec::with_capacity(state.particles.len());
        let mut proposals = Vec::with_capacity(state.particles.len());

        // Conditional mutation stage
        for (i, particle) in state.particles.into_iter().enumerate() {
            // let rng_key = rng.wrapping_add(i as u64);

            match self.update.should_mutate(rng, &particle, context) {
                MutationDecision::Accept(proposal) => {
                    let mut particle_clone = particle;
                    self.update
                        .apply_mutation(&mut particle_clone, &proposal, context);
                    new_particles.push(particle_clone);
                    proposals.push(Some(proposal));
                }
                MutationDecision::Reject => {
                    new_particles.push(particle);
                    proposals.push(None);
                }
            }
        }

        // Weight calculation stage
        let log_weights: Vec<f64> = new_particles
            .iter()
            .zip(&proposals)
            .map(|(particle, proposal_opt)| {
                if let Some(proposal) = proposal_opt {
                    self.weight.log_weight(particle, context)
                } else {
                    0.0 // Base weight for rejected mutations
                }
            })
            .collect();

        // Weight normalization
        let max_log_weight = log_weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let weights: Vec<f64> = log_weights
            .iter()
            .map(|&lw| (lw - max_log_weight).exp())
            .collect();

        // Resampling stage
        let ancestors = R::resample(rng, &weights);

        // TODO: This is wrong!!
        let resampled_particles: Vec<Particle<MAX_NODES>> = ancestors
            .iter()
            .map(|&idx| Rc::clone(&new_particles[idx]))
            .collect();

        BartState {
            particles: resampled_particles,
            weights,
        }
    }

    /// Execute multiple SMC iterations
    pub fn run(
        &self,
        rng: &mut impl Rng,
        initial_state: BartState<MAX_NODES>,
        context: &U::Context,
        n_iterations: usize,
    ) -> BartState<MAX_NODES> {
        (0..n_iterations).fold(initial_state, |state, _| self.step(rng, state, context))
    }
}
