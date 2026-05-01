//! Pure SMC step function for BART tree proposals.

use std::sync::Arc;

use numpy::ndarray::{Array, ArrayView1, Ix1};
use rand::Rng;
use rand::distr::weighted::WeightedIndex;
use rand_distr::{Distribution, Normal};

use crate::config::BartConfig;
use crate::data::DataView;
use crate::particle::Particle;
use crate::resampling::ResamplingStrategy;
use crate::splitting::SplitRules;
use crate::tree::TreeArrays;
use crate::update::{MutationDecision, TreeProposal};
use crate::weight::WeightFn;

/// Diagnostics from a single SMC tree step.
pub struct SmcStepInfo {
    pub log_likelihood: f64,
    pub acceptance_count: usize,
}

/// Run one SMC step to produce a new tree.
pub fn smc_step<R, W>(
    rng: &mut impl Rng,
    residuals: &Array<f64, Ix1>,
    config: &BartConfig,
    data: &DataView,
    split_rules: &[SplitRules],
    resampling: &R,
    weight_fn: &W,
) -> (TreeArrays, SmcStepInfo)
where
    R: ResamplingStrategy,
    W: WeightFn,
{
    let n_samples = data.n_samples();
    let init_leaf_value = data.y.mean().unwrap_or(0.0) / config.n_trees as f64;

    let mut particles: Vec<Particle> = (0..config.n_particles)
        .map(|i| {
            if i == 0 {
                Particle::new_reference(init_leaf_value, n_samples, config.max_depth)
            } else {
                Particle::new(init_leaf_value, n_samples, config.max_depth)
            }
        })
        .collect();

    let n_non_ref = config.n_particles - 1;
    let mut inner_weights = vec![0.0f64; n_non_ref];
    let mut acceptance_count = 0;

    let mut predictions_buf = Array::zeros(n_samples);
    let mut ancestors_buf: Vec<usize> = Vec::with_capacity(n_non_ref);
    let mut scratch_particles: Vec<Particle> = Vec::with_capacity(n_non_ref);
    let mut mutated = vec![false; n_non_ref];

    while particles[1..].iter().any(|p| p.has_expandable_nodes()) {
        mutated.iter_mut().for_each(|m| *m = false);
        for (i, particle) in particles[1..].iter_mut().enumerate() {
            if let Some(node_idx) = particle.peek_next_expandable() {
                let node_idx = node_idx as usize;

                match propose_mutation(
                    rng,
                    particle,
                    node_idx,
                    residuals,
                    config,
                    data,
                    split_rules,
                ) {
                    MutationDecision::Accept(proposal) => {
                        particle.pop_next_expandable();
                        particle.apply_mutation(&proposal, data.x);
                        acceptance_count += 1;
                        mutated[i] = true;
                    }
                    MutationDecision::Reject => {
                        particle.pop_next_expandable();
                    }
                }
            }
        }

        for (i, particle) in particles[1..].iter().enumerate() {
            if mutated[i] {
                particle.tree.predict_training_into(&mut predictions_buf);
                predictions_buf += residuals;
                inner_weights[i] = weight_fn.log_weight(&predictions_buf);
            }
        }
        normalize_weights_inplace(&mut inner_weights);

        resampling.resample_into(rng, &inner_weights, &mut ancestors_buf);
        scratch_particles.clear();
        scratch_particles.extend(ancestors_buf.iter().map(|&idx| particles[1 + idx].clone()));
        particles.truncate(1);
        particles.append(&mut scratch_particles);
    }

    let mut weights = vec![0.0f64; config.n_particles];
    for (i, particle) in particles.iter().enumerate() {
        particle.tree.predict_training_into(&mut predictions_buf);
        predictions_buf += residuals;
        weights[i] = weight_fn.log_weight(&predictions_buf);
    }
    normalize_weights_inplace(&mut weights);

    let dist = WeightedIndex::new(&weights).unwrap();
    let selected_idx = dist.sample(rng);

    let selected_particle = particles.swap_remove(selected_idx);
    // Drop remaining particles now so their Arc refs are released before try_unwrap.
    drop(particles);
    let final_tree = match Arc::try_unwrap(selected_particle.tree) {
        Ok(tree) => tree,
        Err(arc) => (*arc).clone(),
    };

    let info = SmcStepInfo {
        log_likelihood: weights[0],
        acceptance_count,
    };

    (final_tree, info)
}

/// Propose a mutation for a particle at a given node.
fn propose_mutation(
    rng: &mut impl Rng,
    particle: &Particle,
    node_idx: usize,
    ensemble_predictions: &Array<f64, Ix1>,
    config: &BartConfig,
    data: &DataView,
    split_rules: &[SplitRules],
) -> MutationDecision {
    let depth = particle.tree.get_depth(node_idx);
    let prob_not_expanding = 1.0 - (config.alpha * (1.0 + depth as f64).powf(-config.beta));

    if prob_not_expanding > rng.random::<f64>() {
        return MutationDecision::Reject;
    }

    let node_samples = particle.leaf_samples(node_idx);
    if node_samples.is_empty() {
        return MutationDecision::Reject;
    }

    let split_var = if let Some(ref probs) = config.splitting_probs {
        sample_feature_from_probs(rng, probs.as_slice().unwrap())
    } else {
        rng.random_range(0..data.n_features())
    };

    let col = data.x.column(split_var);
    let feature_values = node_samples
        .iter()
        .map(|&s| unsafe { *col.uget(s as usize) });

    let split_strategy = &split_rules[split_var];
    let split_val = match split_strategy.sample_split_value(rng, feature_values) {
        Some(v) => v,
        None => return MutationDecision::Reject,
    };

    let (left_value, right_value) = propose_leaf_values(
        rng,
        node_samples,
        &col,
        split_val,
        ensemble_predictions,
        config,
    );

    MutationDecision::Accept(TreeProposal {
        node_idx,
        split_var: split_var as u32,
        split_val,
        left_value,
        right_value,
    })
}

fn propose_leaf_values(
    rng: &mut impl Rng,
    node_samples: &[u32],
    col: &ArrayView1<f64>,
    split_val: f64,
    ensemble_predictions: &Array<f64, Ix1>,
    config: &BartConfig,
) -> (f64, f64) {
    let pred_slice = ensemble_predictions
        .as_slice()
        .expect("ensemble_predictions must be contiguous");

    let (left_sum, left_n, right_sum, right_n) = node_samples.iter().fold(
        (0.0, 0usize, 0.0, 0usize),
        |(mut l_sum, mut l_n, mut r_sum, mut r_n), &s| {
            let idx = s as usize;
            // SAFETY: indices come from particle.leaf_to_samples and are in [0, n_samples).
            let v = unsafe { *col.uget(idx) };
            let p = unsafe { *pred_slice.get_unchecked(idx) };
            if v < split_val {
                l_sum += p;
                l_n += 1;
            } else {
                r_sum += p;
                r_n += 1;
            }
            (l_sum, l_n, r_sum, r_n)
        },
    );

    let dist = Normal::new(0.0, 1.0).unwrap();

    let left_value = {
        let noise = dist.sample(rng) * config.sigma;
        if left_n == 0 {
            noise
        } else {
            left_sum / left_n as f64 / config.n_trees as f64 + noise
        }
    };

    let right_value = {
        let noise = dist.sample(rng) * config.sigma;
        if right_n == 0 {
            noise
        } else {
            right_sum / right_n as f64 / config.n_trees as f64 + noise
        }
    };

    (left_value, right_value)
}

fn sample_feature_from_probs(rng: &mut impl Rng, probs: &[f64]) -> usize {
    let total: f64 = probs.iter().sum();
    let mut target = rng.random::<f64>() * total;

    for (idx, &prob) in probs.iter().enumerate() {
        target -= prob;
        if target <= 0.0 {
            return idx;
        }
    }

    probs.len() - 1
}

/// Normalize log-weights in-place using log-sum-exp trick.
pub fn normalize_weights_inplace(weights: &mut [f64]) {
    let max_log_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for w in weights.iter_mut() {
        *w = (*w - max_log_weight).exp();
    }

    let sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= sum;
    }
}
