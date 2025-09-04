use std::collections::VecDeque;
use std::path::Ancestors;
use std::rc::Rc;

use numpy::ndarray::{Array, Array1, Ix1};
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

use crate::base::BartState;
use crate::forest::Forest;
use crate::particle::{Particle, Predict, Tree};
use crate::resampling::ResamplingStrategy;
use crate::update::{MutationDecision, TreeContext, TreeProposal, Update, Weight};

pub struct ParticleGibbsSampler<const MAX_NODES: usize, U, W, R> {
    update: U,
    weight: W,
    _resample: R,
    forest: Forest<MAX_NODES>,
}

impl<const MAX_NODES: usize, U, W, R> ParticleGibbsSampler<MAX_NODES, U, W, R>
where
    U: Update<MAX_NODES, Proposal = TreeProposal, Context = TreeContext>,
    W: Weight<MAX_NODES>,
    R: ResamplingStrategy,
{
    pub fn new(update: U, weight: W, resample: R, n_particles: usize, n_samples: usize) -> Self {
        Self {
            update,
            weight,
            _resample: resample,
            forest: Forest::new(n_particles, n_samples),
        }
    }

    // Mutates State in-place
    pub fn step_trees(
        &mut self,
        rng: &mut impl Rng,
        state: &mut BartState<MAX_NODES>,
        context: &U::Context,
    ) {
        let start = std::time::Instant::now();
        // println!(
        //     "\n🚀 Starting Particle Gibbs step for {} trees",
        //     context.n_trees
        // );

        // TODO: This should come from context or a settings?
        let init_leaf_value = context.y_data.mean().unwrap() as f64 / context.n_trees as f64;
        let n_particles = 5;
        state.ensemble_trees.clear();
        state.ensemble_predictions =
            Array1::from_elem(context.x_data.nrows(), context.y_data.mean().unwrap_or(0.0));

        // println!("{:?}", state.ensemble_predictions);

        let init_predictions = Array1::from_elem(context.x_data.nrows(), init_leaf_value);

        for tree_idx in 0..context.n_trees {
            // println!(
            //     "\n🌳 Processing tree {} of {}",
            //     tree_idx + 1,
            //     context.n_trees
            // );

            // Calculate residuals by removing current tree's contribution from the *current total ensemble prediction*.
            // let residuals = &state.ensemble_predictions - &current_tree.predict_training();
            let residuals = &state.ensemble_predictions - &init_predictions;

            let tree_start = std::time::Instant::now();
            // let new_tree = self.step_particles_old(
            //     rng,
            //     context,
            //     &state.ensemble_predictions,
            //     &residuals,
            //     n_particles,
            //     tree_idx,
            // );
            let new_tree = self.step_particles(
                rng,
                context,
                &state.ensemble_predictions,
                &residuals,
                n_particles,
                // tree_idx,
            );
            let tree_duration = tree_start.elapsed();

            // Update the total ensemble predictions directly.
            state.ensemble_predictions = &residuals + new_tree.predict_training();

            // println!("Ensemble predictions: {:?}", current_total_ensemble_preds);

            // println!(
            //     "✅ Tree {} completed in {:?} - New size: {}, leaves: {}",
            //     tree_idx + 1,
            //     tree_duration,
            //     new_tree.size,
            //     new_tree.get_leaf_indices().len(),
            // );

            // Add the new tree to the ensemble
            state.ensemble_trees.push(new_tree);
        }

        // println!("{:?}", state.ensemble_predictions);

        // let duration = start.elapsed();
        // println!("🏁 Particle Gibbs step completed in {:?}", duration);

        // BartState {
        //     ensemble_trees: updated_trees,
        //     ensemble_predictions: current_total_ensemble_preds,
        // }
    }

    // In sampler.rs -> ParticleGibbsSampler impl

    fn step_particles(
        &mut self, // Note: &mut self is now required
        rng: &mut impl Rng,
        context: &U::Context,
        ensemble_predictions: &Array<f64, Ix1>,
        residuals: &Array<f64, Ix1>,
        n_particles: usize,
    ) -> Tree<MAX_NODES> {
        // Reset the pool for the new tree
        let init_leaf_value = context.y_data.mean().unwrap() / context.n_trees as f64;
        self.forest.reset(init_leaf_value, context.y_data.len());

        self.forest
            .particles
            .iter()
            .enumerate()
            .for_each(|(i, particle)| {
                println!(
                    "init, particle: {}, count: {}",
                    i,
                    Rc::strong_count(&particle.tree)
                );
            });

        // while self.forest.has_expandable_particles() {
        for iter in 0..3 {
            // Process each particle (skipping reference particle)
            for (i, particle) in self.forest.particles.iter_mut().enumerate() {
                if let Some(node_idx) = particle.peek_next_expandable() {
                    println!(
                        "start of grow pass, particle: {:?}, count: {}, expandable nodes: {:?}",
                        i,
                        Rc::strong_count(&particle.tree),
                        particle.expandable_nodes
                    );
                    match self.update.should_update(
                        rng,
                        &particle.tree,
                        node_idx,
                        ensemble_predictions,
                        context,
                    ) {
                        MutationDecision::Accept(proposal) => {
                            // Remove node from queue (queue COW only)
                            particle.pop_next_expandable();

                            // Apply mutation (tree COW only if needed)
                            particle.apply_mutation(&proposal, context);

                            println!(
                                "   ✅ mutation, particle: {:?}, count: {}",
                                i,
                                Rc::strong_count(&particle.tree)
                            );
                        }
                        MutationDecision::Reject => {
                            // Only queue modification (queue COW only - tree untouched!)
                            particle.pop_next_expandable();

                            println!(
                                "   ❌ mutation, particle: {:?}, count: {}",
                                i,
                                Rc::strong_count(&particle.tree)
                            );
                        }
                    }

                    println!(
                        "end of grow pass, particle: {:?}, count: {}, expandable nodes: {:?}",
                        i,
                        Rc::strong_count(&particle.tree),
                        particle.expandable_nodes
                    );
                }
            }

            // Weight calculation - use tree references directly
            self.forest.weights.clear();
            for particle in &self.forest.particles {
                let predictions = residuals + &particle.tree.predict_training();
                self.forest
                    .weights
                    .push(self.weight.log_weight(&predictions));
            }

            println!("weights: {:?}", self.forest.weights);
            // self.forest
            //     .particles
            //     .iter()
            //     .enumerate()
            //     .for_each(|(i, particle)| {
            //         println!(
            //             "particle: {}, expandable nodes: {:?}",
            //             i, particle.expandable_nodes
            //         );
            //     });

            // Resampling - clone both Rcs but no data structures!
            // let weights: &[f64] = &self.forest.weights[1..];
            let norm_weights: Vec<f64> = normalize_weights(&self.forest.weights).collect();
            println!("len norm weights: {}", norm_weights.len());
            let ancestors: Vec<usize> = R::resample(rng, &norm_weights);

            // let ancestors: Vec<usize> = vec![1, 1, 1, 1, 1];

            // let ancestors: Vec<usize> = match iter {
            //     0 => vec![1, 1, 1, 1, 2],
            //     1 => vec![0, 1, 3, 3, 4],
            //     2 => vec![0, 2, 3, 4, 4],
            //     _ => panic!("Unknown"),
            // };

            println!("ancestors: {:?}", ancestors);

            self.forest.resample_particles(&ancestors);

            self.forest
                .particles
                .iter()
                .enumerate()
                .for_each(|(i, particle)| {
                    println!(
                        "after resampling, particle: {}, particle count: {}, expandable nodes: {:?}, count: {}",
                        i,
                        Rc::strong_count(&particle.tree),
                        particle.expandable_nodes,
                        Rc::strong_count(&particle.expandable_nodes),
                    );
                });
        }

        let norm_weights: Vec<f64> = normalize_weights(&self.forest.weights).collect();
        let dist = WeightedIndex::new(norm_weights).unwrap();
        let selected_idx = dist.sample(rng);

        // Extract final tree
        let selected_particle = self.forest.particles.swap_remove(selected_idx);
        self.forest.particles.clear(); // Ensure shared ownership is dropped

        println!(
            "selected tree count: {}",
            Rc::strong_count(&selected_particle.tree)
        );

        Rc::try_unwrap(selected_particle.tree).unwrap_or_else(|rc| (*rc).clone())
    }

    // // Runs particle Gibbs for a single tree position in the ensemble
    // fn step_particles_old(
    //     &mut self,
    //     rng: &mut impl Rng,
    //     context: &U::Context,
    //     ensemble_predictions: &Array<f64, Ix1>,
    //     residuals: &Array<f64, Ix1>,
    //     n_particles: usize,
    //     tree_idx: usize,
    // ) -> Tree<MAX_NODES> {
    //     // println!(
    //     //     "  🌱 Initializing {} particles for tree {}",
    //     //     n_particles,
    //     //     tree_idx + 1
    //     // );

    //     // Initialize particles: include current tree + (n_particles - 1) new particles
    //     let init_leaf_value = context.y_data.mean().unwrap() / context.n_trees as f64;

    //     let mut particles: Vec<Particle<MAX_NODES>> = (0..n_particles)
    //         .map(|i| {
    //             let particle = Rc::new(Tree::new(init_leaf_value, context.y_data.len()));
    //             println!(
    //                 "    Particle {}: initialized with Rc count {}",
    //                 i,
    //                 Rc::strong_count(&particle)
    //             );
    //             particle
    //         })
    //         .collect();

    //     // Initialize queues for particle growth
    //     let mut queues = ParticleQueues::new(n_particles);
    //     let mut iteration_count = 0;

    //     // Main growth loop with debugging
    //     // while queues.has_any_non_empty() {
    //     // iteration_count += 1;
    //     // println!("  🔄 Tree {} - Iteration {}", tree_idx + 1, iteration_count);

    //     // Debug: Print current state before processing
    //     // self.print_iteration_debug_info(&particles, &queues, iteration_count);

    //     // Process each particle
    //     for step in 0..5 {
    //         particles
    //             .iter_mut()
    //             .skip(1) // Reference particle trajectory
    //             .enumerate()
    //             .for_each(|(i, particle)| {
    //                 let actual_i = i + 1;
    //                 if let Some(node_idx) = queues.pop_front(actual_i) {
    //                     println!(
    //                         "    Growth pass of particle: {}, attempting to grow node: {}, Rc count: {}",
    //                         i,
    //                         node_idx,
    //                         Rc::strong_count(particle)
    //                     );

    //                     match self.update.should_update(
    //                         rng,
    //                         particle,
    //                         node_idx,
    //                         ensemble_predictions,
    //                         context,
    //                     ) {
    //                         MutationDecision::Accept(proposal) => {
    //                             println!(
    //                                 "      ✅ Particle {}: Mutation ACCEPTED for node {} -> split_var: {}, split_val: {:.3}",
    //                                 i, node_idx, proposal.split_var, proposal.split_val
    //                             );

    //                             // Apply mutation and update particle weight
    //                             let tree_mut = Rc::make_mut(particle);
    //                             self.update.apply_update(tree_mut, &proposal, context);

    //                             // Add children to queue
    //                             let left_child = 2 * node_idx + 1;
    //                             let right_child = 2 * node_idx + 2;
    //                             queues.push_back(actual_i, left_child);
    //                             queues.push_back(actual_i, right_child);

    //                             println!(
    //                                 "      📝 Particle: {}, mutated, count: {}, added children nodes {} and {} to particle {} queue",
    //                                 i, Rc::strong_count(&particle), left_child, right_child, i
    //                             );
    //                         }
    //                         MutationDecision::Reject => {
    //                             println!(
    //                                 "      ❌ Not mutating particle: {}, count: {},  mutation REJECTED for node {} (removed from queue)",
    //                                 i, Rc::strong_count(&particle),  node_idx
    //                             );
    //                         }
    //                     }
    //                 }
    //             });

    //         particles.iter().for_each(|particle| {
    //             println!("After mutating Rc count: {}", Rc::strong_count(&particle))
    //         });

    //         // After growing each particle for one growth iteration
    //         // compute weights, normalize, and resample the particles
    //         // TODO: Move this to a buffer
    //         let weights: Vec<f64> = particles
    //             .iter()
    //             // .skip(1)
    //             .map(|particle| {
    //                 let predictions = residuals + particle.predict_training();
    //                 let weight = self.weight.log_weight(&predictions);
    //                 weight
    //             })
    //             .collect();

    //         let norm_weights = normalize_weights(&weights);
    //         let ancestors: Vec<usize> = R::resample(rng, norm_weights).collect();
    //         println!("    🔄 Resampled indices: {:?}", ancestors);

    //         // TODO: I think this is wrong because we only initialized N - 1 queues
    //         // but it is possible to resample the skipped particle (see resampling below)

    //         queues.resample(&ancestors);

    //         particles = ancestors
    //             .into_iter()
    //             .map(|idx| Rc::clone(&particles[idx])) // Resample N particles
    //             .collect();

    //         particles.iter().enumerate().for_each(|(i, particle)| {
    //             println!(
    //                 "resampled particle: {}, Rc count: {}",
    //                 i,
    //                 Rc::strong_count(&particle)
    //             )
    //         });
    //     }

    //     // Debug: Print state after processing
    //     // println!("    --- End of iteration {} ---", iteration_count);
    //     // self.print_post_iteration_debug_info(&particles, &queues);
    //     // }

    //     // println!(
    //     // "  🏁 Tree {} growth phase completed after {} iterations",
    //     // tree_idx + 1,
    //     // iteration_count
    //     // );
    //     // self.print_final_debug_info(&particles);

    //     // Final selection phase
    //     // TODO: We should store these weights in a buffer
    //     let final_weights: Vec<f64> = particles
    //         .iter()
    //         .map(|particle| {
    //             let predictions = residuals + particle.predict_training();
    //             let weight = self.weight.log_weight(&predictions);
    //             weight
    //         })
    //         .collect();

    //     // println!(" 📊 Final particle weights: {:?}", final_weights);

    //     let norm_weights: Vec<f64> = normalize_weights(&final_weights).collect();
    //     // println!("normalized weights: {:?}", norm_weights);

    //     let dist = WeightedIndex::new(norm_weights.clone()).unwrap();
    //     let selected_idx = dist.sample(rng);

    //     // println!(
    //     //     "  🎯 Selected particle {} with normalized weight {:.4}",
    //     //     selected_idx, norm_weights[selected_idx]
    //     // );

    //     // TODO: Unnessary allocation
    //     let mut particles = particles;
    //     let selected_particle = particles.swap_remove(selected_idx);

    //     let final_tree = Rc::try_unwrap(selected_particle).unwrap_or_else(|rc| (*rc).clone());

    //     // println!(
    //     //     "  ✨ Final tree for tree {}: size={}, leaves={}",
    //     //     tree_idx + 1,
    //     //     final_tree.size,
    //     //     final_tree.get_leaf_indices().len()
    //     // );

    //     final_tree
    // }

    // /// Print debugging information at the start of each iteration
    // fn print_iteration_debug_info(&self, tree_idx: usize, iteration: usize) {
    //     println!("🔍 Tree {} - Iteration {} State:", tree_idx + 1, iteration);

    //     for (i, particle) in self.particle_pool.particles().iter().enumerate() {
    //         let rc_count = Rc::strong_count(particle);
    //         let tree_size = particle.size;
    //         let total_leaves = particle.get_leaf_indices();

    //         println!(
    //             "  Particle {}: Rc={}, TreeSize={}, TotalLeaves={}",
    //             i,
    //             rc_count,
    //             tree_size,
    //             total_leaves.len(),
    //         );
    //     }
    // }

    // /// Print debugging information after processing each iteration
    // fn print_post_iteration_debug_info(&mut self) {
    //     let total_expandable: usize = self
    //         .particle_pool
    //         .queues_mut()
    //         .iter()
    //         .map(|q| q.len())
    //         .sum();

    //     println!(
    //         "  📊 Total expandable nodes remaining: {}",
    //         total_expandable
    //     );

    //     // Show which particles still have expandable nodes
    //     let active_particles: Vec<usize> = self
    //         .particle_pool
    //         .queues_mut()
    //         .iter()
    //         .enumerate()
    //         .filter_map(|(i, q)| if !q.is_empty() { Some(i) } else { None })
    //         .collect();

    //     if !active_particles.is_empty() {
    //         println!("  🔄 Active particles: {:?}", active_particles);
    //     }
    // }

    // /// Print final debugging information
    // fn print_final_debug_info(&self) {
    //     println!("📈 Final Particle Summary:");
    //     for (i, particle) in self.particle_pool.particles().iter().enumerate() {
    //         let rc_count = Rc::strong_count(particle);
    //         let tree_size = particle.size;
    //         let leaf_count = particle.get_leaf_indices().len();

    //         println!(
    //             "  Particle {}: Rc={}, FinalSize={}, FinalLeaves={}",
    //             i, rc_count, tree_size, leaf_count
    //         );
    //     }
    // }
}

pub fn normalize_weights(weights: &[f64]) -> impl Iterator<Item = f64> + '_ {
    let max_log_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = weights.iter().map(|&w| (w - max_log_weight).exp()).sum();

    weights
        .iter()
        .map(move |&w| (w - max_log_weight).exp() / sum_exp)
}

#[derive(Debug)]
struct ParticleQueues {
    queues: Vec<VecDeque<usize>>,
}

impl ParticleQueues {
    /// Creates a new `ParticleQueues` manager for a given number of particles.
    ///
    /// Each particle's queue is initialized with the root node (index 0),
    /// making it ready for the first growth step.
    ///
    /// # Arguments
    /// * `num_particles` - The total number of particles to manage.
    pub fn new(num_particles: usize) -> Self {
        let queues = (0..num_particles)
            .map(|i| {
                if i == 0 {
                    // First particle queue starts empty - it won't be mutated
                    VecDeque::new()
                } else {
                    // Other particles start with the root node (0) in their queue
                    VecDeque::from([0])
                }
            })
            .collect();

        Self { queues }
    }

    /// Removes and returns the next node index from the front of a particle's queue.
    ///
    /// Returns `None` if the specified particle's queue is empty.
    /// This is an O(1) operation.
    #[inline]
    pub fn pop_front(&mut self, particle_idx: usize) -> Option<usize> {
        self.queues
            .get_mut(particle_idx)
            .and_then(|q| q.pop_front())
    }

    /// Adds a new node index to the back of a particle's queue.
    ///
    /// This is an amortized O(1) operation.
    #[inline]
    pub fn push_back(&mut self, particle_idx: usize, value: usize) {
        if let Some(q) = self.queues.get_mut(particle_idx) {
            q.push_back(value);
        }
    }

    /// Checks if any particle has a non-empty queue.
    ///
    /// This is used to determine if the main algorithm loop should continue.
    /// It returns `true` as long as there is at least one expandable leaf
    /// in any particle's tree.
    #[inline]
    pub fn has_any_non_empty(&self) -> bool {
        self.queues.iter().any(|q| !q.is_empty())
    }

    /// Resamples the queues based on a list of ancestor indices.
    ///
    /// This is the critical operation for synchronizing the queue state with the
    /// particle state after the resampling step of the SMC algorithm. It creates
    /// a new set of queues by cloning the queues of the chosen ancestors.
    ///
    /// # Arguments
    /// * `ancestors` - A slice of indices indicating which parent particles were
    ///   selected for the next generation.
    pub fn resample(&mut self, ancestors: &[usize]) {
        self.queues = ancestors
            .iter()
            .map(|&ancestor_idx| self.queues[ancestor_idx].clone())
            .collect();
    }

    /// Returns the current contents of a specific particle's queue for debugging.
    pub fn get_queue_contents(&self, particle_idx: usize) -> Option<Vec<usize>> {
        self.queues
            .get(particle_idx)
            .map(|q| q.iter().cloned().collect())
    }

    /// Returns the number of items in a specific particle's queue.
    pub fn queue_len(&self, particle_idx: usize) -> usize {
        self.queues.get(particle_idx).map(|q| q.len()).unwrap_or(0)
    }

    /// Checks if a specific particle's queue is empty.
    pub fn is_queue_empty(&self, particle_idx: usize) -> bool {
        self.queues
            .get(particle_idx)
            .map(|q| q.is_empty())
            .unwrap_or(true)
    }
}

pub fn normalize_weights_inplace(weights: &mut [f64]) {
    let max_log_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Convert to probabilities
    for w in weights.iter_mut() {
        *w = (*w - max_log_weight).exp();
    }

    // Normalize
    let sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= sum;
    }
}
