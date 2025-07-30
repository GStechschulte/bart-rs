use std::collections::VecDeque;
use std::rc::Rc;

use numpy::ndarray::{Array, Ix1};
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

use crate::base::BartState;
use crate::particle::{Particle, Predict, Tree};
use crate::resampling::ResamplingStrategy;
use crate::update::{MutationDecision, TreeContext, TreeProposal, Update, Weight};

pub struct ParticleGibbsSampler<const MAX_NODES: usize, U, W, R> {
    update: U,
    weight: W,
    resample: R,
}

impl<const MAX_NODES: usize, U, W, R> ParticleGibbsSampler<MAX_NODES, U, W, R>
where
    U: Update<MAX_NODES, Proposal = TreeProposal, Context = TreeContext>,
    W: Weight<MAX_NODES>,
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
        let start = std::time::Instant::now();
        // println!(
        //     "\n🚀 Starting Particle Gibbs step for {} trees",
        //     context.n_trees
        // );

        let n_particles = 20;
        let mut updated_trees: Vec<Tree<MAX_NODES>> = Vec::with_capacity(context.n_trees);
        // let mut trees = state.ensemble_trees;
        // println!(
        //     "init ensemble predictions: {:?}",
        //     state.ensemble_predictions
        // );
        let mut current_total_ensemble_preds = state.ensemble_predictions;
        let original_targets = context.y_data.clone();
        let mut tree_context = context.clone();

        // println!("init_sum_trees: {:?}", current_total_ensemble_preds);

        for tree_idx in 0..context.n_trees {
            // for tree_idx in 0..1 {
            let current_tree = &state.ensemble_trees[tree_idx];

            // println!(
            //     "\n🌳 Processing tree {} of {}",
            //     tree_idx + 1,
            //     context.n_trees
            // );

            // println!(
            //     "📊 Current tree stats: size={}, leaves={}",
            //     current_tree.size,
            //     current_tree.get_leaf_indices().len()
            // );

            // Calculate residuals by removing current tree's contribution from the *current total ensemble prediction*.
            let current_tree_prediction = current_tree.predict_training();
            let residuals = &current_total_ensemble_preds - &current_tree_prediction;
            // println!("residual: {:?}", residuals);

            tree_context.y_data = current_total_ensemble_preds;

            // println!("current_tree_prediction: {:?}", current_tree_prediction);
            // println!("residuals: {:?}", residuals);

            // Run particle Gibbs step for this tree
            let tree_start = std::time::Instant::now();
            let new_tree = self.step_particles(rng, context, &residuals, n_particles, tree_idx);
            let tree_duration = tree_start.elapsed();

            let new_tree_predictions = new_tree.predict_training();

            // Update the total ensemble predictions directly.
            current_total_ensemble_preds = &residuals + &new_tree_predictions;

            // println!("Ensemble predictions: {:?}", current_total_ensemble_preds);

            // println!(
            //     "✅ Tree {} completed in {:?} - New size: {}, leaves: {}, new_tree_preds: {:?}, predictions: {:?}",
            //     tree_idx + 1,
            //     tree_duration,
            //     new_tree.size,
            //     new_tree.get_leaf_indices().len(),
            //     new_tree_predictions,
            //     current_total_ensemble_preds
            // );

            updated_trees.push(new_tree); // Add the new tree to our
        }

        let duration = start.elapsed();
        // println!("🏁 Particle Gibbs step completed in {:?}", duration);

        BartState {
            ensemble_trees: updated_trees,
            ensemble_predictions: current_total_ensemble_preds,
        }
    }

    /// Runs particle Gibbs for a single tree position in the ensemble
    fn step_particles(
        &self,
        rng: &mut impl Rng,
        context: &U::Context,
        residuals: &Array<f64, Ix1>,
        n_particles: usize,
        tree_idx: usize,
    ) -> Tree<MAX_NODES> {
        // println!(
        //     "  🌱 Initializing {} particles for tree {}",
        //     n_particles,
        //     tree_idx + 1
        // );

        // Initialize particles: include current tree + (n_particles - 1) new particles
        let init_leaf_value = context.y_data.mean().unwrap() / context.n_trees as f64;
        let mut particles: Vec<Particle<MAX_NODES>> = (0..n_particles)
            .map(|i| {
                let particle = Rc::new(Tree::new(init_leaf_value, context.y_data.len()));
                // println!(
                //     "    Particle {}: initialized with Rc count {}",
                //     i,
                //     Rc::strong_count(&particle)
                // );
                particle
            })
            .collect();

        // Initialize queues for particle growth
        let mut queues = ParticleQueues::new(n_particles);
        let mut iteration_count = 0;

        // Main growth loop with debugging
        while queues.has_any_non_empty() {
            iteration_count += 1;
            // println!("  🔄 Tree {} - Iteration {}", tree_idx + 1, iteration_count);

            // Debug: Print current state before processing
            // self.print_iteration_debug_info(&particles, &queues, iteration_count);

            // Process each particle
            particles.iter_mut().enumerate().for_each(|(i, particle)| {
                if let Some(node_idx) = queues.pop_front(i) {
                    // println!(
                    //     "    Processing particle {}: attempting to grow node {} (Rc count: {})",
                    //     i,
                    //     node_idx,
                    //     Rc::strong_count(particle)
                    // );

                    match self.update.should_update(rng, particle, node_idx, context) {
                        MutationDecision::Accept(proposal) => {
                            // println!(
                            //     "      ✅ Particle {}: Mutation ACCEPTED for node {} -> split_var: {}, split_val: {:.3}",
                            //     i, node_idx, proposal.split_var, proposal.split_val
                            // );

                            // Apply mutation and update particle weight
                            let tree_mut = Rc::make_mut(particle);
                            self.update.apply_update(tree_mut, &proposal, context);

                            // Add children to queue
                            let left_child = 2 * node_idx + 1;
                            let right_child = 2 * node_idx + 2;
                            queues.push_back(i, left_child);
                            queues.push_back(i, right_child);

                            // println!(
                            //     "      📝 Added children nodes {} and {} to particle {} queue",
                            //     left_child, right_child, i
                            // );
                        }
                        MutationDecision::Reject => {
                            // println!(
                            //     "      ❌ Particle {}: Mutation REJECTED for node {} (removed from queue)",
                            //     i, node_idx
                            // );
                        }
                    }
                }
            });

            // After growing each particle for one growth iteration
            // compute weights, normalize, and resample the particles
            let weights: Vec<f64> = particles
                .iter()
                .map(|particle| {
                    let predictions = residuals + particle.predict_training();
                    let weight = self.weight.log_weight(&predictions);
                    weight
                })
                .collect();

            // println!("    📊 Particle weights: {:?}", weights);

            let norm_weights = normalize_weights(&weights);

            let ancestors: Vec<usize> = R::resample(rng, norm_weights).collect();

            // println!("    🔄 Resampling ancestors: {:?}", ancestors);

            queues.resample(&ancestors);

            particles = ancestors
                .into_iter()
                .map(|idx| Rc::clone(&particles[idx]))
                .collect();

            // Debug: Print state after processing
            // println!("    --- End of iteration {} ---", iteration_count);
            // self.print_post_iteration_debug_info(&particles, &queues);
        }

        // println!(
        // "  🏁 Tree {} growth phase completed after {} iterations",
        // tree_idx + 1,
        // iteration_count
        // );
        // self.print_final_debug_info(&particles);

        // Final selection phase
        // TODO: We should store these weights somewhere instead of computing again
        let final_weights: Vec<f64> = particles
            .iter()
            .map(|particle| {
                let predictions = residuals + particle.predict_training();
                let weight = self.weight.log_weight(&predictions);
                weight
            })
            .collect();

        // println!(" 📊 Final particle weights: {:?}", final_weights);

        let norm_weights: Vec<f64> = normalize_weights(&final_weights).collect();
        // println!("normalized weights: {:?}", norm_weights);

        let dist = WeightedIndex::new(norm_weights.clone()).unwrap();
        let selected_idx = dist.sample(rng);

        // println!(
        //     "  🎯 Selected particle {} with normalized weight {:.4}",
        //     selected_idx, norm_weights[selected_idx]
        // );

        let mut particles = particles;
        let selected_particle = particles.swap_remove(selected_idx);

        let final_tree = Rc::try_unwrap(selected_particle).unwrap_or_else(|rc| (*rc).clone());

        // println!(
        //     "  ✨ Final tree for tree {}: size={}, leaves={}",
        //     tree_idx + 1,
        //     final_tree.size,
        //     final_tree.get_leaf_indices().len()
        // );

        final_tree
    }

    /// Print debugging information at the start of each iteration
    fn print_iteration_debug_info(
        &self,
        particles: &[Particle<MAX_NODES>],
        queues: &ParticleQueues,
        iteration: usize,
    ) {
        println!("🔍 Iteration {} State:", iteration);

        for (i, particle) in particles.iter().enumerate() {
            let available_nodes = queues.get_queue_contents(i).unwrap_or_default();
            let rc_count = Rc::strong_count(particle);
            let tree_size = particle.size;
            let total_leaves = particle.get_leaf_indices();

            println!(
                "  Particle {}: Rc={}, TreeSize={}, TotalLeaves={}, ExpandableNodes={:?}",
                i,
                rc_count,
                tree_size,
                total_leaves.len(),
                available_nodes
            );
        }
    }

    /// Print debugging information after processing each iteration
    fn print_post_iteration_debug_info(
        &self,
        particles: &[Particle<MAX_NODES>],
        queues: &ParticleQueues,
    ) {
        let total_expandable: usize = (0..particles.len()).map(|i| queues.queue_len(i)).sum();

        println!(
            "  📊 Total expandable nodes remaining: {}",
            total_expandable
        );

        // Show which particles still have expandable nodes
        let active_particles: Vec<usize> = (0..particles.len())
            .filter(|&i| !queues.is_queue_empty(i))
            .collect();

        if !active_particles.is_empty() {
            println!("  🔄 Active particles: {:?}", active_particles);
        }
    }

    /// Print final debugging information
    fn print_final_debug_info(&self, particles: &[Particle<MAX_NODES>]) {
        println!("📈 Final Particle Summary:");
        for (i, particle) in particles.iter().enumerate() {
            let rc_count = Rc::strong_count(particle);
            let tree_size = particle.size;
            let leaf_count = particle.get_leaf_indices().len();

            println!(
                "  Particle {}: Rc={}, FinalSize={}, FinalLeaves={}",
                i, rc_count, tree_size, leaf_count
            );
        }
    }
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
            .map(|_| {
                // Each particle starts with the root node (0) in its queue.
                // VecDeque::from creates a deque from an array slice.
                VecDeque::from([0])
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
