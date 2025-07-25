use std::rc::Rc;

use rand::Rng;

use crate::base::BartState;
use crate::particle::Particle;
use crate::resampling::ResamplingStrategy;
use crate::update::{MutationDecision, TreeProposal, Update, Weight};

pub struct ParticleGibbsSampler<const MAX_NODES: usize, U, W, R> {
    update: U,
    weight: W,
    resample: R,
}

impl<const MAX_NODES: usize, U, W, R> ParticleGibbsSampler<MAX_NODES, U, W, R>
where
    U: Update<MAX_NODES, Proposal = TreeProposal>,
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
        let start = std::time::Instant::now();
        let mut particles = state.particles;

        let estimated_max_queue_size = (MAX_NODES / 4).max(16);
        let mut queues = ParticleQueues::new(particles.len(), estimated_max_queue_size);

        let mut iteration_count = 0;

        // Main growth loop with debugging
        while queues.has_any_non_empty() {
            iteration_count += 1;
            println!("\n=== Iteration {} ===", iteration_count);

            // Debug: Print current state before processing
            self.print_iteration_debug_info(&particles, &queues, iteration_count);

            // Process each particle
            particles.iter_mut()
                       .enumerate()
                       .for_each(|(i, particle)| {
                           if let Some(node_idx) = queues.pop_front(i) {
                               println!(
                                   "  Processing particle {}: attempting to grow node {} (Rc count: {})",
                                   i,
                                   node_idx,
                                   Rc::strong_count(particle)
                               );

                               match self.update.should_update(rng, particle, node_idx, context) {
                                   MutationDecision::Accept(proposal) => {
                                       println!(
                                           "    ✅ Particle {}: Mutation ACCEPTED for node {} -> split_var: {}, split_val: {:.3}",
                                           i, node_idx, proposal.split_var, proposal.split_val
                                       );

                                       // Apply mutation
                                       self.update.apply_update(particle, &proposal, context);

                                       // Add children to queue
                                       let left_child = 2 * node_idx + 1;
                                       let right_child = 2 * node_idx + 2;
                                       queues.push_back(i, left_child);
                                       queues.push_back(i, right_child);

                                       println!(
                                           "    📝 Added children nodes {} and {} to particle {} queue",
                                           left_child, right_child, i
                                       );
                                   }
                                   MutationDecision::Reject => {
                                       println!(
                                           "    ❌ Particle {}: Mutation REJECTED for node {} (removed from queue)",
                                           i, node_idx
                                       );
                                   }
                               }
                           }
                       });

            let weights: Vec<f64> = particles
                .iter()
                .map(|particle| self.weight.log_weight(particle, context))
                .collect();

            let norm_weights = normalize_weights(&weights);
            let ancestors = R::resample(rng, norm_weights);
            particles = ancestors
                .into_iter()
                .map(|idx| Rc::clone(&particles[idx]))
                .collect();

            // Debug: Print state after processing
            println!("  --- End of iteration {} ---", iteration_count);
            self.print_post_iteration_debug_info(&particles, &queues);
        }

        println!(
            "\n🏁 Growth phase completed after {} iterations",
            iteration_count
        );
        self.print_final_debug_info(&particles);

        // Final processing
        let weights: Vec<f64> = particles
            .iter()
            .map(|particle| self.weight.log_weight(particle, context))
            .collect();

        let norm_weights = normalize_weights(&weights);
        let ancestors = R::resample(rng, norm_weights);

        let resampled_particles: Vec<Particle<MAX_NODES>> = ancestors
            .into_iter()
            .map(|idx| Rc::clone(&particles[idx]))
            .collect();

        let duration = start.elapsed();
        println!("Rust finished stepping in {:?}", duration);

        BartState {
            particles: resampled_particles,
            weights: weights,
        }
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
            let available_nodes = queues.get_queue_contents(i);
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
        let total_expandable: usize = (0..particles.len()).map(|i| queues.get_queue_size(i)).sum();

        println!(
            "  📊 Total expandable nodes remaining: {}",
            total_expandable
        );

        // Show which particles still have expandable nodes
        let active_particles: Vec<usize> = (0..particles.len())
            .filter(|&i| !queues.is_empty(i))
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

pub fn normalize_weights(weights: &[f64]) -> impl Iterator<Item = f64> + '_ {
    // TODO: Do we have to clone here?
    let max_log_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = weights.iter().map(|&w| (w - max_log_weight).exp()).sum();

    weights
        .iter()
        .map(move |&w| (w - max_log_weight).exp() / sum_exp)
}

/// High-performance queue management using pre-allocated flat memory
#[derive(Debug)]
struct ParticleQueues {
    // Single flat buffer for all particle queues - excellent spatial locality
    buffer: Vec<usize>,
    // Track start/end indices for each particle's queue region
    queue_starts: Vec<usize>,
    queue_ends: Vec<usize>,
    // Current read positions (for FIFO behavior)
    read_positions: Vec<usize>,
    // Maximum queue size per particle (prevents reallocations)
    max_queue_size: usize,
}

impl ParticleQueues {
    fn new(num_particles: usize, max_queue_size: usize) -> Self {
        let total_capacity = num_particles * max_queue_size;
        let mut buffer = vec![0; total_capacity];

        let queue_starts: Vec<usize> = (0..num_particles).map(|i| i * max_queue_size).collect();

        let mut queue_ends = queue_starts.clone();
        let read_positions = queue_starts.clone();

        // Initialize each particle with root node (0)
        for i in 0..num_particles {
            buffer[queue_starts[i]] = 0; // Root node
            queue_ends[i] += 1;
        }

        Self {
            buffer,
            queue_starts,
            queue_ends,
            read_positions,
            max_queue_size,
        }
    }

    #[inline]
    fn pop_front(&mut self, particle_idx: usize) -> Option<usize> {
        if self.read_positions[particle_idx] >= self.queue_ends[particle_idx] {
            return None; // Queue empty
        }

        let value = self.buffer[self.read_positions[particle_idx]];
        self.read_positions[particle_idx] += 1;
        Some(value)
    }

    #[inline]
    fn push_back(&mut self, particle_idx: usize, value: usize) {
        debug_assert!(
            self.queue_ends[particle_idx] < self.queue_starts[particle_idx] + self.max_queue_size
        );
        self.buffer[self.queue_ends[particle_idx]] = value;
        self.queue_ends[particle_idx] += 1;
    }

    #[inline]
    fn is_empty(&self, particle_idx: usize) -> bool {
        self.read_positions[particle_idx] >= self.queue_ends[particle_idx]
    }

    #[inline]
    fn has_any_non_empty(&self) -> bool {
        self.read_positions
            .iter()
            .zip(self.queue_ends.iter())
            .any(|(&read_pos, &end_pos)| read_pos < end_pos)
    }

    /// Get the current contents of a particle's queue (for debugging)
    fn get_queue_contents(&self, particle_idx: usize) -> Vec<usize> {
        let start = self.read_positions[particle_idx];
        let end = self.queue_ends[particle_idx];

        if start >= end {
            Vec::new()
        } else {
            self.buffer[start..end].to_vec()
        }
    }

    /// Get the current size of a particle's queue
    fn get_queue_size(&self, particle_idx: usize) -> usize {
        self.queue_ends[particle_idx].saturating_sub(self.read_positions[particle_idx])
    }

    /// Get debug info for all queues
    fn debug_all_queues(&self) -> Vec<(usize, Vec<usize>, usize)> {
        (0..self.queue_starts.len())
            .map(|i| {
                let contents = self.get_queue_contents(i);
                let rc_count = 0; // Will be filled by caller
                (i, contents, rc_count)
            })
            .collect()
    }

    /// Print detailed queue state
    fn print_queue_debug(&self) {
        println!("🗂️  Queue Debug Info:");
        for i in 0..self.queue_starts.len() {
            let contents = self.get_queue_contents(i);
            let size = self.get_queue_size(i);
            println!("    Queue {}: size={}, contents={:?}", i, size, contents);
        }
    }
}
