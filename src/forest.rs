use bumpalo::{collections::Vec, Bump};
use numpy::{ndarray::Array, Ix1, Ix2};

pub type SplitVariable = usize;
pub type SplitValue = f64;
pub type LeafValue = f64;
pub type LeafIndex = usize;

pub trait Predict {
    fn predict(&self, X: Array<f64, Ix2>) -> Array<f64, Ix1>;
}

/// A Forest owns the arena and passes a reference to it to a Tree.
#[derive(Debug)]
pub struct Tree<'arena> {
    split_var: Vec<'arena, SplitVariable>,
    split_value: Vec<'arena, SplitValue>,
    leaf_values: Vec<'arena, LeafValue>,
    leaf_indices: Vec<'arena, LeafIndex>,
    max_depth: usize,
}

impl<'arena> Tree<'arena> {
    fn stump(
        arena: &'arena Bump,
        init_leaf: LeafValue,
        n_samples: usize,
        max_depth: usize,
    ) -> Self {
        let max_leaf_nodes = 1 << (max_depth + 1); // 2^(depth)
        let max_internal_nodes = max_leaf_nodes - 1;

        // A stump starts with one leaf node
        let mut leaf_values = Vec::with_capacity_in(max_leaf_nodes, arena);
        leaf_values.push(init_leaf);

        // All samples initially point to the first leaf (index = 0)
        let leaf_indices = Vec::from_iter_in((0..n_samples).map(|_| 0), arena);

        // Pre-allocate remaining vectors with enough capacity for a full tree to avoid
        // reallocations
        Self {
            split_var: Vec::with_capacity_in(max_internal_nodes, arena),
            split_value: Vec::with_capacity_in(max_internal_nodes, arena),
            leaf_values,
            leaf_indices,
            max_depth,
        }
    }
}

/// Ephemeral set of Particles.
///
/// Represents the transient collection of candidate trees (particles) that are
/// being proposed and evaluated for a tree update in the ensemble of trees (forest).
#[derive(Debug)]
pub struct Forest<'arena> {
    arena: &'arena Bump,
    trees: Vec<'arena, Tree<'arena>>,
    weights: Vec<'arena, f64>,
}

impl<'arena> Forest<'arena> {
    pub fn new(arena: &'arena Bump, n_particles: usize) -> Self {
        Self {
            arena,
            trees: Vec::with_capacity_in(n_particles, arena),
            weights: Vec::with_capacity_in(n_particles, arena),
        }
    }

    /// Adds a new Particle (tree) to the Forest
    pub fn add_particle(&mut self, init_leaf: LeafValue, n_samples: usize, max_depth: usize) {
        let tree = Tree::stump(self.arena, init_leaf, n_samples, max_depth);
        self.trees.push(tree);
    }
}
