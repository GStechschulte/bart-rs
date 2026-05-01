use std::collections::VecDeque;
use std::sync::Arc;

use numpy::ndarray::ArrayView2;

use crate::tree::{TreeArrays, max_nodes_for_depth};
use crate::update::TreeProposal;

/// Flat CSR-style mapping from leaf node index to sample indices.
///
/// All sample indices live in one contiguous `data` Vec. Each leaf's samples
/// occupy `data[node_start[i]..node_start[i] + node_len[i]]`.
#[derive(Clone, Debug)]
pub struct LeafSamplesFlat {
    /// Flat storage of sample indices. Layout is determined by `node_start`/`node_len`.
    pub data: Vec<u32>,
    /// Start offset in `data` for each node (0 for internal/unused nodes).
    pub node_start: Vec<u32>,
    /// Sample count for each node (0 for internal nodes after a split).
    pub node_len: Vec<u32>,
}

impl LeafSamplesFlat {
    pub fn new(n_samples: usize, max_depth: u8) -> Self {
        let max_nodes = max_nodes_for_depth(max_depth);
        let data: Vec<u32> = (0..n_samples as u32).collect();
        let mut node_start = vec![0u32; max_nodes];
        let mut node_len = vec![0u32; max_nodes];
        node_start[0] = 0;
        node_len[0] = n_samples as u32;
        LeafSamplesFlat {
            data,
            node_start,
            node_len,
        }
    }

    #[inline]
    pub fn samples(&self, node_idx: usize) -> &[u32] {
        let start = self.node_start[node_idx] as usize;
        let len = self.node_len[node_idx] as usize;
        &self.data[start..start + len]
    }
}

#[derive(Debug, Clone)]
pub struct Particle {
    pub tree: Arc<TreeArrays>,
    pub expandable_nodes: VecDeque<u32>,
    pub sample_map: LeafSamplesFlat,
}

impl Particle {
    pub fn new(init_leaf_value: f64, n_samples: usize, max_depth: u8) -> Self {
        let tree = Arc::new(TreeArrays::new(init_leaf_value, n_samples, max_depth));
        Self {
            tree,
            expandable_nodes: VecDeque::from([0]),
            sample_map: LeafSamplesFlat::new(n_samples, max_depth),
        }
    }

    pub fn new_reference(init_leaf_value: f64, n_samples: usize, max_depth: u8) -> Self {
        let tree = Arc::new(TreeArrays::new(init_leaf_value, n_samples, max_depth));
        Self {
            tree,
            expandable_nodes: VecDeque::new(),
            sample_map: LeafSamplesFlat::new(n_samples, max_depth),
        }
    }

    pub fn has_expandable_nodes(&self) -> bool {
        !self.expandable_nodes.is_empty()
    }

    pub fn peek_next_expandable(&self) -> Option<u32> {
        self.expandable_nodes.front().copied()
    }

    pub fn pop_next_expandable(&mut self) -> Option<u32> {
        self.expandable_nodes.pop_front()
    }

    pub fn leaf_samples(&self, leaf_idx: usize) -> &[u32] {
        self.sample_map.samples(leaf_idx)
    }

    /// Apply a mutation is a COW-clone if shared, split the node, and
    /// partition the parent's sample slice in-place while simultaneously
    /// updating `leaf_indices`.
    pub fn apply_mutation(&mut self, proposal: &TreeProposal, x_data: ArrayView2<f64>) {
        let node_idx = proposal.node_idx;
        let split_var = proposal.split_var as usize;
        let split_val = proposal.split_val;
        let left_child = 2 * node_idx + 1;
        let right_child = 2 * node_idx + 2;
        let lc = left_child as u32;
        let rc = right_child as u32;

        // COW clones TreeArrays on the heap only when Arc is shared (refcount > 1).
        let tree = Arc::make_mut(&mut self.tree);
        tree.split_node(
            node_idx,
            proposal.split_var,
            split_val,
            proposal.left_value,
            proposal.right_value,
        );

        let start = self.sample_map.node_start[node_idx] as usize;
        let len = self.sample_map.node_len[node_idx] as usize;
        let col = x_data.column(split_var);

        // In-place two-pointer partition: samples going left accumulate at the
        // front, samples going right at the back. leaf_indices is updated in
        // the same pass — no extra allocation needed.
        let left_count = {
            let slice = &mut self.sample_map.data[start..start + len];
            let leaf_indices = &mut tree.leaf_indices;
            let mut l = 0usize;
            let mut r = len;
            while l < r {
                let s = slice[l];
                let idx = s as usize;
                // SAFETY: sample indices are in [0, n_samples); col and leaf_indices
                // both have length n_samples by construction.
                let v = unsafe { *col.uget(idx) };
                if v < split_val {
                    unsafe { *leaf_indices.get_unchecked_mut(idx) = lc }
                    l += 1;
                } else {
                    unsafe { *leaf_indices.get_unchecked_mut(idx) = rc }
                    r -= 1;
                    slice.swap(l, r);
                }
            }
            l
        };

        self.sample_map.node_start[left_child] = start as u32;
        self.sample_map.node_len[left_child] = left_count as u32;
        self.sample_map.node_start[right_child] = (start + left_count) as u32;
        self.sample_map.node_len[right_child] = (len - left_count) as u32;
        self.sample_map.node_len[node_idx] = 0;

        self.expandable_nodes.push_back(left_child as u32);
        self.expandable_nodes.push_back(right_child as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array2;

    #[test]
    fn test_leaf_samples_flat_new() {
        let m = LeafSamplesFlat::new(5, 3);
        assert_eq!(m.samples(0), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_particle_clone_shares_arc() {
        let p = Particle::new(0.0, 10, 3);
        let q = p.clone();
        assert!(Arc::ptr_eq(&p.tree, &q.tree));
    }

    #[test]
    fn test_apply_mutation_partitions_correctly() {
        // x: 5 samples, 1 feature: [0, 1, 2, 3, 4]
        let x = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
        let mut p = Particle::new(0.0, 5, 3);

        let proposal = TreeProposal {
            node_idx: 0,
            split_var: 0,
            split_val: 2.5,
            left_value: -1.0,
            right_value: 1.0,
        };
        p.apply_mutation(&proposal, x.view());

        // Samples 0,1,2 go left (< 2.5); samples 3,4 go right.
        let mut left: Vec<u32> = p.leaf_samples(1).to_vec();
        let mut right: Vec<u32> = p.leaf_samples(2).to_vec();
        left.sort_unstable();
        right.sort_unstable();
        assert_eq!(left, vec![0, 1, 2]);
        assert_eq!(right, vec![3, 4]);

        // leaf_indices updated correctly.
        assert_eq!(p.tree.leaf_indices[0], 1);
        assert_eq!(p.tree.leaf_indices[1], 1);
        assert_eq!(p.tree.leaf_indices[2], 1);
        assert_eq!(p.tree.leaf_indices[3], 2);
        assert_eq!(p.tree.leaf_indices[4], 2);

        // Arc is unique after mutation (was unique from new()).
        assert_eq!(Arc::strong_count(&p.tree), 1);
    }

    #[test]
    fn test_apply_mutation_cow_unshares_arc() {
        let x = Array2::from_shape_fn((4, 1), |(i, _)| i as f64);
        let p = Particle::new(0.0, 4, 3);
        let mut q = p.clone();

        // p and q share the same Arc before mutation.
        assert!(Arc::ptr_eq(&p.tree, &q.tree));

        let proposal = TreeProposal {
            node_idx: 0,
            split_var: 0,
            split_val: 2.0,
            left_value: -1.0,
            right_value: 1.0,
        };
        q.apply_mutation(&proposal, x.view());

        // After mutation q has its own tree; p's tree is unchanged.
        assert!(!Arc::ptr_eq(&p.tree, &q.tree));
        assert!(
            p.tree.is_leaf(0),
            "original particle tree should still be root-only"
        );
        assert!(!q.tree.is_leaf(0), "mutated particle tree should be split");
    }

    #[test]
    fn test_flat_samples_all_in_data() {
        let n = 8;
        let m = LeafSamplesFlat::new(n, 3);
        // All sample indices in data, summing to n*(n-1)/2.
        let sum: u32 = m.data.iter().sum();
        assert_eq!(sum, (n * (n - 1) / 2) as u32);
    }
}
