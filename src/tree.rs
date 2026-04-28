use numpy::ndarray::{Array, ArrayView2, Ix1, Ix2};

/// Bartz-style heap-indexed tree with separate internal/leaf arrays.
///
/// Uses heap convention: root=0, left=2i+1, right=2i+2.
/// Internal nodes store split variable and threshold. Leaf nodes store
/// predicted values. The `leaf_indices` vector maps each training sample
/// to its assigned leaf node.
#[derive(Clone, Debug)]
pub struct TreeArrays {
    /// Split variable per node (usize::MAX = leaf sentinel)
    pub split_var: Vec<u32>,
    /// Split threshold per node (NaN for leaves)
    pub split_val: Vec<f64>,
    /// Leaf value per node (NaN for internal nodes)
    pub leaf_val: Vec<f64>,
    /// Sample -> leaf node mapping for training data
    pub leaf_indices: Vec<u32>,
    /// Number of allocated nodes in the tree
    pub size: usize,
    /// Maximum allowed depth
    pub max_depth: u8,
}

const LEAF_SENTINEL: u32 = u32::MAX;

impl TreeArrays {
    /// Create a new tree with just a root leaf node.
    pub fn new(init_leaf_value: f64, n_samples: usize, max_depth: u8) -> Self {
        let max_nodes = max_nodes_for_depth(max_depth);

        let mut split_var = Vec::with_capacity(max_nodes);
        let mut split_val = Vec::with_capacity(max_nodes);
        let mut leaf_val = Vec::with_capacity(max_nodes);

        // Root is a leaf
        split_var.push(LEAF_SENTINEL);
        split_val.push(f64::NAN);
        leaf_val.push(init_leaf_value);

        Self {
            split_var,
            split_val,
            leaf_val,
            leaf_indices: vec![0; n_samples],
            size: 1,
            max_depth,
        }
    }

    /// Get the depth of a node in the binary tree.
    ///
    /// Heap layout: depth(i) = floor(log2(i + 1)).
    #[inline]
    pub fn get_depth(&self, node_idx: usize) -> usize {
        63 - ((node_idx + 1) as u64).leading_zeros() as usize
    }

    /// Check if a node is a leaf (no split variable assigned).
    pub fn is_leaf(&self, node_idx: usize) -> bool {
        node_idx < self.split_var.len() && self.split_var[node_idx] == LEAF_SENTINEL
    }

    /// Get all leaf node indices.
    pub fn get_leaf_indices(&self) -> Vec<usize> {
        (0..self.size).filter(|&i| self.is_leaf(i)).collect()
    }

    /// Get data (sample) indices for a leaf node.
    pub fn get_leaf_samples(&self, leaf_idx: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.is_leaf(leaf_idx), "Node {} is not a leaf", leaf_idx);
        let leaf_idx = leaf_idx as u32;
        self.leaf_indices
            .iter()
            .enumerate()
            .filter_map(move |(sample_idx, &assigned_leaf)| {
                if assigned_leaf == leaf_idx {
                    Some(sample_idx)
                } else {
                    None
                }
            })
    }

    /// Split a leaf node into an internal node with two new child leaves.
    pub fn split_node(
        &mut self,
        leaf_idx: usize,
        split_var: u32,
        split_val: f64,
        left_val: f64,
        right_val: f64,
    ) {
        let left_child = 2 * leaf_idx + 1;
        let right_child = 2 * leaf_idx + 2;

        let max_nodes = max_nodes_for_depth(self.max_depth);
        assert!(
            right_child < max_nodes,
            "Tree mutation would exceed maximum capacity of {}",
            max_nodes
        );

        // Extend vectors if needed
        let required_size = right_child + 1;
        while self.split_var.len() < required_size {
            self.split_var.push(LEAF_SENTINEL);
            self.split_val.push(f64::NAN);
            self.leaf_val.push(0.0);
        }

        // Convert leaf to internal node
        self.split_var[leaf_idx] = split_var;
        self.split_val[leaf_idx] = split_val;
        self.leaf_val[leaf_idx] = f64::NAN;

        // Set left child as leaf
        self.split_var[left_child] = LEAF_SENTINEL;
        self.split_val[left_child] = f64::NAN;
        self.leaf_val[left_child] = left_val;

        // Set right child as leaf
        self.split_var[right_child] = LEAF_SENTINEL;
        self.split_val[right_child] = f64::NAN;
        self.leaf_val[right_child] = right_val;

        self.size = self.size.max(right_child + 1);
    }

    /// Update leaf assignments after a split using branchless bit trick.
    pub fn update_leaf_assignments(
        &mut self,
        split_node_idx: usize,
        split_var: u32,
        split_val: f64,
        affected_samples: &[usize],
        x_data: ArrayView2<f64>,
    ) {
        let base_child = (2 * split_node_idx + 1) as u32;

        for &sample_idx in affected_samples {
            let sample_val = x_data[[sample_idx, split_var as usize]];
            let child_offset = (sample_val >= split_val) as u32;
            self.leaf_indices[sample_idx] = base_child + child_offset;
        }
    }

    /// Predict training data using leaf_indices lookup.
    pub fn predict_training(&self) -> Array<f64, Ix1> {
        self.leaf_indices
            .iter()
            .map(|&leaf_idx| self.leaf_val[leaf_idx as usize])
            .collect()
    }

    /// Predict training data into a pre-allocated buffer.
    pub fn predict_training_into(&self, out: &mut Array<f64, Ix1>) {
        let leaf_val = &self.leaf_val;
        let out_slice = out
            .as_slice_mut()
            .expect("predictions buffer must be contiguous");
        for (dst, &leaf_idx) in out_slice.iter_mut().zip(self.leaf_indices.iter()) {
            // SAFETY: leaf_indices values always reference an existing entry in leaf_val.
            *dst = unsafe { *leaf_val.get_unchecked(leaf_idx as usize) };
        }
    }

    /// Predict on test data by traversing the tree.
    pub fn predict_batch_test(&self, data: &Array<f64, Ix2>) -> Array<f64, Ix1> {
        let mut predictions = Array::zeros(data.nrows());

        for (sample_idx, sample) in data.outer_iter().enumerate() {
            let mut node_idx = 0usize;

            while node_idx < self.size && !self.is_leaf(node_idx) {
                let sv = self.split_var[node_idx] as usize;
                let st = self.split_val[node_idx];

                if sample[sv] < st {
                    node_idx = 2 * node_idx + 1;
                } else {
                    node_idx = 2 * node_idx + 2;
                }
            }

            if node_idx < self.size && self.is_leaf(node_idx) {
                predictions[sample_idx] = self.leaf_val[node_idx];
            }
        }

        predictions
    }
}

/// Calculate maximum number of nodes for a given depth.
pub fn max_nodes_for_depth(depth: u8) -> usize {
    (1usize << (depth as usize + 1)) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tree_is_single_leaf() {
        let tree = TreeArrays::new(1.5, 10, 5);
        assert_eq!(tree.size, 1);
        assert!(tree.is_leaf(0));
        assert_eq!(tree.leaf_val[0], 1.5);
        assert_eq!(tree.leaf_indices.len(), 10);
        assert!(tree.leaf_indices.iter().all(|&idx| idx == 0));
    }

    #[test]
    fn test_split_node_creates_children() {
        let mut tree = TreeArrays::new(0.0, 5, 5);
        tree.split_node(0, 0, 2.5, -1.0, 1.0);

        assert!(!tree.is_leaf(0));
        assert!(tree.is_leaf(1));
        assert!(tree.is_leaf(2));
        assert_eq!(tree.leaf_val[1], -1.0);
        assert_eq!(tree.leaf_val[2], 1.0);
        assert_eq!(tree.size, 3);
    }

    #[test]
    fn test_get_depth() {
        let tree = TreeArrays::new(0.0, 1, 5);
        assert_eq!(tree.get_depth(0), 0);
        assert_eq!(tree.get_depth(1), 1);
        assert_eq!(tree.get_depth(2), 1);
        assert_eq!(tree.get_depth(3), 2);
        assert_eq!(tree.get_depth(6), 2);
    }

    #[test]
    fn test_max_nodes_for_depth() {
        assert_eq!(max_nodes_for_depth(0), 1);
        assert_eq!(max_nodes_for_depth(1), 3);
        assert_eq!(max_nodes_for_depth(2), 7);
        assert_eq!(max_nodes_for_depth(5), 63);
        assert_eq!(max_nodes_for_depth(6), 127);
        assert_eq!(max_nodes_for_depth(9), 1023);
    }

    #[test]
    fn test_predict_training_root_only() {
        let tree = TreeArrays::new(3.14, 4, 5);
        let preds = tree.predict_training();
        assert_eq!(preds.len(), 4);
        assert!(preds.iter().all(|&v| (v - 3.14).abs() < 1e-10));
    }

    #[test]
    fn test_get_leaf_samples() {
        let mut tree = TreeArrays::new(0.0, 4, 5);
        // Manually assign samples to different leaves
        tree.split_node(0, 0, 0.5, -1.0, 1.0);
        tree.leaf_indices = vec![1, 1, 2, 2]; // samples 0,1 -> left; 2,3 -> right

        let left_samples: Vec<usize> = tree.get_leaf_samples(1).collect();
        let right_samples: Vec<usize> = tree.get_leaf_samples(2).collect();

        assert_eq!(left_samples, vec![0, 1]);
        assert_eq!(right_samples, vec![2, 3]);
    }

    #[test]
    #[should_panic(expected = "Tree mutation would exceed maximum capacity")]
    fn test_split_beyond_max_depth_panics() {
        let mut tree = TreeArrays::new(0.0, 1, 1); // max_depth=1, max_nodes=3
        tree.split_node(0, 0, 0.5, -1.0, 1.0); // OK: creates nodes 1,2
        tree.split_node(1, 0, 0.3, -2.0, 2.0); // Panic: would need nodes 3,4 but max=3
    }
}
