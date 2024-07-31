use std::cmp::Ordering;

/// A `DecisionTree` structure is implemented as a number of parallel
/// vectors. Using parallel vectors allows for a more cache-efficient
/// implementation. Moreover, index accessing allows one to avoid borrow
/// checker issues related to recursive binary tree implementations.
///
/// The i-th element of each vector holds information about
/// node `i`. Node 0 is the tree's root. Some of the arrays only apply
/// to either leaves or split nodes. In this case, the values of the
/// nodes of the other vector is arbitrary. For example, `feature` and
/// `threshold` vectors only apply to split nodes. The values for leaf
/// nodes in these arrays are therefore arbitrary. Among the vectors, we
/// have:
/// - `feature`: Stores the feature index for splitting at the i'th node.
/// - `threshold`: Stores the threshold value for the i'th node split.
/// - `value`: Stores output value for the i'th node
///
/// # Examples
///
/// ```
/// let mut treee = DecisionTree::new();
/// ```
#[derive(Debug)]
pub struct DecisionTree {
    pub feature: Vec<usize>,
    pub threshold: Vec<f64>,
    pub value: Vec<f64>,
}

// TODO: Implement
enum TreeError {
    NotLeaf(usize),
}

impl DecisionTree {
    pub fn new() -> Self {
        DecisionTree {
            feature: Vec::new(),
            threshold: Vec::new(),
            value: Vec::new(),
        }
    }

    pub fn add_node(&mut self, feature: usize, threshold: f64, value: f64) -> usize {
        let node_id = self.feature.len();
        self.feature.push(feature);
        self.threshold.push(threshold);
        self.value.push(value);
        node_id
    }

    pub fn left_child(&self, index: usize) -> Option<usize> {
        let left_index = index * 2 + 1;
        if left_index < self.feature.len() {
            Some(left_index)
        } else {
            None
        }
    }

    pub fn right_child(&self, index: usize) -> Option<usize> {
        let right_index = index * 2 + 2;
        if right_index < self.feature.len() {
            Some(right_index)
        } else {
            None
        }
    }

    pub fn is_leaf(&self, index: usize) -> bool {
        self.left_child(index).is_none() && self.right_child(index).is_none()
    }

    // Leaf nodes do not have a threshold value
    pub fn get_leaf_nodes(&self) -> Vec<usize> {
        let mut leaf_nodes = Vec::new();
        for (index, threshold) in self.threshold.iter().enumerate() {
            if *threshold == 0.0 {
                leaf_nodes.push(index);
            }
        }
        leaf_nodes
    }

    pub fn node_depth(&self, index: usize) -> usize {
        if index == 0 {
            0
        } else {
            let parent_index = (index - 1) / 2;
            1 + self.node_depth(parent_index)
        }
    }

    pub fn split_node(
        &mut self,
        node_index: usize,
        feature: usize,
        threshold: f64,
        left_value: f64,
        right_value: f64,
    ) -> (usize, usize) {
        if self.is_leaf(node_index) {
            // TODO: We should be using the `add_node` method here
            let left_child_index = node_index * 2 + 1;
            let right_child_index = node_index * 2 + 2;

            // Update the current node
            self.feature[node_index] = feature;
            self.threshold[node_index] = threshold;

            // Add left child
            self.feature.insert(left_child_index, 0); // Placeholder feature
            self.threshold.insert(left_child_index, 0.0); // Placeholder threshold
            self.value.insert(left_child_index, left_value);

            // Add right child
            self.feature.insert(right_child_index, 0); // Placeholder feature
            self.threshold.insert(right_child_index, 0.0); // Placeholder threshold
            self.value.insert(right_child_index, right_value);

            (left_child_index, right_child_index)
        } else {
            // TODO: Error enum
            panic!("Cannot split a non-leaf node");
        }
    }

    pub fn predict(&self, sample: &[f64]) -> f64 {
        let mut node = 0;
        loop {
            if self.is_leaf(node) {
                return self.value[node];
            }
            let feature = self.feature[node];
            let threshold = self.threshold[node];
            node = match sample[feature].partial_cmp(&threshold).unwrap() {
                Ordering::Less => self.left_child(node).unwrap(),
                _ => self.right_child(node).unwrap(),
            };
        }
    }
}
