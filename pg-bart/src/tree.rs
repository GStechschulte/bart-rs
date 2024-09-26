use core::fmt;
use std::cmp::Ordering;

#[derive(Debug)]
pub struct DecisionTree {
    pub feature: Vec<usize>,
    pub threshold: Vec<f64>,
    pub value: Vec<f64>,
}

#[derive(Debug)]
pub enum TreeError {
    NonLeafSplit,
    InvalidNodeIndex,
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TreeError::NonLeafSplit => write!(f, "Cannot split a non-leaf node"),
            TreeError::InvalidNodeIndex => write!(f, "Node index does not exist"),
        }
    }
}

impl DecisionTree {
    /// Creates a new DecisionTree with an initial value set as the root node.
    /// A decision tree is implemented as three parallel vectors.
    ///
    /// Using parallel vectors allows for a more cache-efficient implementation.
    /// Moreover, index accessing allows one to more easily avoid borrow checker
    /// issues related to classical recursive binary tree implementations.
    ///
    /// The i-th element of each vector holds information about node `i`. Node 0
    /// is the tree's root. Some of the arrays only apply to either leaves or
    /// split nodes. In this case, the values of the nodes of other vectors is
    /// arbitrary. For example, `feature` and `threshold` vectors only apply to
    /// split nodes. The values for leaf nodes in these arrays are therefore
    /// arbitrary. The threee vectors are:
    /// - `feature`. Stores the feature index for splitting at the i'th node.
    /// - `threshold`. Stores the threshold value for the i'th node split.
    /// - `value`. Stores output value for the i'th node
    ///
    /// # Examples
    ///
    /// ```
    /// let mut tree = DecisionTree::new(0.5);
    /// ```
    pub fn new(init_value: f64) -> Self {
        Self {
            feature: vec![0],     // Initialize with a placeholder feature
            threshold: vec![0.0], // Initialize with a placeholder threshold
            value: vec![init_value],
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

    /// Check whether the passed index is a leaf.
    ///
    /// Assumes that leaf nodes have a feature index of 0 and a threshold of 0.0.
    /// This is consistent with the initializing of new leaf nodes in the add_node method
    pub fn is_leaf(&self, index: usize) -> bool {
        index >= self.feature.len() || (self.feature[index] == 0 && self.threshold[index] == 0.0)
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
    ) -> Result<(usize, usize), TreeError> {
        if node_index >= self.value.len() {
            return Err(TreeError::InvalidNodeIndex);
        }

        if !self.is_leaf(node_index) {
            return Err(TreeError::NonLeafSplit);
        }

        // Update the current node
        self.feature[node_index] = feature;
        self.threshold[node_index] = threshold;

        // Add new left and right leaf nodes
        let left_child_index = self.add_node(0, 0.0, left_value);
        let right_child_index = self.add_node(0, 0.0, right_value);

        Ok((left_child_index, right_child_index))
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
