use std::cmp::Ordering;

/// A `DecisionTree` structure is implemented as a number of parallel
/// vectors. The i-th element of each vector holds information about
/// node `i`. Node 0 is the tree's root. Some of the arrays only apply
/// to either leaves or split nodes. In this case, the values of the
/// nodes of the other type is arbitrary. For example, `feature` and
/// `threshold` vectors only apply to split nodes. The values for leaf
/// nodes in these arrays are therefore arbitrary. Among the arrays,
/// we have:
/// - `feature`: Stores the feature index for splitting at the i'th node.
/// - `threshold`: Stores the threshold value for the i'th node split.
/// - `children_left`: Store indices of the left child for the i'th node.
/// - `children_right`: Stores indices of the right child for the i'th node
/// - `value`: Stores output value for the i'th node
///
/// # Examples
///
/// ```
/// let mut treee = DecisionTree::new();
/// ```
#[derive(Debug)]
pub struct DecisionTree {
    feature: Vec<usize>,
    threshold: Vec<f64>,
    children_left: Vec<i32>,
    children_right: Vec<i32>,
    value: Vec<Vec<f64>>,
}

impl DecisionTree {
    pub fn new() -> Self {
        DecisionTree {
            feature: Vec::new(),
            threshold: Vec::new(),
            children_left: Vec::new(),
            children_right: Vec::new(),
            value: Vec::new(),
        }
    }

    pub fn add_node(&mut self, feature: usize, threshold: f64, value: Vec<f64>) -> usize {
        let node_id = self.feature.len();
        self.feature.push(feature);
        self.threshold.push(threshold);
        self.children_left.push(-1);
        self.children_right.push(-1);
        self.value.push(value);
        node_id
    }

    pub fn set_child(&mut self, parent: usize, is_left: bool, child: usize) {
        if is_left {
            self.children_left[parent] = child as i32;
        } else {
            self.children_right[parent] = child as i32;
        }
    }

    pub fn predict(&self, sample: &[f64]) -> &[f64] {
        let mut node = 0;
        loop {
            if self.children_left[node] == -1 && self.children_right[node] == -1 {
                return &self.value[node]
            }
            let feature = self.feature[node];
            let threshold = self.threshold[node];
            node = match sample[feature].partial_cmp(&threshold).unwrap() {
                Ordering::Less => self.children_left[node] as usize,
                _ => self.children_right[node] as usize,
            };
        }
    }
}
