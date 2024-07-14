use std::collections::BTreeMap;

#[derive(Clone)]
pub struct Leaf {
    index: usize,
    value: f64,
}

#[derive(Clone)]
pub struct Internal {
    index: usize,
    split_idx: usize,
    split_value: f64,
}

// A node may be either: a Leaf or Internal node
#[derive(Clone)]
pub enum Node {
    Leaf(Leaf),
    Internal(Internal),
}

#[derive(Clone)]
pub struct Tree {
    nodes: BTreeMap<usize, Node>,
}

#[derive(Debug)]
pub enum TreeError {
    NotLeaf(usize),
    NotInternal(usize),
    IndexNotFound(usize),
}

impl Leaf {
    // Create a new leaf node
    pub fn new(index: usize, value: f64) -> Self {
        Leaf { index, value }
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

impl Internal {
    // Creates a new internal node
    pub fn new(index: usize, split_idx: usize, value: f64) -> Self {
        Internal {
            index,
            split_idx,
            split_value: value,
        }
    }

    // Get index of left child for this node
    fn left(&self) -> usize {
        self.index * 2 + 1
    }

    // Get index of right child for this node
    fn right(&self) -> usize {
        self.index * 2 + 2
    }
}

impl Node {
    // Create a new internal node
    pub fn internal(index: usize, split_idx: usize, split_value: f64) -> Self {
        Node::Internal(Internal::new(index, split_idx, split_value))
    }

    // Create a new leaf node
    pub fn leaf(index: usize, value: f64) -> Self {
        Node::Leaf(Leaf::new(index, value))
    }
}
