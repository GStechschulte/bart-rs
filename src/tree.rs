use std::collections::BTreeMap;

// #[derive(Clone)]
pub struct Leaf {
    index: usize,
    value: f64,
}

// #[derive(Clone)]
pub struct Internal {
    index: usize,
    split_idx: usize,
    split_value: f64,
}

// A node may be either: a Leaf or Internal node
// #[derive(Clone)]
pub enum Node {
    Leaf(Leaf),
    Internal(Internal),
}

// #[derive(Clone)]
// #[derive(Debug)]
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

    fn as_internal(&self) -> Result<&Internal, TreeError> {
        match self {
            Node::Internal(n) => Ok(n),
            Node::Leaf(n) => Err(TreeError::NotInternal(n.index)),
        }
    }

    fn index(&self) -> usize {
        match self {
            Node::Internal(n) => n.index,
            Node::Leaf(n) => n.index,
        }
    }

    pub fn depth(&self) -> usize {
        ((self.index() + 1) as f64).log2().floor() as usize
    }
}

impl Tree {
    // Create a Tree with a single root node
    pub fn new(root_value: f64) -> Self {
        let root = Node::Leaf(Leaf::new(0, root_value));
        let nodes = BTreeMap::from_iter([(0, root)]);
        Tree { nodes }
    }

    pub fn root(&self) -> &Node{
        self.nodes.get(&0).expect("The tree should always have a root node at index 0.")
    }

    pub fn get_node(&self, idx: &usize) -> Result<&Node, TreeError> {
        self.nodes.get(idx).ok_or(TreeError::IndexNotFound(*idx))
    }

    // pub fn check_leaf

    pub fn add_node(&mut self, node: Node) -> &Self {
        let idx = node.index();
        self.nodes.insert(idx, node);

        self
    }

    // pub fn update_leaf_node(&mut self, idx: usize, value: f64) -> Result<(), TreeError> {}

    pub fn split_leaf_node(
        &mut self,
        idx: usize,
        split_idx: usize,
        split_value: f64,
        left_value: f64,
        right_value: f64
    ) -> Result<(usize, usize), TreeError> {
        // self.check_leaf(idx)?;

        // Create the new parent
        let new = Internal::new(idx, split_idx, split_value);
        let (left_idx, right_idx) = (new.left(), new.right());

        // Set the new Internal and Leaf nodes
        self.add_node(Node::Internal(new));
        self.add_node(Node::leaf(left_idx, left_value));
        self.add_node(Node::leaf(right_idx, right_value));

        Ok((left_idx, right_idx))
    }
}
