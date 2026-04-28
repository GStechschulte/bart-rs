use crate::tree::TreeArrays;

/// Simple ensemble of trees. No longer conflated with SMC particle management.
#[derive(Clone, Debug)]
pub struct Forest {
    pub trees: Vec<TreeArrays>,
}

impl Forest {
    pub fn new() -> Self {
        Self { trees: Vec::new() }
    }

    pub fn with_capacity(n_trees: usize) -> Self {
        Self {
            trees: Vec::with_capacity(n_trees),
        }
    }

    pub fn push(&mut self, tree: TreeArrays) {
        self.trees.push(tree);
    }

    pub fn len(&self) -> usize {
        self.trees.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }
}

impl Default for Forest {
    fn default() -> Self {
        Self::new()
    }
}
