// Tests for DecisionTree primitives
use pg_bart::tree::{DecisionTree, TreeError};

#[test]
fn test_new_tree() {
    let init_val = 0.0;
    let tree = DecisionTree::new(init_val);

    assert_eq!(tree.feature[0], 0 as usize);
    assert_eq!(tree.threshold[0], 0.0);
    assert_eq!(tree.value[0], init_val);
}

#[test]
fn test_add_root_node() {
    let mut tree = DecisionTree::new(0.0);
    let root_index = tree.add_node(0, 0.0, 5.0);

    assert_eq!(root_index, 1 as usize);
    assert_eq!(tree.feature[0], 0 as usize);
    assert_eq!(tree.threshold[0], 0.0);
    assert_eq!(tree.value[0], 0.0);
}

#[test]
fn test_split_node() {
    let mut tree = DecisionTree::new(0.0);
    let root_index = tree.add_node(0, 0.0, 5.0);

    let result = tree.split_node(root_index, 0, 0.5, 2.0, 3.0);

    assert!(result.is_ok());

    let (left_index, right_index) = result.unwrap();

    // Assert indexes are correct
    assert_eq!(left_index, 2 as usize);
    assert_eq!(right_index, 3 as usize);

    // Assert feature and threshold values  are correct for  root/left/right indexes
    assert_eq!(tree.feature[root_index], 0 as usize);
    assert_eq!(tree.threshold[root_index], 0.5);

    assert_eq!(tree.feature[left_index], 0 as usize);
    assert_eq!(tree.value[left_index], 2.0);

    assert_eq!(tree.feature[right_index], 0 as usize);
    assert_eq!(tree.value[right_index], 3.0);
}

#[test]
fn test_split_non_leaf_node() {
    let mut tree = DecisionTree::new(0.0);

    let (left_index, right_index) = tree
        .split_node(0, 0, 0.5, 2.0, 3.0)
        .expect("First split should be successful");

    // Root node has already been split above
    let result = tree.split_node(0, 1, 0.7, 4.0, 5.0);

    assert!(matches!(result, Err(TreeError::NonLeafSplit)));
}

#[test]
fn test_split_invalid_node() {
    let tree = DecisionTree::new(0.0);

    // There are no left or right children yet
    assert_eq!(tree.left_child(1), None);
    assert_eq!(tree.right_child(1), None);
}

#[test]
fn test_is_leaf() {
    let mut tree = DecisionTree::new(0.0);

    // Root should be a leaf initially
    assert!(tree.is_leaf(0));

    // Split root node
    let (left_index, right_index) = tree.split_node(0, 0, 0.5, 2.0, 3.0).unwrap();

    // Root should no longer be a leaf
    assert!(!tree.is_leaf(0));
    // Left and right children should be leafs
    assert!(tree.is_leaf(left_index));
    assert!(tree.is_leaf(right_index));
}
