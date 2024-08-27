/// Tests for DecisionTree primitives
use pg_bart::tree::{DecisionTree, TreeError};

#[test]
fn test_new_tree() {
    let tree = DecisionTree::new();
    assert!(tree.feature.is_empty());
    assert!(tree.threshold.is_empty());
    assert!(tree.value.is_empty());
}

#[test]
fn test_add_root_node() {
    let mut tree = DecisionTree::new();
    let root_index = tree.add_node(usize::MAX, 0.0, 5.0);
    assert_eq!(root_index, 0);
    assert_eq!(tree.feature[0], usize::MAX);
    assert_eq!(tree.threshold[0], 0.0);
    assert_eq!(tree.value[0], 5.0);
}

#[test]
fn test_split_node() {
    let mut tree = DecisionTree::new();
    let root_index = tree.add_node(usize::MAX, 0.0, 5.0);

    let result = tree.split_node(root_index, 0, 0.5, 2.0, 3.0);
    assert!(result.is_ok());

    let (left_index, right_index) = result.unwrap();
    assert_eq!(left_index, 1);
    assert_eq!(right_index, 2);

    assert_eq!(tree.feature[root_index], 0);
    assert_eq!(tree.threshold[root_index], 0.5);

    assert_eq!(tree.feature[left_index], usize::MAX);
    assert_eq!(tree.value[left_index], 2.0);

    assert_eq!(tree.feature[right_index], usize::MAX);
    assert_eq!(tree.value[right_index], 3.0);
}

#[test]
fn test_split_non_leaf_node() {
    let mut tree = DecisionTree::new();
    let root_index = tree.add_node(usize::MAX, 0.0, 5.0);
    tree.split_node(root_index, 0, 0.5, 2.0, 3.0).unwrap();

    let result = tree.split_node(root_index, 1, 0.7, 4.0, 5.0);
    assert!(result.is_err());
    // assert_eq!(result.unwrap_err(), TreeError::NonLeafSplit);
}

#[test]
fn test_split_invalid_node() {
    let mut tree = DecisionTree::new();
    let root_index = tree.add_node(usize::MAX, 0.0, 5.0);

    let result = tree.split_node(root_index + 1, 0, 0.5, 2.0, 3.0);
    assert!(result.is_err());
    // assert_eq!(result.unwrap_err(), TreeError::InvalidNodeIndex);
}

#[test]
fn test_is_leaf() {
    let mut tree = DecisionTree::new();
    let root_index = tree.add_node(usize::MAX, 0.0, 5.0);
    assert!(tree.is_leaf(root_index));

    tree.split_node(root_index, 0, 0.5, 2.0, 3.0).unwrap();
    assert!(!tree.is_leaf(root_index));
    assert!(tree.is_leaf(1));
    assert!(tree.is_leaf(2));
}
