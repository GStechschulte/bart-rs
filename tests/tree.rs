use bart_rs::tree::DecisionTree;

#[test]
fn test_tree_primitives() {
    let mut tree = DecisionTree::new();
    let root = tree.add_node(0, 1500.0, 200000.0);
    let (left_idx, right_idx) = tree.split_node(0, 0, 1500.0, 1000.0, 2000.0);

    // Test threshold value a node was split on
    assert_eq!(tree.threshold[root], 1500.0);

    // Test index of left and right children of split node
    assert_eq!(tree.left_child(0), Some(1));
    assert_eq!(tree.right_child(0), Some(2));

    // Test is leaf value
    assert_eq!(tree.is_leaf(0), false);
    assert_eq!(tree.is_leaf(1), true);
    assert_eq!(tree.is_leaf(2), true);

    // Test leaf value
    assert_eq!(tree.value.last(), Some(&2000.0));
}
