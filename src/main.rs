use std::collections::BTreeMap;
use std::cmp::Ordering;

use bart_rs::tree::DecisionTree;


fn main() {
    // Create a regression decision tree
    let mut tree = DecisionTree::new();

    // Build the tree (house size in sq ft, number of bedrooms)
    let root = tree.add_node(0, 1500.0, 200000.0); // Split on house size

    println!("Initial tree  : {:?}", tree);
    println!("Is leaf node? : {}", tree.is_leaf(root));

    println!("Feature length: {:?}", tree.feature.len());

    tree.split_node(root, 0, 1500.0, 150000.0, 250000.0);

    println!("Tree after split: {:?}", tree);

    println!("Split value of root node: {:?}", tree.threshold[root]);

    println!("Leaf values: {:?}", tree.value);

    let idx = tree.threshold.into_iter().position(|x| x == 0.0);
    println!("{:?}", idx);

    // Set up the tree structure
    // tree.set_child(root, true, left);
    // tree.set_child(root, false, right);
    // tree.set_child(left, true, left_left);
    // tree.set_child(left, false, left_right);
    // tree.set_child(right, true, right_left);
    // tree.set_child(right, false, right_right);

    // println!("{:?}", tree);

    // Test cases
    // let test_cases = vec![
    //     (vec![1200.0, 2.0], "small house, 2 bedrooms"),
    //     (vec![1800.0, 2.0], "large house, 2 bedrooms"),
    //     (vec![1400.0, 3.0], "small house, 3 bedrooms"),
    //     (vec![2000.0, 4.0], "large house, 4 bedrooms"),
    // ];

    // Make predictions
    // for (sample, description) in test_cases {
    //     let prediction = tree.predict(&sample);
    //     println!("Prediction for {} (size: {} sq ft, {} bedrooms): ${:.2}",
    //              description, sample[0], sample[1], prediction[0]);
    // }
}
