use std::collections::BTreeMap;
use std::cmp::Ordering;

use bart_rs::tree::DecisionTree;


fn main() {
    // Create a regression decision tree
    let mut tree = DecisionTree::new();

    // Build the tree (house size in sq ft, number of bedrooms)
    let root = tree.add_node(0, 1500.0, vec![200000.0]); // Split on house size

    // Left subtree (smaller houses)
    let left = tree.add_node(1, 2.0, vec![150000.0]); // Split on number of bedrooms
    let left_left = tree.add_node(0, 0.0, vec![120000.0]); // Leaf node
    let left_right = tree.add_node(0, 0.0, vec![180000.0]); // Leaf node

    // Right subtree (larger houses)
    let right = tree.add_node(1, 3.0, vec![300000.0]); // Split on number of bedrooms
    let right_left = tree.add_node(0, 0.0, vec![250000.0]); // Leaf node
    let right_right = tree.add_node(0, 0.0, vec![350000.0]); // Leaf node

    // Set up the tree structure
    tree.set_child(root, true, left);
    tree.set_child(root, false, right);
    tree.set_child(left, true, left_left);
    tree.set_child(left, false, left_right);
    tree.set_child(right, true, right_left);
    tree.set_child(right, false, right_right);

    println!("{:?}", tree);

    // Test cases
    let test_cases = vec![
        (vec![1200.0, 2.0], "small house, 2 bedrooms"),
        (vec![1800.0, 2.0], "large house, 2 bedrooms"),
        (vec![1400.0, 3.0], "small house, 3 bedrooms"),
        (vec![2000.0, 4.0], "large house, 4 bedrooms"),
    ];

    // Make predictions
    for (sample, description) in test_cases {
        let prediction = tree.predict(&sample);
        println!("Prediction for {} (size: {} sq ft, {} bedrooms): ${:.2}",
                 description, sample[0], sample[1], prediction[0]);
    }

    let mut voc: BTreeMap<usize, f64> = BTreeMap::new();
    voc.insert(1, 10.);
    voc.insert(2, 50.);
    voc.insert(3, 25.);

    // println!("{:#?}", voc);
    // println!("{:#?}", voc);

    // let dt = DecisionTree::new(1, 5);
    // println!("{:?}", dt);

    // let pg = ParticleGibbsSampler::new(10, 1, 5);
    // println!("{:?}", pg);
}
