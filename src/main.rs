use std::collections::BTreeMap;
// use rand::prelude::*;
//
use bart_rs::tree::{Node, Tree};


// #[derive(Debug, Clone)]
// struct DecisionTree {
//     leaf: Vec<f64>,     // Contains the values in the leaves
//     var: Vec<usize>,    // Contains the axes along which the decision nodes operate
//     split: Vec<f64>     // Contains the decision boundaries
// }

// impl DecisionTree {
//     fn new(max_depth: usize, num_features: usize) -> Self {
//         let num_nodes = 2usize.pow(max_depth as u32) - 1;
//         let num_leaves = 2usize.pow(max_depth as u32);

//         DecisionTree {
//             leaf: vec![0.0; num_leaves],
//             var: vec![0; num_nodes / 2],
//             split: vec![0.0; num_nodes / 2]
//         }
//     }

//     fn left_child(i: usize) -> usize {2 * i}

//     fn right_child(i: usize) -> usize {2 * i + 1}

//     fn is_leaf(&self, i: usize) -> bool {self.split[i] == 0.0}
// }

// #[derive(Debug)]
// struct ParticleGibbsSampler {
//     trees: Vec<DecisionTree>,
//     weights: Vec<f64>
// }

// impl ParticleGibbsSampler {
//     fn new(num_particles: usize, max_depth: usize, num_features: usize) -> Self {
//         // let mut rng = thread_rng();
//         let trees = (0..num_particles)
//             .map(|_| DecisionTree::new(max_depth, num_features))
//             .collect();
//         let weights = vec![1.0 / num_particles as f64; num_particles];

//         ParticleGibbsSampler { trees, weights }
//     }

//     // Systematic resample of all but first Particle
//     fn resample(&mut self) {
//         let mut rng = thread_rng();
//         let mut new_trees: Vec<DecisionTree> = Vec::with_capacity(self.trees.len());
//     }
// }

fn main() {

    let tree = Tree::new(10.);
    // println!("{:#?}", tree);

    let mut voc: BTreeMap<usize, f64> = BTreeMap::new();
    voc.insert(1, 10.);
    voc.insert(2, 50.);
    voc.insert(3, 25.);

    println!("{:#?}", voc);
    voc.split_off(&2);
    println!("{:#?}", voc);

    // let dt = DecisionTree::new(1, 5);
    // println!("{:?}", dt);

    // let pg = ParticleGibbsSampler::new(10, 1, 5);
    // println!("{:?}", pg);
}
