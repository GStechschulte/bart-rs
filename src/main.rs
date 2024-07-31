use bart_rs::data::Matrix;
use ndarray::{array, Array2};
use numpy::PyArray;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use bart_rs::tree::DecisionTree;

#[derive(Debug)]
struct Indices {
    leaf_nodes: HashSet<usize>,
    expansion_nodes: VecDeque<usize>,
    data_indices: HashMap<usize, Vec<usize>>,
}

fn main() {
    // let sampler = ParticleGibbsSampler::new(10, 5);
    // println!("{:?}", sampler);

    let a2 = array![[1, 2], [3, 4]];

    let matrix = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    println!("{:?}", matrix);

    // Create a regression decision tree
    let mut tree = DecisionTree::new();

    let mut tree = DecisionTree::new();
    let root = tree.add_node(0, 1500.0, 200000.0);

    let mut rng = rand::thread_rng();
    println!("{:?}", rng);

    let n_points = 10 as usize;
    let n: Vec<usize> = Vec::from_iter(0..n_points);
    let indices = Indices {
        leaf_nodes: HashSet::from([0]),
        expansion_nodes: VecDeque::from([0]),
        data_indices: HashMap::from([(0, n)]),
    };

    println!("{:?}", indices);
}
