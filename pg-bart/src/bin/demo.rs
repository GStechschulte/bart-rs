use ndarray::{array, Array2};
// use numpy::PyArray;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use pg_bart::tree::DecisionTree;

#[derive(Debug)]
struct Indices {
    leaf_nodes: HashSet<usize>,
    expansion_nodes: VecDeque<usize>,
    data_indices: HashMap<usize, Vec<usize>>,
}

fn main() {
    println!("Hello BART")
}
