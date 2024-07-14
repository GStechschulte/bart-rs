use std::collections::BTreeMap;

use bart_rs::tree;

fn main() {
    let mut voc = BTreeMap::new();

    voc.insert(1, "Amsterdam");
    voc.insert(2, "Middelburg");
    voc.insert(3, "Delft");

    for (idx, kamer) in &voc {
        println!("{} invested {}", idx, kamer)
    }

    println!("{:?}", voc.get(&1));
}
