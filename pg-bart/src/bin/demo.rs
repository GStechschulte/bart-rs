use pg_bart::split_rules::{ContinuousSplit, OneHotSplit, SplitRule};

fn main() {
    let feature_values: Vec<i32> = vec![1, 1, 1, 1];
    let rule = OneHotSplit;

    let split_value = rule.sample_split_value(&feature_values);
    assert_eq!(split_value, None);

    println!("{:?}", split_value);

    let static_split_value = 1 as i32;
    let (left, right) = rule.divide(&feature_values, &static_split_value);

    println!("{:?}, {:?}", left, right);
}
