use pymc_bart::split_rules::{ContinuousSplit, OneHotSplit, SplitRule};

#[test]
fn test_continuous_split_rule() {
    let rule = ContinuousSplit;

    let feature_values: Vec<f64> = (0..10).map(f64::from).collect();
    let split_value = rule.sample_split_value(&feature_values);

    assert!(split_value.is_some());
    assert!(feature_values.contains(&split_value.unwrap()));

    let static_split_value = 4.0;
    let (left, right) = rule.divide(&feature_values, &static_split_value);

    // Indices where value <= 2 go left
    assert_eq!(left, vec![0, 1, 2, 3, 4]);
    // Indices where value > 2 go right
    assert_eq!(right, vec![5, 6, 7, 8, 9]);
}

#[test]
fn test_one_hot_split_rule() {
    let rule = OneHotSplit;

    // Test heterogeneous vector of integers
    let feature_values: Vec<i32> = vec![1, 2, 3, 2];
    let split_value = rule.sample_split_value(&feature_values);

    assert!(split_value.is_some());
    assert!(feature_values.contains(&split_value.unwrap()));

    let static_split_value = 2 as i32;
    let (left, right) = rule.divide(&feature_values, &static_split_value);

    // Indices where value = 2 go left
    assert_eq!(left, vec![1, 3]);
    // Indices where value != 2 go right
    assert_eq!(right, vec![0, 2]);

    // Test homogeneous vector of integers
    let feature_values: Vec<i32> = vec![1, 1, 1, 1];
    let split_value = rule.sample_split_value(&feature_values);

    assert_eq!(split_value, None);

    let static_split_value = 1 as i32;
    let (left, right) = rule.divide(&feature_values, &static_split_value);

    // Indices where value = 1 go left
    assert_eq!(left, vec![0, 1, 2, 3]);
    // Indices where value != 1 go right
    assert_eq!(right, vec![]);
}
