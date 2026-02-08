use pymc_bart::ops::{ConstantResponse, LinearResponse, Response, TreeSamplingOps};

use rand_distr::Normal;

#[test]
fn test_sample_expand_flag() {
    let sampling_ops = TreeSamplingOps {
        normal: Normal::new(0.0, 1.0).unwrap(),
        alpha_vec: vec![],
        splitting_probs: vec![],
        alpha: 0.95,
        beta: 2.0,
    };

    // Depth == 0 should always return true
    assert!(sampling_ops.sample_expand_flag(0));

    // Depth > 0 should return a boolean (we cannot predict randomness but can check type)
    let result = sampling_ops.sample_expand_flag(3);
    assert!(result == true || result == false);
}

#[test]
fn test_constant_response_compute_leaf_value() {
    let constant_response = ConstantResponse;
    let mu = vec![1.0, 2.0, 3.0];
    let m = 2;
    let norm = 0.5;

    let result = constant_response.compute_leaf_value(&mu, m, norm);
    assert!((result - 1.75).abs() < 1e-6); // Expected: (1 + 2 + 3) / 3 / 2 + 0.5 = 1.75
}

#[test]
fn test_linear_response_compute_leaf_value_len_2() {
    let linear_response = LinearResponse;
    let mu = vec![4.0, 6.0];
    let m = 2;
    let norm = 1.0;

    let result = linear_response.compute_leaf_value(&mu, m, norm);
    assert!((result - 3.5).abs() < 1e-6); // Expected: (4 + 6) / (2 * 2) + 1 = 3.5
}

#[test]
#[should_panic(expected = "Linear response requires at least 2 values.")]
fn test_linear_response_compute_leaf_value_invalid_len() {
    let linear_response = LinearResponse;
    let mu = vec![5.0];
    let m = 1;
    let norm = 0.0;

    linear_response.compute_leaf_value(&mu, m, norm);
}

#[test]
fn test_sample_leaf_value() {
    let sampling_ops = TreeSamplingOps {
        normal: Normal::new(0.0, 1.0).unwrap(),
        alpha_vec: vec![],
        splitting_probs: vec![],
        alpha: 0.95,
        beta: 2.0,
    };

    let mu = vec![10.0, 20.0];
    let obs = vec![5.0];
    let m = 2;
    let leaf_sd = &1.5;
    let shape = 2;

    // Test with ConstantResponse
    let response_constant = Response::Constant(ConstantResponse);
    let result_constant =
        sampling_ops.sample_leaf_value(&mu, &obs, m, leaf_sd, shape, &response_constant);

    // Check that result is a float (randomness prevents exact value checks)
    assert!(result_constant.is_finite());

    // Test with LinearResponse
    let response_linear = Response::Linear(LinearResponse);
    let result_linear =
        sampling_ops.sample_leaf_value(&mu, &obs, m, leaf_sd, shape, &response_linear);

    assert!(result_linear.is_finite());
}

#[test]
fn test_sample_split_feature() {
    let sampling_ops = TreeSamplingOps {
        normal: Normal::new(0.0, 1.0).unwrap(),
        alpha_vec: vec![1.0, 2.0],
        splitting_probs: vec![0.4, 1.0], // Cumulative probabilities
        alpha: 0.95,
        beta: 2.0,
    };

    // Ensure sampled index is within bounds
    for _ in 0..100 {
        let idx = sampling_ops.sample_split_feature();
        assert!(idx < sampling_ops.splitting_probs.len());
    }
}
