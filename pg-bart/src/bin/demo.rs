fn main() {
    let preds = vec![0.5, 0.25, 0.75];

    let mean = vec![0.1; 3];

    let res: Vec<f64> = preds
        .iter()
        .zip(mean.iter())
        .map(|(&a, &b)| a - b)
        .collect();

    println!("mean: {:?}", res);
}
