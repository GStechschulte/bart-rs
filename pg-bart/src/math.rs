pub fn normalized_cumsum(v: &[f64]) -> Vec<f64> {
    let total: f64 = v.iter().sum();
    let ret: Vec<f64> = v
        .iter()
        .scan(0f64, |state, item| {
            *state += *item;
            let ret = *state / total;
            Some(ret)
        })
        .collect();

    ret
}
