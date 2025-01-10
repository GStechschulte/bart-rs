//! Utility functions for computing averages, standard deviations,
//! and normalized cumulative sums.

/// Computes the running standard deviation using Welford's algorithm.
pub struct RunningStd {
    count: usize,
    mean: Vec<f64>,
    mean_2: Vec<f64>,
}

impl RunningStd {
    /// Create a new `RunningSd` with a specified shape (size of the vector).
    pub fn new(shape: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; shape],
            mean_2: vec![0.0; shape],
        }
    }

    /// Update the running statistics with a new value
    pub fn update(&mut self, new_value: &[f64]) -> Vec<f64> {
        self.count += 1;
        let (mean, mean_2, std) = update_stats(self.count, &self.mean, &self.mean_2, new_value);
        self.mean = mean;
        self.mean_2 = mean_2;
        compute_mean(&std)
    }
}

/// Update function to calculate the new `mean` and `mean_2` values
fn update_stats(
    count: usize,
    mean: &[f64],
    mean_2: &[f64],
    new_value: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut new_mean = vec![0.0; mean.len()];
    let mut new_mean_2 = vec![0.0; mean_2.len()];

    for (i, ((&m, &m2), &nv)) in mean
        .iter()
        .zip(mean_2.iter())
        .zip(new_value.iter())
        .enumerate()
    {
        let delta = nv - m;
        let updated_mean = m + delta / count as f64;
        let delta2 = nv - updated_mean;

        new_mean[i] = updated_mean;
        new_mean_2[i] = m2 + delta * delta2;
    }

    let std: Vec<f64> = new_mean_2
        .iter()
        .map(|&m| (m / count as f64).sqrt())
        .collect();

    (new_mean, new_mean_2, std)
}

/// Calculate the mean of the array
fn compute_mean(ari: &[f64]) -> Vec<f64> {
    let sum: f64 = ari.iter().sum();
    vec![sum / ari.len() as f64]
}

/// Computes the normalized cumulative sum.
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
