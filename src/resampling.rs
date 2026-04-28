//! Resampling strategies for particle filtering in BART.
use rand::Rng;

/// Resampling interface for implementing different resampling methods.
pub trait ResamplingStrategy {
    /// Resample into a pre-allocated buffer, avoiding allocation.
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>);

    /// Convenience wrapper that allocates a new Vec.
    fn resample(&self, rng: &mut impl Rng, weights: &[f64]) -> Vec<usize> {
        let mut out = Vec::with_capacity(weights.len());
        self.resample_into(rng, weights, &mut out);
        out
    }
}

/// Systematic resampling strategy.
///
/// Builds a CDF and samples a single random offset to determine which bin
/// each particle lands in. Provides lower variance than multinomial resampling.
#[derive(Clone, Copy, Debug)]
pub struct SystematicResampling;

impl ResamplingStrategy for SystematicResampling {
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>) {
        let n = weights.len();
        let u = rng.random::<f64>();

        let mut current_idx = 0usize;
        let mut current_cum = 0.0f64;
        out.clear();

        for i in 0..n {
            let target = (i as f64 + u) / n as f64;

            while current_cum < target && current_idx < n {
                current_cum += weights[current_idx];
                current_idx += 1;
            }

            let ancestor = if current_idx == 0 { 0 } else { current_idx - 1 };
            out.push(ancestor);
        }
    }
}

/// Multinomial resampling strategy.
///
/// Standard multinomial resampling where each particle is independently
/// sampled according to its weight.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialResampling;

impl ResamplingStrategy for MultinomialResampling {
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>) {
        let n = weights.len();
        out.clear();

        // Build CDF (reuses allocation if called repeatedly via resample() wrapper,
        // but this internal vec is unavoidable without changing the trait further)
        let mut cdf = Vec::with_capacity(n);
        let mut cumsum = 0.0;
        for &w in weights {
            cumsum += w;
            cdf.push(cumsum);
        }

        for _ in 0..n {
            let u: f64 = rng.random();
            let idx = cdf.partition_point(|&c| c < u).min(n - 1);
            out.push(idx);
        }
    }
}

/// Stratified resampling strategy.
///
/// Uses independent random numbers for each stratum, providing a middle
/// ground between systematic and multinomial.
#[derive(Clone, Copy, Debug)]
pub struct StratifiedResampling;

impl ResamplingStrategy for StratifiedResampling {
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>) {
        let n = weights.len();
        let mut current_idx = 0usize;
        let mut current_cum = 0.0f64;
        out.clear();

        for i in 0..n {
            let u: f64 = rng.random();
            let target = (i as f64 + u) / n as f64;

            while current_cum < target && current_idx < n {
                current_cum += weights[current_idx];
                current_idx += 1;
            }

            let ancestor = if current_idx == 0 { 0 } else { current_idx - 1 };
            out.push(ancestor);
        }
    }
}

/// Enum for dynamic dispatch over resampling strategies.
#[derive(Clone, Copy, Debug)]
pub enum ResamplingStrategies {
    Systematic(SystematicResampling),
    Multinomial(MultinomialResampling),
    Stratified(StratifiedResampling),
}

impl ResamplingStrategies {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.to_lowercase().as_str() {
            "systematic" => Ok(ResamplingStrategies::Systematic(SystematicResampling)),
            "multinomial" => Ok(ResamplingStrategies::Multinomial(MultinomialResampling)),
            "stratified" => Ok(ResamplingStrategies::Stratified(StratifiedResampling)),
            _ => Err(format!("Unknown resampling strategy: '{}'", name)),
        }
    }
}

impl ResamplingStrategy for ResamplingStrategies {
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>) {
        match self {
            ResamplingStrategies::Systematic(s) => s.resample_into(rng, weights, out),
            ResamplingStrategies::Multinomial(s) => s.resample_into(rng, weights, out),
            ResamplingStrategies::Stratified(s) => s.resample_into(rng, weights, out),
        }
    }
}
