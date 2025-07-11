//! Resampling strategies for particle filtering in BART.
//!
//! This module provides different resampling methods for selecting particles
//! based on their normalized weights.

use rand::rngs::SmallRng;
use rand::Rng;
use std::cmp::Ordering;

/// Resampling interface for implementing different resampling methods
pub trait ResamplingStrategy {
    /// Resample particle indices based on their normalized weights
    fn resample(&self, weights: &[f64], n_samples: usize, rng: &mut SmallRng) -> Vec<usize>;
}

/// Systematic resampling strategy
///
/// Provides lower variance than multinomial resampling by using a deterministic
/// grid with a single random offset. This is the preferred method for most applications.
pub struct SystematicResampling;

impl ResamplingStrategy for SystematicResampling {
    fn resample(&self, weights: &[f64], n_samples: usize, rng: &mut SmallRng) -> Vec<usize> {
        if weights.is_empty() {
            return Vec::new();
        }

        let mut indices = Vec::with_capacity(n_samples);
        let mut cumulative_weights = Vec::with_capacity(weights.len());

        // Compute cumulative weights
        let mut cum_sum = 0.0;
        for &weight in weights {
            cum_sum += weight;
            cumulative_weights.push(cum_sum);
        }

        // Generate systematic samples
        let step = 1.0 / n_samples as f64;
        let start = rng.random::<f64>() * step;

        for i in 0..n_samples {
            let target = start + i as f64 * step;

            // Find the particle corresponding to this target
            let idx = cumulative_weights
                .binary_search_by(|&x| {
                    if x < target {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                })
                .unwrap_or_else(|x| x);

            indices.push(idx.min(weights.len() - 1));
        }

        indices
    }
}

/// Multinomial resampling strategy
///
/// Standard multinomial resampling where each particle is independently
/// sampled according to its weight. Has higher variance than systematic resampling.
pub struct MultinomialResampling;

impl ResamplingStrategy for MultinomialResampling {
    fn resample(&self, weights: &[f64], n_samples: usize, rng: &mut SmallRng) -> Vec<usize> {
        if weights.is_empty() {
            return Vec::new();
        }

        let mut indices = Vec::with_capacity(n_samples);
        let mut cumulative_weights = Vec::with_capacity(weights.len());

        // Compute cumulative weights
        let mut cum_sum = 0.0;
        for &weight in weights {
            cum_sum += weight;
            cumulative_weights.push(cum_sum);
        }

        // Generate multinomial samples
        for _ in 0..n_samples {
            let target = rng.random::<f64>();

            // Find the particle corresponding to this target
            let idx = cumulative_weights
                .binary_search_by(|&x| {
                    if x < target {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                })
                .unwrap_or_else(|x| x);

            indices.push(idx.min(weights.len() - 1));
        }

        indices
    }
}

/// Stratified resampling strategy
///
/// Similar to systematic resampling but uses independent random numbers
/// for each stratum, providing a middle ground between systematic and multinomial.
pub struct StratifiedResampling;

impl ResamplingStrategy for StratifiedResampling {
    fn resample(&self, weights: &[f64], n_samples: usize, rng: &mut SmallRng) -> Vec<usize> {
        if weights.is_empty() {
            return Vec::new();
        }

        let mut indices = Vec::with_capacity(n_samples);
        let mut cumulative_weights = Vec::with_capacity(weights.len());

        // Compute cumulative weights
        let mut cum_sum = 0.0;
        for &weight in weights {
            cum_sum += weight;
            cumulative_weights.push(cum_sum);
        }

        // Generate stratified samples
        let step = 1.0 / n_samples as f64;

        for i in 0..n_samples {
            let target = (i as f64 + rng.random::<f64>()) * step;

            // Find the particle corresponding to this target
            let idx = cumulative_weights
                .binary_search_by(|&x| {
                    if x < target {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                })
                .unwrap_or_else(|x| x);

            indices.push(idx.min(weights.len() - 1));
        }

        indices
    }
}

/// Enum for dynamic dispatch over resampling strategies
pub enum ResamplingStrategies {
    Systematic(SystematicResampling),
    Multinomial(MultinomialResampling),
    Stratified(StratifiedResampling),
}

impl ResamplingStrategies {
    pub fn resample(&self, weights: &[f64], n_samples: usize, rng: &mut SmallRng) -> Vec<usize> {
        match self {
            ResamplingStrategies::Systematic(strategy) => {
                strategy.resample(weights, n_samples, rng)
            }
            ResamplingStrategies::Multinomial(strategy) => {
                strategy.resample(weights, n_samples, rng)
            }
            ResamplingStrategies::Stratified(strategy) => {
                strategy.resample(weights, n_samples, rng)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_systematic_resampling() {
        let mut rng = SmallRng::seed_from_u64(42);
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let resampler = SystematicResampling;

        let indices = resampler.resample(&weights, 10, &mut rng);
        assert_eq!(indices.len(), 10);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < weights.len());
        }
    }

    #[test]
    fn test_multinomial_resampling() {
        let mut rng = SmallRng::seed_from_u64(42);
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let resampler = MultinomialResampling;

        let indices = resampler.resample(&weights, 10, &mut rng);
        assert_eq!(indices.len(), 10);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < weights.len());
        }
    }

    #[test]
    fn test_stratified_resampling() {
        let mut rng = SmallRng::seed_from_u64(42);
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let resampler = StratifiedResampling;

        let indices = resampler.resample(&weights, 10, &mut rng);
        assert_eq!(indices.len(), 10);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < weights.len());
        }
    }

    #[test]
    fn test_empty_weights() {
        let mut rng = SmallRng::seed_from_u64(42);
        let weights: Vec<f64> = vec![];
        let resampler = SystematicResampling;

        let indices = resampler.resample(&weights, 0, &mut rng);
        assert_eq!(indices.len(), 0);
    }
}
