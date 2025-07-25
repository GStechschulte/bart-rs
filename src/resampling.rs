//! Resampling strategies for particle filtering in BART.
//!
//! This module provides different resampling methods for selecting particles
//! based on their normalized weights.
use rand::Rng;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Resampling interface for implementing different resampling methods
pub trait ResamplingStrategy {
    /// Resample particle indices based on their normalized weights
    fn resample<R, I>(rng: &mut R, weights: I) -> impl Iterator<Item = usize>
    where
        R: Rng,
        I: Iterator<Item = f64>;
}

/// Systematic resampling strategy
///
/// Builds a cumulative distribution function (CDF) and samples a single random offset
/// `u` to determine which bin (interval) each particle `u_i` lands in. Provides
/// a lower variance than multinomial resampling and is the preferred method for most
/// applications.
#[derive(Clone, Copy, Debug)]
pub struct SystematicResampling;

impl ResamplingStrategy for SystematicResampling {
    fn resample<R, I>(rng: &mut R, weights: I) -> impl Iterator<Item = usize>
    where
        R: Rng,
        I: Iterator<Item = f64>,
    {
        // let n = weights.len();
        // let u = rng.random::<f64>(); // Exclusive (does not sample 0 or 1)
        // let mut cumsum_iter = weights
        //     .iter()
        //     .scan(0.0, |acc, &w| {
        //         *acc += w;
        //         Some(*acc)
        //     })
        //     .enumerate()
        //     .peekable();

        // let linspace = (0..n).map(|i| (i as f64 + u) / n as f64);

        // // Map each target point to a particle index.
        // let indices = linspace
        //     .map(|point| {
        //         while let Some(&(_, cumsum_val)) = cumsum_iter.peek() {
        //             if cumsum_val < point {
        //                 cumsum_iter.next(); // Consume and advance to the next bin.
        //             } else {
        //                 break; // Found the correct bin.
        //             }
        //         }

        //         // The index of the peeked item is the resampled index.
        //         cumsum_iter.peek().map_or(n - 1, |(idx, _)| *idx)
        //     })
        //     .collect();

        // indices

        let weights_vec: Vec<f64> = weights.collect();
        let n = weights_vec.len();
        let u = rng.random::<f64>(); // Random offset

        // Pre-compute cumulative sum
        let mut cumsum = 0.0;
        let cumsum_vec: Vec<f64> = weights_vec
            .iter()
            .map(|&w| {
                cumsum += w;
                cumsum
            })
            .collect();

        // Return iterator that performs systematic resampling
        (0..n).map(move |i| {
            let target = (i as f64 + u) / n as f64;

            // Find the first cumsum value >= target
            cumsum_vec
                .iter()
                .position(|&cum| cum >= target)
                .unwrap_or(n - 1)
        })
    }
}

/// Multinomial resampling strategy
///
/// Standard multinomial resampling where each particle is independently
/// sampled according to its weight. Has higher variance than systematic resampling.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialResampling;

// impl ResamplingStrategy for MultinomialResampling {
//     fn resample<R, I>(rng: &mut R, weights: I) -> impl Iterator<Item = usize>
//     where
//         R: Rng,
//         I: Iterator<Item = f64>,
//     {
//         todo!("Not implemented")
//     }
// }

/// Stratified resampling strategy
///
/// Similar to systematic resampling but uses independent random numbers
/// for each stratum, providing a middle ground between systematic and multinomial.
#[derive(Clone, Copy, Debug)]
pub struct StratifiedResampling;

// impl ResamplingStrategy for StratifiedResampling {
//     fn resample<R, I>(rng: &mut R, weights: I) -> impl Iterator<Item = usize>
//     where
//         R: Rng,
//         I: Iterator<Item = f64>,
//     {
//         todo!("Not implemented")
//     }
// }

/// Enum for dynamic dispatch over resampling strategies
#[derive(Clone, Copy, Debug)]
pub enum ResamplingStrategies {
    Systematic(SystematicResampling),
    Multinomial(MultinomialResampling),
    Stratified(StratifiedResampling),
}

impl ResamplingStrategies {
    pub fn from_str(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "systematic" => Ok(ResamplingStrategies::Systematic(SystematicResampling)),
            _ => Err(PyValueError::new_err(format!(
                "Unknown resampling strategy: '{}'",
                name
            ))),
        }
    }
}
