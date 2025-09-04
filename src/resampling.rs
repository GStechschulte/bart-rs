//! Resampling strategies for particle filtering in BART.
//!
//! This module provides different resampling methods for selecting particles
//! based on their normalized weights.
use rand::Rng;

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

/// Resampling interface for implementing different resampling methods
pub trait ResamplingStrategy {
    /// Resample particle indices based on their normalized weights
    // fn resample<R, I>(rng: &mut R, weights: I) -> impl Iterator<Item = usize>
    fn resample<R>(rng: &mut R, weights: &[f64]) -> Vec<usize>
    where
        R: Rng;
    // I: Iterator<Item = f64>;
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
    // fn resample<R, I>(rng: &mut R, weights: I) -> impl Iterator<Item = usize>
    fn resample<R>(rng: &mut R, weights: &[f64]) -> Vec<usize>
    where
        R: Rng,
        // I: Iterator<Item = f64>,
    {
        // let weights_vec: Vec<f64> = weights.collect();
        println!("normalized weights: {:?}", weights);
        let n = weights.len();
        let u = rng.random::<f64>();

        let mut current_idx = 0usize;
        let mut current_cum = 0.0f64;
        let mut ancestors = Vec::with_capacity(n);

        for i in 0..n {
            let target = (i as f64 + u) / n as f64;

            while current_cum < target && current_idx < n {
                current_cum += weights[current_idx];
                current_idx += 1;
            }

            let ancestor = if current_idx == 0 { 0 } else { current_idx - 1 };
            ancestors.push(ancestor);
        }

        ancestors
        // let weights_vec: Vec<f64> = weights.collect();
        // println!("normalized weights: {:?}", weights_vec);
        // let n = weights_vec.len();
        // let u = rng.random::<f64>(); // Random offset between 0 and 1

        // let mut current_idx = 0usize;
        // let mut current_cum = 0.0f64;

        // (0..n).map(move |i| {
        //     let target = (i as f64 + u) / n as f64;
        //     while current_cum < target && current_idx < n {
        //         current_cum += weights_vec[current_idx];
        //         current_idx += 1;
        //     }
        //     // If we've reached the end, clamp to the last index
        //     if current_idx == 0 { 0 } else { current_idx - 1 }
        // })
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
