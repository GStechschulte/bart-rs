//! # Bayesian Additive Regression Trees crate
//!
//! This crate provides an implementation of Bayesian Additive
//! Regression Trees (BART). The core algorithm used to sample
//! BART is a Particle Gibbs sampler.

pub mod data;
pub mod math;
pub mod ops;
pub mod particle;
pub mod pgbart;
pub mod split_rules;
pub mod tree;
