//   Copyright 2024 The PyMC Developers
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
#![warn(missing_docs, clippy::needless_borrow)]

//! pg_bart provides an extensible implementation of Bayesian Additive
//! Regression Trees (BART). BART is a non-parametric method to
//! approximate functions based on the sum of many trees where
//! priors are used to regularize inference, mainly by restricting
//! a tree's learning capacity so that no individual tree is able
//! to explain the data, but rather the sum of trees. Inference is
//! performed using a sampler inspired by the Particle Gibbs method
//! introduced by Lakshminarayanan et al. [2015].

pub mod data;
pub mod math;
pub mod ops;
pub mod particle;
pub mod pgbart;
pub mod split_rules;
pub mod tree;
