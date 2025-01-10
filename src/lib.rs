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
#![warn(missing_docs)]
#![allow(non_snake_case)]

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

use crate::data::ExternalData;
use crate::ops::Response;
use crate::pgbart::{PgBartSettings, PgBartState};
use crate::split_rules::{ContinuousSplit, OneHotSplit, SplitRuleType};

use std::str::FromStr;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// `StateWrapper` wraps around `PgBartState` to hold state pertaining to
/// the Particle Gibbs sampler.
///
/// This class is `unsendable`, i.e., it cannot be sent across threads safely.
#[pyclass(unsendable)]
struct StateWrapper {
    state: PgBartState,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn initialize(
    X: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    logp: usize,
    alpha: f64,
    beta: f64,
    split_prior: PyReadonlyArray1<f64>,
    split_rules: Vec<String>,
    response: String,
    n_trees: usize,
    n_particles: usize,
    leaf_sd: Vec<f64>,
    batch: (f64, f64),
    leaves_shape: usize,
) -> PyResult<StateWrapper> {
    // Heap allocation because size of 'ExternalData' is not known at compile time
    let data = Box::new(ExternalData::new(X, y, logp));
    let response = Response::from_str(&response).unwrap();
    let mut rules: Vec<SplitRuleType> = Vec::new();

    for rule in split_rules {
        let split = match rule.as_str() {
            "ContinuousSplit" => SplitRuleType::Continuous(ContinuousSplit),
            "OneHotSplit" => SplitRuleType::OneHot(OneHotSplit),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown split type: {}",
                    rule
                )))
            }
        };
        rules.push(split);
    }

    let params = PgBartSettings::new(
        n_trees,
        n_particles,
        alpha,
        beta,
        leaf_sd,
        batch,
        split_prior.to_vec().unwrap(),
        response,
        rules,
        leaves_shape,
    );
    let state = PgBartState::new(params, data);

    Ok(StateWrapper { state })
}

#[pyfunction]
fn step<'py>(
    py: Python<'py>,
    wrapper: &mut StateWrapper,
    tune: bool,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>) {
    // Update whether or not `pm.sampler` is in tuning phase or not
    wrapper.state.tune = tune;
    // Run the Particle Gibbs sampler
    wrapper.state.step();

    // Get predictions (sum of trees) and convert to PyArray
    let predictions = wrapper.state.predictions();
    let py_preds_array = PyArray1::from_array_bound(py, &predictions.view());

    // Get variable inclusion counter and convert to PyArray
    let variable_inclusion = wrapper.state.variable_inclusion().clone();
    let py_variable_inclusion_array = PyArray1::from_vec_bound(py, variable_inclusion);

    (py_preds_array, py_variable_inclusion_array)
}

#[pymodule]
fn pymc_bart(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(step, m)?)?;

    Ok(())
}
