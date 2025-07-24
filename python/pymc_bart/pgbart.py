#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import math

from time import perf_counter
from typing import Optional, Tuple

import numpy as np

from pymc.initial_point import PointType
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor.graph.basic import Variable

from pymc_bart.bart import BARTRV
from pymc_bart.compile_pymc import CompiledPyMCModel
from pymc_bart.pymc_bart import PyBartSettings, PySampler


class PGBART(ArrayStepShared):
    """
    Particle Gibss BART sampling step.

    Parameters
    ----------
    vars: list
        List of value variables for sampler
    num_particles : tuple
        Number of particles. Defaults to 10
    batch : tuple
        Number of trees fitted per step. The first element is the batch size during tuning and the
        second the batch size after tuning.  Defaults to  (0.1, 0.1), meaning 10% of the `m` trees
        during tuning and after tuning.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """

    name = "pgbart"
    default_blocked = False
    generates_stats = True
    stats_dtypes_shapes: dict[str, tuple[type, list]] = {
        "variable_inclusion": (object, []),
        "tune": (bool, []),
        "time": (float, []),
    }

    def __init__(  # noqa: PLR0915
        self,
        vars=None,  # pylint: disable=redefined-builtin
        num_particles: int = 10,
        batch: Tuple[float, float] = (0.1, 0.1),
        model: Optional[Model] = None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,  # pylint: disable=unused-argument
    ):
        model = modelcontext(model)
        if initial_point is None:
            initial_point = model.initial_point()

        # Get the instances of the BART random variable from the PyMC model
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)
        value_bart = vars[0]
        self.bart = model.values_to_rvs[value_bart].owner.op

        if isinstance(self.bart.X, Variable):
            self.X = self.bart.X.eval()
        else:
            self.X = self.bart.X

        if isinstance(self.bart.Y, Variable):
            self.Y = self.bart.Y.eval()
        else:
            self.Y = self.bart.Y

        self.m = self.bart.m
        self.response = self.bart.response

        shape = initial_point[value_bart.name].shape
        self.shape = 1 if len(shape) == 1 else shape[0]

        print(f"shape: {shape}, self.shape: {self.shape}")

        # Set trees_shape (dim for separate tree structures)
        # and leaves_shape (dim for leaf node values)
        # One of the two is always one, the other equal to self.shape
        self.trees_shape = self.shape if self.bart.separate_trees else 1
        self.leaves_shape = self.shape if not self.bart.separate_trees else 1

        if self.bart.split_prior.size == 0:
            self.alpha_vec = np.ones(self.X.shape[1])
        else:
            self.alpha_vec = self.bart.split_prior

        splitting_probs = np.cumsum(self.alpha_vec)

        print(f"split_rules: {self.bart.split_rules}")

        if self.bart.split_rules:
            self.split_rules = self.bart.split_rules
        else:
            self.split_rules = ["ContinuousSplit"] * self.X.shape[1]

        # If data is binary
        # self.leaf_sd = np.ones((self.trees_shape, self.leaves_shape))
        self.leaf_sd = np.ones(self.leaves_shape)

        y_unique = np.unique(self.bart.Y)
        if y_unique.size == 2 and np.all(y_unique == [0, 1]):
            self.leaf_sd *= 3 / self.m**0.5
        else:
            self.leaf_sd *= self.bart.Y.std() / self.m**0.5

        init_leaf_value = np.mean(self.bart.Y) / self.bart.m

        # Compile the PyMC model to create a C callback. This function pointer is
        # passed to Rust and called using Rust's foreign function interface (FFI)
        self.compiled_pymc_model = CompiledPyMCModel(model, vars)

        # Set a random u64 seed for reproducibility
        seed = np.random.randint(2 ** 31 - 1)

        # TODO: Initialize the settings that correspond to the BART extension
        # if self.bart.response == "constant":
        #     settings = PyBartSettings.Constant()
        # elif self.bart_response == "motr":
        #     settings = PyBartSettings.Motr()
        # elif self.bart_response == "tvp":
        #     settings = PyBartSettings.Tvp()
        # elif self.bart_response == "gp":
        #     settings = PyBartSettings.Gp()

        self._sum_of_trees_buffer = np.zeros(self.bart.Y.shape[0], dtype=np.float64)
        max_depth = calculate_max_tree_depth(self.bart.alpha, self.bart.beta, probs_leaf=0.99)
        max_nodes_per_tree = 2 ** (max_depth + 1) - 1

        print(f"init_leaf_value: {init_leaf_value}")
        print(f"response_rule: {self.bart.response}")
        print(f"alpha vector: {self.alpha_vec}")
        print(f"splitting probability: {splitting_probs}")
        print(f"alpha: {self.bart.alpha}, beta: {self.bart.beta}")
        print(f"max_depth: {max_depth}")
        print(f"max_nodes_per_tree: {max_nodes_per_tree}")
        print(f"self._sum_of_trees_buffer: {self._sum_of_trees_buffer}")


        # Build the Particle Gibbs sampler
        settings = PyBartSettings(
            init_leaf_value=init_leaf_value,
            init_leaf_std=self.leaf_sd,
            n_trees=self.bart.m,
            n_particles=num_particles,
            max_depth=max_depth,
            alpha=self.bart.alpha,
            beta=self.bart.beta,
            split_prior=splitting_probs,
            split_rules_py=self.bart.split_rules,
            response_rule=self.bart.response,
            resampling_rule="systematic",
            batch_size=batch
        )

        # INFO: Only at the end do we return the State structure back to Python to avoid
        #       overhead. Otherwise, we would need to create create a PyState to wrap
        #       around to contain the PgBartState in order to return the results back to the user.
        #
        # pg = PySampler(...)
        # state = pg.init(...)
        # new_state, info = pg.step(rng, state)

        self.pg_bart = PySampler.init(
            x=self.X,
            y=self.bart.Y,
            model=self.compiled_pymc_model.get_function_pointer(),
            settings=settings
        )

        self.tune = True
        super().__init__(vars, self.compiled_pymc_model.shared)

    def astep(self, _):
    #     # Record time to quantify performance improvements
        t0 = perf_counter()
    #     self.compiled_pymc_model.update_shared_arrays()
    #     sum_trees, variable_inclusion = step(self.state, self.tune)
        # sum_trees = self.pg_bart.step()
        sum_trees = np.zeros(self.bart.y.len())
        t1 = perf_counter()

        print((t1 - t0) * 1e6)

        stats = {
            "variable_inclusion": np.array([0.0]),
            "tune": self.tune,
            "time": t1 - t0,
        }

        return sum_trees, [stats]


    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


def calculate_max_tree_depth(alpha: float, beta: float, probs_leaf: float) -> int:
    """Calculates the maximum tree depth for which the probability of a node
    remaining a leaf node is given by the `probs_leaf`.

    Parameters
    ----------
    alpha : float
        Base prior probability parameter, between 0 and 1.
    beta : float
        Prior probability decaying factor, greater than or equal to 0.
    probs_leaf : float
        Probability a node remains a leaf node.

    Returns
    -------
    int : The calculated maximum tree depth
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1. Received {alpha}")
    if not (0 < probs_leaf < 1):
        raise ValueError(f"alpha must be between 0 and 1. Received {alpha}")
    if beta <= 0:
        raise ValueError(f"beta must be greater than 0. Received {beta}")

    probs_not_leaf = 1 - probs_leaf
    reciprocal = 1 / probs_not_leaf
    term = reciprocal * alpha
    exponent = 1.0 / beta
    depth = int(math.pow(term, exponent) - 1)
    return depth
