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
import ctypes

from time import perf_counter
from typing import Optional, Tuple

import numpy as np

from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor.graph.basic import Variable

from bart_rs.bart import BARTRV
from bart_rs.compile_pymc import compile_pymc_model_numba

from bart_rs.bart_rs import initialize, step


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
    # stats_dtypes = [{"variable_inclusion": object, "tune": bool}]
    # stats_dtypes = [{"time_rs": float, "tune": bool}]
    stats_dtypes = [{"time_rs": float}]

    def __init__(  # noqa: PLR0915
        self,
        vars=None,  # pylint: disable=redefined-builtin
        num_particles: int = 10,
        batch: Tuple[float, float] = (0.1, 0.1),
        model: Optional[Model] = None,
    ):

        model = modelcontext(model)
        initial_values = model.initial_point()
        self.compiled_pymc_model = compile_pymc_model_numba(model)

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

        self.m = self.bart.m
        self.response = self.bart.response

        shape = initial_values[value_bart.name].shape
        self.shape = 1 if len(shape) == 1 else shape[0]

        # Set trees_shape (dim for separate tree structures)
        # and leaves_shape (dim for leaf node values)
        # One of the two is always one, the other equal to self.shape
        self.trees_shape = self.shape if self.bart.separate_trees else 1
        self.leaves_shape = self.shape if not self.bart.separate_trees else 1

        if self.bart.split_prior.size == 0:
            self.alpha_vec = np.ones(self.X.shape[1])
        else:
            self.alpha_vec = self.bart.split_prior

        if self.bart.split_rules:
            self.split_rules = self.bart.split_rules
        else:
            self.split_rules = ["ContinuousSplit"] * self.X.shape[1]

        # If data is binary
        self.leaf_sd = np.ones((self.trees_shape, self.leaves_shape))

        y_unique = np.unique(self.bart.Y)
        if y_unique.size == 2 and np.all(y_unique == [0, 1]):
            self.leaf_sd *= 3 / self.m**0.5
        else:
            self.leaf_sd *= self.bart.Y.std() / self.m**0.5

        # Get the user data address
        if isinstance(self.compiled_pymc_model.user_data, ctypes.Array):
            user_data_address = ctypes.cast(self.compiled_pymc_model.user_data, ctypes.c_void_p).value
        else:
            # If it's a numpy array or other object, get its data pointer
            user_data_address = self.compiled_pymc_model.user_data.ctypes.data_as(ctypes.c_void_p).value

        print(f"self.compiled_pymc_model._n_dim: {self.compiled_pymc_model._n_dim}")

        # Initialize the Rust sampler
        self.state = initialize(
            X=self.X,
            y=self.bart.Y,
            logp=self.compiled_pymc_model.compiled_logp_func.address,
            n_dim=self.compiled_pymc_model._n_dim,
            user_data=user_data_address,
            alpha=self.bart.alpha,
            beta=self.bart.beta,
            split_prior=self.alpha_vec,
            split_rules=self.split_rules,
            response=self.bart.response,
            n_trees=self.bart.m,
            n_particles=num_particles,
            leaf_sd=self.leaf_sd,
            batch=batch
        )

        self.tune = True
        super().__init__(vars, self.compiled_pymc_model.shared_data)

    def astep(self, _):

        t0 = perf_counter()
        sum_trees = step(self.state, self.tune)
        t1 = perf_counter()

        # return sum_trees, [{"time_rs": t1 - t0}, {"tune": self.tune}]
        return sum_trees, [{"time_rs": t1 - t0}]

    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE
