import dataclasses
import itertools
import warnings
from dataclasses import dataclass
from importlib.util import find_spec
from math import prod
from typing import TYPE_CHECKING, Any, Literal, Optional

import numba
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pytensor import function as pytensor_fn
from pymc.pytensorf import join_nonshared_inputs, make_shared_replacements
from numba.core.ccallback import CFunc
from numba import cfunc, types, njit

try:
    from numba.extending import intrinsic
except ImportError:

    def intrinsic(f):
        return f


if TYPE_CHECKING:
    import numba.core.ccallback
    import pymc as pm


@intrinsic
def address_as_void_pointer(typingctx, src):
    """returns a void pointer from a given memory address"""
    from numba.core import cgutils, types

    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


class ModelLogPWrapper:
    def __init__(self, model, vars):
        self.model = model
        self.vars = vars
        self.logp_fn = self.compile_logp()
        self.var_names = [var.name for var in vars]
        # self.compiled_logp_fn = self._compile_logp_fn(self.logp_fn)
        self.logp_fn_pointer = self.logp_function_pointer()
    
    def compile_logp(self):
        initial_values = self.model.initial_point()
        shared = make_shared_replacements(initial_values, self.vars, self.model)

        # Get the model's log probability
        logp = self.model.logp()

        # Join non-shared inputs
        out_list, inarray0 = join_nonshared_inputs(initial_values, [logp], self.vars, shared)

        # Compile the function
        logp_fn = pytensor_fn([inarray0], out_list[0], mode="FAST_RUN")
        logp_fn.trust_input = True

        # Compile the logp function
        # logp_numba = numba.cfunc

        print(f"logp_fn: {logp_fn}")

        return logp_fn
    
    def logp(self, x):
        """Evaluates the log probability for the given input."""
        if isinstance(x, dict):
            # Convert dictionary to 1D array
            x_array = np.concatenate([x[name].flatten() for name in self.var_names])
        elif isinstance(x, np.ndarray):
            x_array = x.flatten()
        else:
            raise ValueError("Input must be a dictionary or numpy array")
        
        return self.logp_fn(x_array)
    
    def logp_function_pointer(self):
        """Return the address of the compiled log probability function."""
        if hasattr(self.logp_fn.vm, 'thunks'):
            # Newer PyTensor versions
            return self.logp_fn.vm.thunks[0].cthunk
        elif hasattr(self.logp_fn.vm, 'thunk'):
            # Older PyTensor versions
            return self.logp_fn.vm.thunk.cfunction.cpointer
        else:
            # Fallback for other cases
            raise AttributeError("Unable to find the C function pointer for the log probability function")
    
    # def _compile_logp_fn(self, logp_fn):
    #     @cfunc(types.float64(types.CPointer(types.float64), types.int64))
    #     def logp_numba(x_ptr, size):
    #         x = numba.carray(x_ptr, (size,), dtype=numba.float64)
    #         return logp_fn(x)
        
    #     return CFunc(logp_numba.address, logp_numba.signature)

    # def logp_function_pointer(self):
    #     return self.compiled_logp_fn.address



def get_model_logp(model):
    """Create a ModelLogPWrapper for the given PyMC model."""
    with model:
        vars = model.value_vars
    return ModelLogPWrapper(model, vars)
