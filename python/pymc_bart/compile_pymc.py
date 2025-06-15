"""
This module provides functionality to compile PyMC models into C-callable functions
using Numba.
"""

import ctypes
import warnings

import numpy as np
import numba
import pandas as pd
import pymc as pm
import pytensor

from pymc.pytensorf import (
    compile,
    inputvars,
    join_nonshared_inputs,
    make_shared_replacements,
)
from numba import carray, cfunc, extending, float64, types, njit
from numba.core import cgutils


@numba.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """Returns a void pointer from a given memory address.

    This intrinsic function allows Numba-compiled code to access memory
    at a specific address, enabling zero-copy access to pre-allocated
    NumPy arrays from within compiled functions.

    Parameters
    ----------
    typingctx : numba typing context
        Numba's typing context (automatically provided)
    src : numba type
        Source memory address as an integer type

    Returns
    -------
    sig : numba signature
        Function signature for the intrinsic
    codegen : callable
        Code generation function

    Note
    ----
    Credit goes to: https://stackoverflow.com/a/61550054
    """

    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        """Generate LLVM IR code to convert integer address to void pointer"""
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


class CompiledPyMCModel:
    """A compiled PyMC model optimized for high-frequency log probability evaluation.

    This class compiles a PyMC model into a C-callable function using Numba.  The
    implementation uses pre-allocated arrays for shared variables and direct memory
    access to minimize per-call overhead.

    Attributes
    ----------
    n_dims : int
        Number of dimensions (parameters) in the model
    logp_fn_ptr : compiled function
        PyTensor compiled log probability function
    shared : dict
        Shared variable replacements from PyMC
    logp_args : list of np.ndarray
        Pre-allocated arrays for shared variables (cache-friendly)
    logp : numba.cfunc
        Compiled C-callable log probability function

    Examples
    --------
    >>> import pymc as pm
    >>> with pm.Model() as model:
    ...     x = pm.Normal('x', 0, 1)
    ...     y = pm.Normal('y', x, 1, observed=data)
    >>> compiled_model = CompiledPyMCModel(model, [x])
    >>> function_ptr = compiled_model.get_function_pointer()
    """
    def __init__(self, model, vars):
        """Initialize and compile the PyMC model.

        Parameters
        ----------
        model : pymc.Model
            The PyMC model to compile
        vars : list
            List of model variables to include in compilation
        """
        # Compile PyTensor functinos and extract PyMC model information
        n_dims, pytensor_logp_fn, shared = self._make_functions(model, vars)
        self.n_dims = n_dims
        self.logp_fn_ptr = pytensor_logp_fn
        self.shared = shared

        # Pre-allocate arrays for shared variables
        self.logp_args = self._make_persistent_arrays()
        # Generate the final C-callback Numba function
        self.logp = self._generate_logp_function()

    def get_function_pointer(self):
        """Get the memory address of the compiled C function.

        Returns
        -------
        int
            Memory address of the compiled function that can be passed
            to external libraries (e.g., Rust via FFI)
        """
        return self.logp.address

    def _make_functions(self, model, vars):
        """Compile PyTensor functions from the PyMC model.

        This method transforms the PyMC model into compiled PyTensor functions
        that can be called efficiently. It handles shared variable replacements
        and prepares the log probability function for compilation.

        Parameters
        ----------
        model : pymc.Model
            PyMC model to compile
        vars : list
            Model variables to include

        Returns
        -------
        shape : int
            Number of model parameters
        logp_fn : compiled function
            Compiled PyTensor log probability function
        shared : dict
            Shared variable replacements
        """
        initial_values = model.initial_point()
        shape = initial_values.get(str(vars[0])).shape[0]
        shared = make_shared_replacements(initial_values, vars, model)
        out_vars = [model.datalogp]

        new_out, new_joined_inputs = join_nonshared_inputs(
            initial_values, out_vars, vars, shared
        )

        logp_fn = compile(
            inputs=[new_joined_inputs], outputs=new_out[0], mode="NUMBA"
        )
        logp_fn.trust_input = True

        return shape, logp_fn, shared

    def _make_persistent_arrays(self):
        """Create persistent copies of shared variable arrays.

        This method creates NumPy array copies of all shared variables
        that will persist in memory and can be accessed directly from
        the compiled function. This avoids runtime pointer dereferencing
        and provides better cache locality for frequent calls.

        Returns
        -------
        list of np.ndarray
            Pre-allocated arrays containing shared variable data

        Note
        ----
        All arrays are ensured to be float64 for consistency with
        the compiled function interface.
        """
        arrays = [item.storage[0].copy() for item in self.logp_fn_ptr.input_storage[1:]]
        assert all(arr.dtype == np.float64 for arr in arrays)
        return arrays

    # TODO: fast update for shared arrays
    # @njit
    # def _fast_update(self, dest, src):
    #     for i in range(len(dest)):
    #         dest[i] = src[i]

    def update_shared_arrays(self):
        """Update the persistent shared arrays with new values from the function storage.

        This method synchronizes the pre-allocated shared arrays with any
        changes made to the underlying PyTensor shared variables. Call this
        method whenever shared variables (e.g., observed data) are modified
        to ensure the compiled function uses the updated values.

        Note
        ----
        'logp_fn_ptr.input_storage' contains PyTensor shared variables used
        for computing the log probability. The first element [0] is the main
        input parameter array, elements [1:] are shared variables.
        """
        for array, storage in zip(self.logp_args, self.logp_fn_ptr.input_storage[1:]):
            np.copyto(array, storage.storage[0])
            # self._fast_update(array, storage.storage[0])

    def _generate_logp_function(self):
        """Generate the final C-callable log probability function.

        This method creates a Numba-compiled C function that can be called
        from external code (e.g., Rust via FFI). The function signature is:

        `double logp(double* ptr, int size)`

        The generated function:
        1. Wraps the input pointer as a NumPy array
        2. Accesses pre-allocated shared arrays using direct memory addresses
        3. Calls the PyTensor compiled function
        4. Returns the log probability as a scalar

        Returns
        -------
        numba.cfunc
            Compiled C function with address accessible via the `.address` attribute
        """
        logp_fn = self.logp_fn_ptr.vm.jit_fn
        shared_arrays = self.logp_args

        code = [
            "def _logp(ptr, size):",
            "    data = carray(ptr, (size, ), dtype=float64)",
        ]

        for i, array in enumerate(shared_arrays):
            line = (
                "    arg{} = carray(address_as_void_pointer({}), {}, dtype={})".format(
                    i, array.ctypes.data, array.shape, array.dtype
                )
            )
            code.append(line)

        ret = f"    return logp_fn(data, {', '.join(f'arg{i}' for i in range(len(shared_arrays)))})[0].item()"
        code.append(ret)
        source = "\n".join(code)

        ldict = locals()
        gdict = {**globals(), **locals()}
        exec(source, gdict, ldict)

        logp = njit(ldict["_logp"])

        sig = types.float64(
            types.CPointer(types.float64),
            types.intc,
        )
        logp = cfunc(sig)(logp)

        return logp
