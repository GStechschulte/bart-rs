import ctypes
import warnings

import numpy as np
import numba
import pandas as pd
import pymc as pm
import pytensor

from pymc.pytensorf import (
    compile_pymc,
    inputvars,
    join_nonshared_inputs,
    make_shared_replacements
)
from numba import carray, cfunc, extending, float64, types, njit
from numba.core import cgutils


@numba.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """Returns a void pointer from a given memory address

    Full credit goes to: https://stackoverflow.com/a/61550054
    """

    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


class CompiledPyMCModel:
    def __init__(self, model, vars):
        n_dims, pytensor_logp_fn, shared = self._make_functions(model, vars)
        self.n_dims = n_dims
        self.logp_fn_ptr = pytensor_logp_fn
        self.shared = shared

        self.logp_args = self._make_persistent_arrays()
        self.logp = self._generate_logp_function()

    def get_function_pointer(self):
        return self.logp.address

    def _make_functions(self, model, vars):
        initial_values = model.initial_point()
        shape = initial_values.get(str(vars[0])).shape[0]
        shared = make_shared_replacements(initial_values, vars, model)
        out_vars = [model.datalogp]

        new_out, new_joined_inputs = join_nonshared_inputs(
            initial_values,
            out_vars,
            vars,
            shared
        )

        logp_fn = compile_pymc(
            inputs=[new_joined_inputs],
            outputs=new_out[0],
            mode="NUMBA"
        )
        logp_fn.trust_input = True

        return shape, logp_fn, shared

    def _make_persistent_arrays(self):
        arrays = [item.storage[0].copy() for item in self.logp_fn_ptr.input_storage[1:]]
        assert all(arr.dtype == np.float64 for arr in arrays)
        return arrays


    def update_shared_arrays(self):
        """Update the persistent shared arrays with new values from the function storage"""
        for array, storage in zip(self.logp_args, self.logp_fn_ptr.input_storage[1:]):
            new_arr = storage.storage[0]
            assert array.shape == new_arr.shape
            array *= 0.0
            array += new_arr

    def _generate_logp_function(self):
        logp_fn = self.logp_fn_ptr.vm.jit_fn
        shared_arrays = self.logp_args

        code = [
            "def _logp(ptr, size):",
            "    data = carray(ptr, (size, ), dtype=float64)"
        ]

        for i, array in enumerate(shared_arrays):
            line = "    arg{} = carray(address_as_void_pointer({}), {}, dtype={})".format(
                i, array.ctypes.data, array.shape, array.dtype
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
