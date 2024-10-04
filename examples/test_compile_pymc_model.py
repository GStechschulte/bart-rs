import ctypes

import pandas as pd
import pymc as pm
import numpy as np

from pymc.logprob.transforms import FunctionGraph
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars, join_nonshared_inputs, make_shared_replacements
from pytensor.compile.function import function as pytensor_function

from numba import cfunc, types, njit

import bart_rs as pmb

from bart_rs import compile_pymc_model_numba


# Create a wrapper function to call the compiled log probability function
def logp_wrapper(compiled_model):
    def wrapper(x):
        n_dim = compiled_model._n_dim
        x_array = np.ascontiguousarray(x, dtype=np.float64)
        out_array = np.zeros(n_dim, dtype=np.float64)
        logp_array = np.zeros(1, dtype=np.float64)

        result = compiled_model.compiled_logp_func(
            ctypes.c_uint64(n_dim),
            x_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            logp_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            compiled_model.user_data.ctypes.data_as(ctypes.c_void_p)
        )

        # if result != 0:
        #     raise RuntimeError(f"Log probability evaluation failed with error code {result}")

        return logp_array[0], out_array

    return wrapper


def test_get_model_logp():

    np.random.seed(0)
    n = 50
    X = np.random.uniform(0, 10, n)
    Y = np.sin(X) + np.random.normal(0, 0.5, n)
    data = pd.DataFrame(data={'Feature': X.flatten(), 'Y': Y})

    with pm.Model() as simple_model:
        mu = pmb.BART("mu", X=X[..., None], Y=Y, m=5)
        # sigma = pm.HalfNormal("sigma", sigma=1)
        y = pm.Normal("y", mu, sigma=1., observed=Y)
        step = pmb.PGBART([mu], num_particles=3)

    pm.BinaryGibbsMetropolis
    print("\nBART method")
    print("-" * 25)
    model = modelcontext(simple_model) # modelcontext captures mu
    vars = model.value_vars
    initial_values = simple_model.initial_point()
    shared = make_shared_replacements(initial_values, vars, simple_model)

    print(f"vars: {vars}")
    print(f"initial_values: {initial_values}")
    print(f"shared: {shared}")

    out_list, inarray0 = join_nonshared_inputs(initial_values, [model.datalogp], vars, shared)
    function = pytensor_function([inarray0], out_list[0])
    function.trust_input = True

    print(f"logp function: {function}")

    init_mu = np.array(initial_values.get("mu", None))
    init_scale = np.array([initial_values.get("sigma_log__")])
    test_point = np.concatenate((init_mu, init_scale))

    print(f"logp: {function(init_mu)}")

    print("\nnutpie method")
    print("-" * 25)

    compiled_pymc = compile_pymc_model_numba(simple_model)

    # Both compiled funcs have a pointer to an address...
    print(f"compiled_logp_fn.address: {compiled_pymc.compiled_logp_func.address}")
    print(f"compiled_expand_func.address: {compiled_pymc.compiled_expand_func.address}")

    print(f"compiled_logp_fn: {compiled_pymc.compiled_logp_func}")
    print(f"compiled_expand_func: {compiled_pymc.compiled_expand_func}")
    print(f"logp_fn: {compiled_pymc.logp_func}")

    initial_point = simple_model.initial_point()
    print(f"initial_point: {initial_point}")

    model_vars = simple_model.value_vars
    print(f"model_vars: {model_vars}")

    # logp_func returns the logp and gradient
    logp_value, gradient = compiled_pymc.logp_func(init_mu)
    print(f"init logp: {logp_value}")

    print(f"shared: {compiled_pymc.shared_data}")

if __name__ == "__main__":
    test_get_model_logp()
