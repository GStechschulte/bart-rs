import pymc as pm
import numpy as np

from bart_rs import compile_pymc_model_numba

def test_get_model_logp():

    with pm.Model() as simple_model:
        mu = pm.Normal('mu', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=1)
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=np.random.randn(100))

    compiled_pymc = compile_pymc_model_numba(simple_model)
    print(f"compiled_logp_fn: {compiled_pymc.compiled_logp_func.address}")
    print(f"logp_fn: {compiled_pymc.logp_func}")
    
    # Get the ModelLogPWrapper
    # wrapper = get_model_logp(simple_model)
    # wrapper2 = _get_model_logp(simple_model)

    # Test the logp method
    initial_point = simple_model.initial_point()
    print(f"initial_point: {initial_point}")

    # logp_value = wrapper.logp(initial_point)
    model_vars = simple_model.value_vars
    print(f"model_vars: {model_vars}")

    # init_val = np.concatenate([initial_point[name].flatten() for name in model_vars])
    test_val = np.array([0., 0.])
    logp_value = compiled_pymc.logp_func(test_val)
    print(f"init logp: {logp_value}")

    # print(f"Log probability at initial point: {logp_value}")

    # # Test that we can get the function pointer
    # func_pointer = wrapper.logp_function_pointer()
    # print(f"Log probability function pointer: {func_pointer}")

    # Assert that the logp value is a float
    # assert isinstance(logp_value, float), "Log probability should be a float"

    # Assert that the function pointer is not None
    # assert func_pointer is not None, "Function pointer should not be None"

if __name__ == "__main__":
    test_get_model_logp()