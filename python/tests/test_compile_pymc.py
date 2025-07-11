import pymc as pm
import numpy as np

from pymc_bart.compile_pymc import CompiledPyMCModel

n_obs = 1000
X = np.random.randn(n_obs, 3)
y_observed = np.random.randn(n_obs)

with pm.Model() as model:
    # These are SAMPLED (not shared)
    beta = pm.Normal("beta", 0, 1, shape=3)
    sigma = pm.HalfNormal("sigma", 1)

    mu = pm.Deterministic("mu", pm.math.dot(X, beta))  # X is shared
    y = pm.Normal("y", mu, sigma, observed=y_observed)  # y_observed is shared

compiled_model = CompiledPyMCModel(model, [beta, sigma])

# Check what's in the PyTensor function storage
print(f"Number of inputs: {len(compiled_model.logp_fn_ptr.input_storage)}")
# print(f"Input 0 (parameters): shape = {compiled_model.logp_fn_ptr.input_storage[0].storage[0].shape}")

# Check the shared arrays (these are the observed data)
for i, storage in enumerate(compiled_model.logp_fn_ptr.input_storage[1:], 1):
    data = storage.storage[0]
    print(f"Input {i} (shared): shape = {data.shape}, dtype = {data.dtype}")
    print(f"  First few values: {data.flat[:5]}")

# Pre-allocated shared arrays
print(f"Number of logp_args: {len(compiled_model.logp_args)}")
for i, arr in enumerate(compiled_model.logp_args):
    print(f"logp_args[{i}]: shape = {arr.shape}, dtype = {arr.dtype}")
