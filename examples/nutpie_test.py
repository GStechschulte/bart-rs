import pymc as pm
import numpy as np
import nutpie
import pandas as pd


def main():

    # Load the radon dataset
    data = pd.read_csv(pm.get_data("radon.csv"))
    data["log_radon"] = data["log_radon"].astype(np.float64)
    county_idx, counties = pd.factorize(data.county)
    coords = {"county": counties, "obs_id": np.arange(len(county_idx))}

    # Create a simple hierarchical model for the radon dataset
    with pm.Model(coords=coords, check_bounds=False) as pymc_model:
        intercept = pm.Normal("intercept", sigma=10)

        # County effects
        raw = pm.ZeroSumNormal("county_raw", dims="county")
        sd = pm.HalfNormal("county_sd")
        county_effect = pm.Deterministic("county_effect", raw * sd, dims="county")

        # Global floor effect
        floor_effect = pm.Normal("floor_effect", sigma=2)

        # County:floor interaction
        raw = pm.ZeroSumNormal("county_floor_raw", dims="county")
        sd = pm.HalfNormal("county_floor_sd")
        county_floor_effect = pm.Deterministic(
            "county_floor_effect", raw * sd, dims="county"
        )

        mu = (
            intercept
            + county_effect[county_idx]
            + floor_effect * data.floor.values
            + county_floor_effect[county_idx] * data.floor.values
        )

        sigma = pm.HalfNormal("sigma", sigma=1.5)
        pm.Normal(
            "log_radon", mu=mu, sigma=sigma, observed=data.log_radon.values, dims="obs_id"
        )

    # Use nutpie to compile the PyMC model into...
    compiled_model = nutpie.compile_pymc_model(pymc_model)
    print(compiled_model)
    trace_pymc = nutpie.sample(compiled_model)


if __name__ == "__main__":
    main()
