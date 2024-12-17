import pymc as pm

from pymc_bart.bart import BART
from pymc_bart.compile_pymc import CompiledPyMCModel
from pymc_bart.pgbart import PGBART
from pymc_bart.utils import (
    compute_variable_importance,
    plot_convergence,
    plot_dependence,
    plot_ice,
    plot_pdp,
    plot_scatter_submodels,
    plot_variable_importance,
    plot_variable_inclusion,
)

__all__ = [
    "BART",
    "PGBART",
    "CompiledPyMCModel",
    "compute_variable_importance",
    "plot_convergence",
    "plot_dependence",
    "plot_ice",
    "plot_pdp",
    "plot_scatter_submodels",
    "plot_variable_importance",
    "plot_variable_inclusion",
]

pm.STEP_METHODS = list(pm.STEP_METHODS) + [PGBART]
