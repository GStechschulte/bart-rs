import pymc as pm

from bart_rs.bart import BART
from bart_rs.compile_pymc import CompiledPyMCModel
from bart_rs.pgbart import PGBART
from bart_rs.utils import (
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
