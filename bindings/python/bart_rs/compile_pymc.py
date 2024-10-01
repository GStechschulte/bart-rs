"""
TODO: Attribute nutepie here...
"""

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


@dataclass(frozen=True)
class CompiledPyMCModel:
    compiled_logp_func: "numba.core.ccallback.CFunc"
    compiled_expand_func: "numba.core.ccallback.CFunc"
    shared_data: dict[str, NDArray]
    user_data: NDArray
    n_expanded: int
    shape_info: Any
    logp_func: Any
    expand_func: Any
    _n_dim: int
    _shapes: dict[str, tuple[int, ...]]
    dims: Optional[dict[str, tuple[str, ...]]]
    _coords: Optional[dict[str, Any]]


@intrinsic
def address_as_void_pointer(typingctx, src):
    """returns a void pointer from a given memory address"""
    from numba.core import cgutils, types

    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


def make_extraction_fn(inner, shared_data, shared_vars, record_dtype):
    import numba
    from numba import literal_unroll
    from numba.cpython.unsafe.tuple import alloca_once, tuple_setitem

    if not shared_vars:

        @numba.njit(inline="always")
        def extract_shared(x, user_data_):
            return inner(x)

        return extract_shared

    shared_metadata = tuple(
        [
            name,
            len(shared_data[name].shape),
            shared_data[name].shape,
            np.dtype(shared_data[name].dtype),
        ]
        for name in shared_vars
    )

    names = shared_vars
    indices = tuple(range(len(names)))
    shared_tuple = tuple(shared_data[name] for name in shared_vars)

    @intrinsic
    def tuple_setitem_literal(typingctx, tup, idx, val):
        """Return a copy of the tuple with item at *idx* replaced with *val*."""
        if not isinstance(idx, numba.types.IntegerLiteral):
            return

        idx_val = idx.literal_value
        assert idx_val >= 0
        assert idx_val < len(tup)

        import llvmlite

        def codegen(context, builder, signature, args):
            tup, idx, val = args
            stack = alloca_once(builder, tup.type)
            builder.store(tup, stack)
            # Unsafe load on unchecked bounds.  Poison value maybe returned.
            tuple_idx = llvmlite.ir.IntType(32)(idx_val)
            offptr = builder.gep(stack, [idx.type(0), tuple_idx], inbounds=True)
            builder.store(val, offptr)
            return builder.load(stack)

        sig = tup(tup, idx, tup[idx_val])
        return sig, codegen

    def extract_array(user_data, index):
        pass

    @numba.extending.overload(extract_array, inline="always")
    def impl_extract_array(user_data, index):
        if not isinstance(index, numba.types.Literal):
            return

        index = index.literal_value

        name, ndim, base_shape, dtype = shared_metadata[index]

        def impl(user_data, index):
            data_ptr = address_as_void_pointer(user_data["data"][name][()])
            data = numba.carray(data_ptr, int(user_data["size"][name][()]), dtype)

            shape = user_data["shape"][name]

            assert len(shape) == len(base_shape)

            shape_ = base_shape

            # For some reason I get typing errors without this if condition
            if ndim > 0:
                for i in range(ndim):
                    shape_ = tuple_setitem(shape_, i, shape[i])

            return data.reshape(shape_)

        return impl

    @numba.njit(inline="always")
    def extract_shared(x, user_data_):
        user_data = numba.carray(user_data_, (), record_dtype)

        _shared_tuple = shared_tuple
        for index in literal_unroll(indices):
            dat = extract_array(user_data["shared"], index)
            _shared_tuple = tuple_setitem_literal(_shared_tuple, index, dat)

        return inner(x, *_shared_tuple)

    return extract_shared


def _make_c_logp_func(n_dim, logp_fn, user_data, shared_logp, shared_data):
    import numba

    extract = make_extraction_fn(logp_fn, shared_data, shared_logp, user_data.dtype)

    c_sig = numba.types.int64(
        numba.types.uint64,
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.voidptr,
    )

    def logp_numba(dim, x_, out_, logp_, user_data_):
        if dim != n_dim:
            return -1

        try:
            x = numba.carray(x_, (n_dim,))
            out = numba.carray(out_, (n_dim,))
            logp = numba.carray(logp_, ())

            logp_val, grad = extract(x, user_data_)
            logp[()] = logp_val
            out[...] = grad

            if not np.all(np.isfinite(out)):
                return 3
            if not np.isfinite(logp_val):
                return 4
            # if np.any(out == 0):
            #    return 4
        except Exception:  # noqa: BLE001
            return 1
        return 0

    return logp_numba, c_sig


def update_user_data(user_data, user_data_storage):
    user_data = user_data[()]
    for name, val in user_data_storage.items():
        user_data["shared"]["data"][name] = val.ctypes.data
        user_data["shared"]["size"][name] = val.size
        user_data["shared"]["shape"][name] = val.shape
    return np.asarray(user_data)


def make_user_data(shared_vars, shared_data):
    record_dtype = np.dtype(
        [
            (
                "shared",
                [
                    ("data", [(var_name, np.uintp) for var_name in shared_vars]),
                    ("size", [(var_name, np.uintp) for var_name in shared_vars]),
                    (
                        "shape",
                        [
                            (var_name, np.uint, (var.ndim,))
                            for var_name, var in shared_vars.items()
                        ],
                    ),
                ],
            )
        ],
    )
    user_data = np.zeros((), dtype=record_dtype)
    update_user_data(user_data, shared_data)
    return user_data

def _compute_shapes(model):
    import pytensor
    from pymc.initial_point import make_initial_point_fn

    point = make_initial_point_fn(model=model, return_transformed=True)(0)

    trace_vars = {
        var.name: var
        for var in model.value_vars + model.free_RVs + model.deterministics
        if var not in model.observed_RVs + model.potentials
    }

    shape_func = pytensor.compile.function.function(
        inputs=[],
        outputs=[var.shape for var in trace_vars.values()],
        givens=(
            [(obs, model.rvs_to_values[obs]) for obs in model.observed_RVs]
            + [
                (trace_vars[name], point[name])
                for name in trace_vars.keys()
                if name in point
            ]
        ),
        mode=pytensor.compile.mode.FAST_COMPILE,
        on_unused_input="ignore",
    )
    return dict(zip(trace_vars.keys(), shape_func()))


def _make_functions(model, *, mode, compute_grad, join_expanded):
    import pytensor
    import pytensor.link.numba.dispatch
    import pytensor.tensor as pt
    from pymc.pytensorf import compile_pymc

    shapes = _compute_shapes(model)

    # Make logp_dlogp_function
    joined = pt.dvector("__joined_variables")

    value_vars = [model.rvs_to_values[var] for var in model.free_RVs]

    logp = model.logp()

    rewrites = ["canonicalize", "stabilize"]
    if not model.check_bounds:
        rewrites.append("local_remove_check_parameter")

    logp = pytensor.graph.rewrite_graph(logp, include=rewrites)

    if compute_grad:
        grads = pytensor.gradient.grad(logp, value_vars)
        grad = pt.concatenate([grad.ravel() for grad in grads])

    count = 0
    joined_slices = []
    joined_shapes = []
    joined_names = []

    splits = []

    for var in model.free_RVs:
        value_var = model.rvs_to_values[var]
        joined_names.append(value_var.name)
        shape = shapes[value_var.name]
        joined_shapes.append(shape)
        length = prod(shape)
        slice_val = slice(count, count + length)
        joined_slices.append(slice_val)
        count += length

        splits.append(length)

    num_free_vars = count

    joined = pt.TensorType("float64", shape=(num_free_vars,))(
        name="_unconstrained_point"
    )

    use_split = False
    if use_split:
        variables = pt.split(joined, splits, len(splits))
    else:
        variables = [joined[slice_val] for slice_val in zip(joined_slices)]

    replacements = {
        model.rvs_to_values[var]: value.reshape(shape).astype(var.dtype)
        for var, shape, value in zip(
            model.free_RVs,
            joined_shapes,
            variables,
        )
    }

    if compute_grad:
        (logp, grad) = pytensor.clone_replace([logp, grad], replacements)
        with model:
            logp_fn_pt = compile_pymc((joined,), (logp, grad), mode=mode)
    else:
        (logp,) = pytensor.clone_replace([logp], replacements)
        with model:
            logp_fn_pt = compile_pymc((joined,), (logp,), mode=mode)

    # Make function that computes remaining variables for the trace
    remaining_rvs = [
        var for var in model.unobserved_value_vars if var.name not in joined_names
    ]

    all_names = joined_names + remaining_rvs

    all_names = joined_names.copy()
    all_slices = joined_slices.copy()
    all_shapes = joined_shapes.copy()

    for var in remaining_rvs:
        all_names.append(var.name)
        shape = shapes[var.name]
        all_shapes.append(shape)
        length = prod(shape)
        all_slices.append(slice(count, count + length))
        count += length

    num_expanded = count

    if join_expanded:
        allvars = [pt.concatenate([joined, *[var.ravel() for var in remaining_rvs]])]
    else:
        allvars = [*variables, *remaining_rvs]
    with model:
        expand_fn_pt = compile_pymc(
            (joined,),
            allvars,
            givens=list(replacements.items()),
            mode=mode,
        )

    return (
        num_free_vars,
        num_expanded,
        logp_fn_pt,
        expand_fn_pt,
        (all_names, all_slices, all_shapes),
    )

def compile_pymc_model_numba(model, **kwargs):
    if find_spec("numba") is None:
        raise ImportError(
            "Numba is not installed in the current environment. "
            "Please install it with something like "
            "'mamba install -c conda-forge numba' "
            "and restart your kernel in case you are in an interactive session."
        )
    import numba

    (
        n_dim,
        n_expanded,
        logp_fn_pt,
        expand_fn_pt,
        shape_info,
    ) = _make_functions(model, mode="NUMBA", compute_grad=True, join_expanded=True)

    # Numba compiled logp function
    expand_fn = expand_fn_pt.vm.jit_fn
    logp_fn = logp_fn_pt.vm.jit_fn

    shared_data = {}
    shared_vars = {}
    seen = set()
    for val in [*logp_fn_pt.get_shared(), *expand_fn_pt.get_shared()]:
        if val.name in shared_data and val not in seen:
            raise ValueError(f"Shared variables must have unique names: {val.name}")
        shared_data[val.name] = val.get_value()
        shared_vars[val.name] = val
        seen.add(val)

    for val in shared_data.values():
        val.flags.writeable = False

    user_data = make_user_data(shared_vars, shared_data)

    logp_shared_names = [var.name for var in logp_fn_pt.get_shared()]

    logp_numba_raw, c_sig = _make_c_logp_func(
        n_dim, logp_fn, user_data, logp_shared_names, shared_data
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Cannot cache compiled function .* as it uses dynamic globals",
            category=numba.NumbaWarning,
        )

        logp_numba = numba.cfunc(c_sig, **kwargs)(logp_numba_raw)


    expand_shared_names = [var.name for var in expand_fn_pt.get_shared()]
    expand_numba_raw, c_sig_expand = _make_c_expand_func(
        n_dim, n_expanded, expand_fn, user_data, expand_shared_names, shared_data
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Cannot cache compiled function .* as it uses dynamic globals",
            category=numba.NumbaWarning,
        )

        expand_numba = numba.cfunc(c_sig_expand, **kwargs)(expand_numba_raw)

    dims, coords = _prepare_dims_and_coords(model, shape_info)

    return CompiledPyMCModel(
        _n_dim=n_dim,
        dims=dims,
        _coords=coords,
        _shapes={name: tuple(shape) for name, _, shape in zip(*shape_info)},
        compiled_logp_func=logp_numba,
        compiled_expand_func=expand_numba,
        shared_data=shared_data,
        user_data=user_data,
        n_expanded=n_expanded,
        shape_info=shape_info,
        logp_func=logp_fn_pt,
        expand_func=expand_fn_pt,
    )


def _make_c_expand_func(
    n_dim, n_expanded, expand_fn, user_data, shared_vars, shared_data
):
    import numba

    extract = make_extraction_fn(expand_fn, shared_data, shared_vars, user_data.dtype)

    c_sig = numba.types.int64(
        numba.types.uint64,
        numba.types.uint64,
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double),
        numba.types.voidptr,
    )

    def expand_numba(dim, expanded, x_, out_, user_data_):
        if dim != n_dim:
            return -1
        if expanded != n_expanded:
            return -1

        try:
            x = numba.carray(x_, (n_dim,))
            out = numba.carray(out_, (n_expanded,))

            (values,) = extract(x, user_data_)
            out[...] = values

        except Exception:  # noqa: BLE001
            return -2
        return 0

    return expand_numba, c_sig


def _prepare_dims_and_coords(model, shape_info):
    coords = {}
    for name, vals in model.coords.items():
        if vals is None:
            vals = pd.RangeIndex(int(model.dim_lengths[name].eval()))
        coords[name] = pd.Index(vals)

    if "unconstrained_parameter" in coords:
        raise ValueError("Model contains invalid name 'unconstrained_parameter'.")

    names = []
    for base, _, shape in zip(*shape_info):
        if base not in [var.name for var in model.value_vars]:
            continue
        for idx in itertools.product(*[range(length) for length in shape]):
            if len(idx) == 0:
                names.append(base)
            else:
                names.append(f"{base}_{'.'.join(str(i) for i in idx)}")
    coords["unconstrained_parameter"] = pd.Index(names)

    dims = model.named_vars_to_dims
    return dims, coords
