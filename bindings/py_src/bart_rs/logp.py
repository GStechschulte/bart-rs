import dataclasses
import itertools
import warnings
from dataclasses import dataclass
from importlib.util import find_spec
from math import prod
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# from nutpie import _lib
# from nutpie.compiled_pyfunc import from_pyfunc
# from nutpie.sample import CompiledModel

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