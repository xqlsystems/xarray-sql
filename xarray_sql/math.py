"""Optional SQL mathematical primitives with equivalent array lowering.

DataFusion (https://datafusion.apache.org/) supplies most numerical SQL
functions. This module supplies the error function for exact GELU, the
activation used by nanoGPT (https://github.com/karpathy/nanoGPT).
"""

import math

import numpy as np
import pyarrow as pa
from datafusion import udf

try:
    from scipy.special import erf as _erf
except ImportError:
    _erf = np.vectorize(math.erf, otypes=[np.float64])

__all__ = ["register_erf"]


def _arrow_erf(values):
    return pa.array(
        _erf(values.to_numpy(zero_copy_only=False)),
        mask=values.is_null().to_numpy(zero_copy_only=False),
        type=pa.float64(),
    )


def register_erf(ctx):
    """Install SQL ``erf(DOUBLE)`` and its equivalent CPU/GPU primitive.

    Differentiation also requires a ddxdb build with an erf rule; see
    ``accelerators/ddx_math``. Replacing this UDF removes its trusted
    primitive status so custom SQL behavior is never silently bypassed.
    """
    ctx.register_udf(
        udf(_arrow_erf, [pa.float64()], pa.float64(), "immutable", name="erf")
    )
    ctx._trusted_math_udfs.add("erf")
