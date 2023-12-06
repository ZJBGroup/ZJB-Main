import numba as nb
import numpy as np

@nb.njit(fastmath=True, inline="always")
def func(
    ${expr._arg_str}
):
    return ${expr.expr}