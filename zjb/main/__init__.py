import os
import sys
import warnings

# 尝试禁用numpy底层的多线程, 因为实践中多线程往往会降低计算效率
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

if "numpy" in sys.modules:
    if "OMP_NUM_THREADS" not in os.environ or os.environ["OMP_NUM_THREADS"] != "1":
        warnings.warn(
            "Numpy has been imported so that multi-threading may not be successfully disabled,"
            "which is likely to lead to a reduction in efficiency."
        )
