"""Native thread pinning for the many small model fits of the forecasters.

Setting OMP_NUM_THREADS / OPENBLAS_NUM_THREADS only works before numpy, scipy or LightGBM load their native
libraries; once loaded (the API, notebooks, the pipeline) they keep one thread per core. With 18 OpenBLAS threads per
process a 252 x 50 BayesianRidge fit takes ~540 ms instead of ~1 ms. `pin_native_threads` sets the variables for
child processes and limits the already-loaded BLAS / OpenMP pools at runtime via threadpoolctl.
"""

from __future__ import annotations

import os

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
_limiter = None


def pin_native_threads(n_threads: int = 1) -> None:
    """Limit BLAS / OpenMP thread pools of this process (and children started later) to `n_threads`."""
    global _limiter
    for var in _THREAD_VARS:
        os.environ[var] = str(n_threads)
    try:
        from threadpoolctl import threadpool_limits

        _limiter = threadpool_limits(limits=n_threads)  # kept referenced: the limit stays in force for the process
    except ImportError:  # threadpoolctl ships with scikit-learn; environment variables still cover child processes
        pass
