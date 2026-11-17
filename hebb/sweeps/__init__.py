"""Experimental sweeps (see docs/findings.md). Pin BLAS to one thread: the sweeps parallelise over runs."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
