"""v2 sweeps answering SPEC.md §3 (see docs/findings_v2.md). Pin BLAS to one thread before numpy
loads: the sweeps parallelise over independent runs instead."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
