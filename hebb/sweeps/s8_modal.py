"""Run s8_private_noise jobs on Modal CPU containers (the VPS is shared; these are pure-numpy units).

Used automatically by s8_private_noise when S8_MODAL=1:
    S8_MODAL=1 PYTHONPATH=. /var/tmp/hebb-mnist-venv/bin/python -u -m hebb.sweeps.s8_private_noise A
Each unit = one job tuple; results come back directly (small), nothing is stored on Modal.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.2.6", "scipy")
    .env({"PYTHONPATH": "/opt/proj", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(str(ROOT / "hebb"), "/opt/proj/hebb",
                   ignore=["**/__pycache__", "**/modal_runs", "**/configs", "mnist/**", "nonlin/**"])
)
app = modal.App("crw-hebbian-s8-private-noise")


@app.function(image=IMAGE, cpu=1.0, memory=2048, timeout=60 * 60, max_containers=64)
def run_job(name, args):
    from hebb.sweeps import s8_private_noise as s8
    return getattr(s8, name)(args)


def modal_map(name, jobs):
    with modal.enable_output():
        with app.run():
            return list(run_job.map([name] * len(jobs), jobs))
