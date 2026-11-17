"""Modal batch runner for the MNIST Hebbian experiment (adapted from the modal skill template).

    cd /home/ubuntu/code/HebbianHarmonics
    PY=/var/tmp/hebb-mnist-venv/bin/python
    $PY hebb/mnist/modal_job.py deploy
    $PY hebb/mnist/modal_job.py launch hebb/mnist/configs/<grid>.json
    $PY hebb/mnist/modal_job.py status <run_id>
    $PY hebb/mnist/modal_job.py fetch  <run_id> /var/tmp/hebb-mnist/runs

Each unit is one training configuration (a hebb.mnist.model.Cfg dict); it writes
<name>.npz and <name>.json under /results/<run_id>/ on the Volume.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import os

import modal

BACKEND = os.environ.get("MNIST_BACKEND", "gpu")      # "gpu" (L4) or "cpu" (twin app, 8 cores)
APP_NAME = "crw-hebbian-mnist" + ("" if BACKEND == "gpu" else "-cpu")
VOL_NAME = "crw-hebbian-mnist-results"
GPU = ["L4", "A10G", "T4"] if BACKEND == "gpu" else None   # fallback list: L4 capacity was scarce
TIMEOUT_S = 60 * 60
MAX_CONCURRENCY = 20
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "scipy")
    .env({"MNIST_DIR": "/data/mnist", "PYTHONPATH": "/opt/proj", "MNIST_BACKEND": BACKEND})
    .add_local_dir("/var/tmp/hebb-mnist/data/mnist", "/data/mnist", copy=True)
    .add_local_dir(str(ROOT / "hebb"), "/opt/proj/hebb", copy=True,
                   ignore=["**/__pycache__", "**/modal_runs", "**/configs"])
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
RESULTS = "/results"


@app.function(image=IMAGE, gpu=GPU, timeout=TIMEOUT_S, volumes={RESULTS: vol},
              max_containers=MAX_CONCURRENCY, **({} if GPU else dict(cpu=8.0, memory=8192)))
def run_unit(run_id: str, unit: dict, cfg: dict) -> str:
    import torch
    assert torch.cuda.is_available() or BACKEND == "cpu", "unit must run on CUDA"
    sys.path.insert(0, "/opt/proj")
    from hebb.mnist.model import Cfg
    from hebb.mnist.train import train
    c = Cfg(**{**cfg.get("defaults", {}), **unit})
    out = Path(RESULTS) / run_id
    res = train(c, out, device="cuda" if BACKEND == "gpu" else "cpu", verbose=True)
    vol.commit()
    return res["name"]


@app.function(image=IMAGE, timeout=24 * 60 * 60, volumes={RESULTS: vol})
def coordinator(run_id: str, cfg: dict) -> dict:
    units = cfg["units"]
    done, failed = [], []
    for unit, res in zip(units, run_unit.map([run_id] * len(units), units, [cfg] * len(units),
                                             return_exceptions=True)):
        (failed if isinstance(res, Exception) else done).append(
            {"unit": unit, "error": repr(res)} if isinstance(res, Exception) else res)
    manifest = {"run_id": run_id, "n_units": len(units), "done": done, "failed": failed}
    p = Path(RESULTS) / run_id / "MANIFEST.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1))
    vol.commit()
    return manifest


def _runs_dir() -> Path:
    d = HERE / "modal_runs"; d.mkdir(exist_ok=True); return d


def cmd_deploy():
    import subprocess
    subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)


def cmd_launch(cfg_path: str):
    cfg = json.loads(Path(cfg_path).read_text())
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + Path(cfg_path).stem
    call = modal.Function.from_name(APP_NAME, "coordinator").spawn(run_id, cfg)
    rec = {"run_id": run_id, "call_id": call.object_id, "cfg": cfg_path,
           "n_units": len(cfg["units"]), "gpu": str(GPU), "launched": time.ctime()}
    (_runs_dir() / f"{run_id}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_status(run_id: str):
    rec = json.loads((_runs_dir() / f"{run_id}.json").read_text())
    call = modal.FunctionCall.from_id(rec["call_id"])
    try:
        manifest = call.get(timeout=0)
    except TimeoutError:
        try:
            n = sum(1 for e in vol.listdir(f"/{run_id}", recursive=True) if e.path.endswith(".npz"))
        except Exception:
            n = 0
        print(f"running: {n}/{rec['n_units']} parts written")
        return
    except Exception as error:
        print(f"coordinator FAILED: {error!r}")
        return
    print(f"finished: {len(manifest['done'])} done, {len(manifest['failed'])} failed")
    for f in manifest["failed"]:
        print("  FAILED", f)


def cmd_fetch(run_id: str, out: str, with_weights: bool = False):
    dest = Path(out) / run_id
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in vol.listdir(f"/{run_id}", recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        target = dest / Path(e.path).relative_to(run_id)
        if "/weights/" in e.path and not with_weights:
            continue
        if target.exists() and target.stat().st_size == e.size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"".join(vol.read_file(e.path)))
        n += 1
    print(f"fetched {n} files to {dest}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "deploy":
        cmd_deploy()
    elif a[0] == "launch":
        cmd_launch(a[1])
    elif a[0] == "status":
        cmd_status(a[1])
    elif a[0] == "fetch":
        cmd_fetch(a[1], a[2] if len(a) > 2 else "/var/tmp/hebb-mnist/runs", "--weights" in a)
    else:
        sys.exit(__doc__)
