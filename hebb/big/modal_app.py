"""Modal fan-out for the nonlinear (rectifier) distance-rule units of hebb/big/dist_runs.py.

Template: ~/.claude/skills/modal/modal_job.py. A detached coordinator maps over chunks of units;
each chunk runs in one container (cfg "backend": "cpu" (default; 2 cores, torch float64 on CPU) or
"gpu" (A100, CUDA)) and writes one .npz/.json per unit to the Volume. Commands (from the repo root):

    PYTHONPATH=. uv run --with modal python hebb/big/modal_app.py deploy
    PYTHONPATH=. uv run --with modal python hebb/big/modal_app.py launch <cfg.json>
    PYTHONPATH=. uv run --with modal python hebb/big/modal_app.py status <run_id>
    PYTHONPATH=. uv run --with modal python hebb/big/modal_app.py fetch  <run_id> scratch/out/big/modal

cfg.json: {"units": [{"name": "...", "items": [unit, ...]}, ...]}, or {"units": [{"validate": 1}]}.
Run records go to scratch/out/big/modal_runs/<run_id>.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

APP_NAME = "crw-hebb-big"
GPU = "A100-40GB"
TIMEOUT_S = 2 * 60 * 60
MAX_CONCURRENCY = 40
# CPU fan-out (used for the production runs: the workspace's GPU slots were taken by another
# project's jobs, and a 400 x 400 batched Cholesky is ~1 s per step on one core anyway)
CPU_CORES = 2.0
CPU_MEM_MB = 12288
TIMEOUT_CPU_S = 12 * 60 * 60
MAX_CPU_CONTAINERS = 150
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "scipy")
    .add_local_python_source("hebb")
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(f"{APP_NAME}-results", create_if_missing=True)
RESULTS = "/results"
RUNS = Path("scratch/out/big/modal_runs")   # relative to the repo root (run from there)


@app.function(image=IMAGE, gpu=GPU, timeout=TIMEOUT_S, volumes={RESULTS: vol}, max_containers=MAX_CONCURRENCY)
def run_unit(run_id: str, unit: dict, cfg: dict) -> str:
    import torch

    assert torch.cuda.is_available(), "unit must run on CUDA"
    t0 = time.time()
    out_dir = Path(RESULTS) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if unit.get("validate"):
        from hebb.big import validate
        res = validate.main("cuda")
        (out_dir / "validate_cuda.json").write_text(json.dumps(res, default=float))
        vol.commit()
        return "validate"
    from hebb.big import dist_runs
    recs = []
    for u in unit["items"]:
        rec = dist_runs.run_unit(u, device="cuda", out_dir=str(out_dir))
        recs.append(rec["name"])
        vol.commit()
    (out_dir / f"chunk_{unit['name']}.done").write_text(json.dumps(dict(units=recs, seconds=time.time() - t0)))
    vol.commit()
    return unit["name"]


@app.function(image=IMAGE, cpu=CPU_CORES, memory=CPU_MEM_MB, timeout=TIMEOUT_CPU_S, volumes={RESULTS: vol},
              max_containers=MAX_CPU_CONTAINERS)
def run_unit_cpu(run_id: str, unit: dict, cfg: dict) -> str:
    """CPU fan-out: one unit per container, torch float64 on CPU (same code path, device='cpu')."""
    import torch

    torch.set_num_threads(int(CPU_CORES))
    t0 = time.time()
    out_dir = Path(RESULTS) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    from hebb.big import dist_runs
    recs = []
    for u in unit["items"]:
        rec = dist_runs.run_unit(u, device="cpu", out_dir=str(out_dir))
        recs.append(rec["name"])
        vol.commit()
    (out_dir / f"chunk_{unit['name']}.done").write_text(json.dumps(dict(units=recs, seconds=time.time() - t0)))
    vol.commit()
    return unit["name"]


@app.function(image=IMAGE, timeout=24 * 60 * 60, volumes={RESULTS: vol})
def coordinator(run_id: str, cfg: dict) -> dict:
    units = cfg["units"]
    fn = run_unit_cpu if cfg.get("backend", "cpu") == "cpu" else run_unit
    done, failed = [], []
    for unit, res in zip(units, fn.map([run_id] * len(units), units, [cfg] * len(units),
                                             return_exceptions=True)):
        (failed if isinstance(res, Exception) else done).append(
            {"unit": unit.get("name", "validate"), "error": repr(res)} if isinstance(res, Exception) else res)
    manifest = {"run_id": run_id, "n_units": len(units), "done": done, "failed": failed}
    p = Path(RESULTS) / run_id / "MANIFEST.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1))
    vol.commit()
    return manifest


def cmd_deploy():
    import subprocess
    subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)


def cmd_launch(cfg_path: str):
    cfg = json.loads(Path(cfg_path).read_text())
    run_id = time.strftime("%Y%m%d-%H%M%S")
    call = modal.Function.from_name(APP_NAME, "coordinator").spawn(run_id, cfg)
    rec = {"run_id": run_id, "call_id": call.object_id, "cfg": cfg_path, "n_units": len(cfg["units"]),
           "gpu": GPU, "launched": time.ctime()}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{run_id}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_status(run_id: str):
    rec = json.loads((RUNS / f"{run_id}.json").read_text())
    call = modal.FunctionCall.from_id(rec["call_id"])
    try:
        manifest = call.get(timeout=0)
    except TimeoutError:
        try:
            n = sum(1 for e in vol.listdir(f"/{run_id}", recursive=True) if e.path.endswith(".json"))
        except Exception:
            n = 0
        print(f"running: {n} unit records written ({rec['n_units']} chunks)")
        return
    except Exception as error:
        print(f"coordinator FAILED: {error!r}")
        return
    print(f"finished: {len(manifest['done'])} done, {len(manifest['failed'])} failed")
    for f in manifest["failed"]:
        print("  FAILED", f)


def cmd_fetch(run_id: str, out: str):
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in vol.listdir(f"/{run_id}", recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        target = dest / Path(e.path).relative_to(run_id)
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
        cmd_fetch(a[1], a[2] if len(a) > 2 else "scratch/out/big/modal")
    else:
        sys.exit(__doc__)
