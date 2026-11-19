"""Modal CPU fan-out for the C3 cold-start units (hebb/perron2/cold.py). Template:
the modal skill's modal_job.py. A detached coordinator maps over chunks of units; each chunk runs
in one 1-core container and writes one JSON per unit to the Volume. From the repo root:

    PYTHONPATH=. uv run --with modal python hebb/perron2/modal_app.py deploy
    PYTHONPATH=. uv run --with modal python hebb/perron2/modal_app.py launch scratch/out/perron2/cold_cfg.json
    PYTHONPATH=. uv run --with modal python hebb/perron2/modal_app.py status <run_id>
    PYTHONPATH=. uv run --with modal python hebb/perron2/modal_app.py fetch  <run_id> scratch/out/perron2/modal
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

APP_NAME = "crw-hebb-perron2"
CHUNK = 6
TIMEOUT_S = 3 * 60 * 60
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .pip_install("numpy", "scipy")
    .add_local_python_source("hebb")
)
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(f"{APP_NAME}-results", create_if_missing=True)
RESULTS = "/results"
RUNS = Path("scratch/out/perron2/modal_runs")


@app.function(image=IMAGE, cpu=1.0, memory=2048, timeout=TIMEOUT_S, volumes={RESULTS: vol}, max_containers=150)
def run_chunk(run_id: str, chunk: list) -> list:
    from hebb.big import common as cm
    from hebb.perron2 import cold
    out = []
    d = Path(RESULTS) / run_id
    d.mkdir(parents=True, exist_ok=True)
    for u in chunk:
        s = cold.run_unit(u["item"])
        (d / f"{u['name']}.json").write_text(json.dumps(cm.clean(s, 6)))
        vol.commit()
        out.append(u["name"])
    return out


@app.function(image=IMAGE, timeout=24 * 60 * 60, volumes={RESULTS: vol})
def coordinator(run_id: str, cfg: dict) -> dict:
    units = cfg["units"]
    chunks = [units[i:i + CHUNK] for i in range(0, len(units), CHUNK)]
    done, failed = [], []
    for ch, res in zip(chunks, run_chunk.map([run_id] * len(chunks), chunks, return_exceptions=True)):
        if isinstance(res, Exception):
            failed.append(dict(units=[u["name"] for u in ch], error=repr(res)))
        else:
            done += res
    manifest = dict(run_id=run_id, n_units=len(units), done=done, failed=failed)
    vol.reload()
    p = Path(RESULTS) / run_id / "MANIFEST.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1))
    vol.commit()
    return manifest


def main():
    a = sys.argv[1:]
    RUNS.mkdir(parents=True, exist_ok=True)
    if a[0] == "deploy":
        import subprocess
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif a[0] == "launch":
        cfg = json.loads(Path(a[1]).read_text())
        run_id = time.strftime("%Y%m%d-%H%M%S")
        call = modal.Function.from_name(APP_NAME, "coordinator").spawn(run_id, cfg)
        rec = dict(run_id=run_id, call_id=call.object_id, cfg=a[1], n_units=len(cfg["units"]), launched=time.ctime())
        (RUNS / f"{run_id}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1))
    elif a[0] == "status":
        rec = json.loads((RUNS / f"{a[1]}.json").read_text())
        try:
            m = modal.FunctionCall.from_id(rec["call_id"]).get(timeout=0)
            print(f"finished: {len(m['done'])} done, {len(m['failed'])} failed chunks")
            for f in m["failed"]:
                print("  FAILED", f)
        except Exception as e:
            if not isinstance(e, TimeoutError):
                print("coordinator error (units may be complete):", repr(e)[:160])
            try:
                n = sum(1 for e in vol.listdir(f"/{a[1]}") if e.path.endswith(".json"))
            except Exception:
                n = 0
            print(f"running: {n}/{rec['n_units']}")
    elif a[0] == "fetch":
        dest = Path(a[2]) / a[1]
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for e in vol.listdir(f"/{a[1]}"):
            if e.type == modal.volume.FileEntryType.FILE:
                (dest / Path(e.path).name).write_bytes(b"".join(vol.read_file(e.path)))
                n += 1
        print(f"fetched {n} files to {dest}")


if __name__ == "__main__":
    main()
