"""Film writer for site/data/perron2: the site/data/big film format (hebb/big/common.write_film,
site/data/big/README.md) plus a per-frame `labels` array (int16, -1 = dead) and per-frame
`theta`. Written to site/data/perron2/ with its own index.json."""
import json

import numpy as np

from hebb.big import common as cm
from hebb.perron2.parts import OUT

ALIGN = 8


def write_film(meta, coords, H, edges, edge_w, labels, index_entry):
    fid = meta["id"]
    chunks, arrays, off = [], {}, 0

    def add(name, arr, extra):
        nonlocal off
        pad = (-off) % ALIGN
        if pad:
            chunks.append(b"\0" * pad)
            off += pad
        b = np.ascontiguousarray(arr).tobytes()
        arrays[name] = dict(offset=off, dtype=str(arr.dtype.name), shape=list(arr.shape), **extra)
        chunks.append(b)
        off += len(b)

    q, lo, hi = cm._quant_coords(np.asarray(coords, float))
    add("coords", q, dict(lo=lo.tolist(), hi=hi.tolist(), decode="x = lo + (hi - lo) * v / 65535"))
    add("H", np.round(np.clip(H, 0, 1) * 255).astype(np.uint8), dict(decode="h = v / 255"))
    F = len(edges)
    E = max(1, max(len(e) for e in edges))
    ea = np.zeros((F, E, 2), "<u2")
    wa = np.zeros((F, E), np.uint8)
    wmax = max([float(w.max()) for w in edge_w if len(w)] + [1e-300])
    for f, (e, w) in enumerate(zip(edges, edge_w)):
        ea[f, :len(e)] = e
        wa[f, :len(e)] = np.round(255 * np.sqrt(np.clip(w / wmax, 0, 1)))
    add("edges", ea, dict(count=[int(len(e)) for e in edges],
                          note="first count[f] rows of frame f are valid, strongest first"))
    add("edge_w", wa, dict(wmax=wmax, decode="w = wmax * (v / 255)^2"))
    add("labels", np.asarray(labels, "<i2"), dict(decode="cluster label of neuron i at frame f (by size, 0 = largest); -1 = dead"))
    meta = dict(meta, bin=f"{fid}.bin", arrays=arrays)
    OUT.mkdir(parents=True, exist_ok=True)
    blob = b"".join(chunks)
    (OUT / f"{fid}.bin").write_bytes(blob)
    cm.write_json(OUT / f"{fid}.json", meta, sig=5)
    total = (OUT / f"{fid}.bin").stat().st_size + (OUT / f"{fid}.json").stat().st_size
    p = OUT / "index.json"
    idx = json.loads(p.read_text()) if p.exists() else []
    idx = [e for e in idx if e["id"] != fid] + [dict(index_entry, id=fid, file=f"{fid}.json", bytes=total)]
    idx.sort(key=lambda e: e["id"])
    p.write_text(json.dumps(idx, indent=1))
    return total
