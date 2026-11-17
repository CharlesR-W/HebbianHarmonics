"""Print the markdown tables used in docs/mnist.md from fetched run JSONs.

    PYTHONPATH=. python -m hebb.mnist.tables /var/tmp/hebb-mnist/runs/<cpu_run> /var/tmp/hebb-mnist/runs/<gpu_run>
"""
import glob
import json
import sys

import numpy as np


def load(dirs):
    runs = {}
    for d in dirs:
        for f in sorted(glob.glob(d + "/*.json")):
            if "MANIFEST" in f:
                continue
            r = json.load(open(f))
            runs[r["name"]] = r
    return runs


def fmt(x, n=2):
    return "–" if x is None else f"{x:.{n}f}"


def main():
    runs = load(sys.argv[1:])
    print(f"{len(runs)} runs; GPU/CPU seconds total: {sum(r['timing']['total_s'] for r in runs.values()):.0f}")
    print("\n### Cluster structure vs theta (sgd, N*w_min = 0.5, seed 0; live = clusters excluding the dead pool)\n")
    print("| lift | N | init | theta | pct | K (tol .05/.1/.2) | live (major) | dead | live sizes (top 6) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, r in sorted(runs.items(), key=lambda kv: (kv[1]["cfg"]["lift"], kv[1]["cfg"]["B"], kv[1]["cfg"]["bw_mult"], kv[1]["cfg"]["N"], kv[1]["cfg"]["ib_shape"], kv[1]["cfg"]["theta"])):
        c = r["cfg"]; f = r["final"]
        if c["mode"] != "sgd" or c["seed"] != 0 or abs(c["wmin"] * c["N"] - 0.5) > 1e-6:
            continue
        S = None
        live = sorted([s for k, s in enumerate(r["cluster_sizes"]) if k not in f.get("dead_clusters", [])], reverse=True)[:6]
        pct = (np.array([]) if True else None)
        q = r["sigma_stats"]
        # percentile of theta among off-diagonal entries (interpolated from stored quantiles)
        pc = float(np.interp(c["theta"], q["value"], q["q"])) if c["theta"] <= q["value"][-1] else 1.0
        lift = c["lift"] if c["lift"] == "pixels" else f"rff{c['B']}" + ("" if c["bw_mult"] == 1 else f" bw×{c['bw_mult']:g}")
        init = "generic" if c["ib_shape"] == 1 else "heavy"
        print(f"| {lift} | {c['N']} | {init} | {c['theta']:g} | {pc:.3f} | {f['n_clusters_tol05']}/{f['n_clusters']}/{f['n_clusters_tol20']} | "
              f"{f['n_live_clusters']} ({f['n_live_major']}) | {f['n_dead']} | {live} |")
    print("\n### Readout (final, time-averaged weights) vs baselines; contrast-subspace score cos / BW, unknown-scale cos\n")
    print("| run | live K | LS both | LS I-I | LS I-B | naive I-B | untrained | identity | diag | rank-K trunc | cos on support (n sup) | rank both / dim |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in sorted(runs.items()):
        c = r["cfg"]; f = r["final"]
        if c["seed"] != 0 or f["n_live_clusters"] == 0:
            continue
        ro, b = r["readout"], r["baselines"]
        cell = lambda d: f"{d['contrast_cos']:.2f} / {d['contrast_bw']:.2f}"
        mr = r["measurement_rank"]
        print(f"| {n} | {f['n_live_clusters']} | {cell(ro['ls_both'])} | {cell(ro['ls_ii'])} | {cell(ro['ls_ib'])} | {cell(ro['naive_ib'])} | "
              f"{cell(ro['init_ls_both'])} | {cell(b['identity'])} | {cell(b['diag'])} | {cell(b['rank_k_truncation'])} | – | {mr['both']['rank']} / {mr['dim_sym']} |")


if __name__ == "__main__":
    main()
