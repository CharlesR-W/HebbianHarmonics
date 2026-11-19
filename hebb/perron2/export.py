"""Build site/data/perron2/*.json from the raw outputs in scratch/out/perron2/.

    uv run python -m hebb.perron2.export [c1|ramps|cold|all]
"""
import glob
import json
import sys

import numpy as np

from hebb.big import common as cm
from hebb.perron2 import parts as pp

OUT, RAW = pp.OUT, pp.RAW
SITE_DIAGRAMS = ["hier16", "hier12_322", "irreg14", "randultra16", "hier16_ultravar", "hier16_drd",
                 "hier16_noise0.01_s0", "hier16_noise0.03_s0", "hier16_noise0.1_s0", "block12", "ring16"]


def r3(x):
    return x if (x is None or isinstance(x, str)) else float(f"{x:.3g}")


def c1():
    D = json.loads((RAW / "diagrams_full.json").read_text())["diagrams"]
    edges = json.loads((RAW / "edges.json").read_text())["edges"]
    rnd = json.loads((RAW / "random_ultra.json").read_text())["runs"]
    site = []
    summary = []
    for d in D:
        rows = d["rows"]
        th = [r["theta"] for r in rows]
        n_eqc = sum(r["eq_cliques"] for r in rows)
        n_eqk = sum(r["eq_components"] for r in rows)
        # stable parts that are not cliques: are they at least inside a component / union of cliques?
        extra = sum(len(r["stable_not_clique"]) for r in rows)
        missing = sum(len(r["clique_not_stable"]) for r in rows)
        uncovered = sum(r["n_covered"] < d["B"] for r in rows)
        firsts = []
        for r in rows:
            if not r["eq_cliques"]:
                firsts.append(dict(theta=r["theta"], stable=[p["S"] for p in r["parts"]], cliques=r["cliques"],
                                   stable_not_clique=r["stable_not_clique"], clique_not_stable=r["clique_not_stable"]))
        summary.append(dict(name=d["name"], kind=d["kind"], desc=d["desc"], B=d["B"], n_theta=len(rows),
                            eq_cliques=n_eqc, eq_components=n_eqk, stable_not_clique_total=extra,
                            clique_not_stable_total=missing, theta_with_uncovered_inputs=uncovered,
                            n_stable_range=[min(r["n_stable"] for r in rows), max(r["n_stable"] for r in rows)],
                            mismatches_first=firsts[:3], n_mismatch=len(firsts)))
        if d["name"] in SITE_DIAGRAMS:
            site.append(dict(name=d["name"], kind=d["kind"], desc=d["desc"], B=d["B"],
                             Sigma=np.round(np.array(d["Sigma"]), 4).tolist(), kappa0=r3(d["kappa0"]),
                             theta=th,
                             parts=[[[p["S"], r3(p["c"]), r3(p["margin"]), r3(p["gap"]), [float(f"{x:.2g}") for x in p["h"]]]
                                     for p in r["parts"]]
                                    for r in rows],
                             n_parts_all=[r["n_parts_all"] for r in rows],
                             n_covered=[r["n_covered"] for r in rows],
                             cliques=[None if r["eq_cliques"] else r["cliques"] for r in rows],
                             components=[None if r["eq_components"] else r["components"] for r in rows],
                             eq_cliques=[int(r["eq_cliques"]) for r in rows],
                             eq_components=[int(r["eq_components"]) for r in rows]))
    cm.write_json(OUT / "part_diagrams.json", dict(
        meta=dict(script="hebb/perron2/parts.py", grid="theta_k = (k + 1/2) / 200 * max variance, k = 0..199",
                  part_fields="[support, c, margin, gap, h on the support (sum 1, 2 significant digits)]",
                  note="stable parts only, sorted by decreasing c; cliques/components: maximal cliques and connected "
                       "components of G_theta = {a != b: Sigma_ab > theta} with a positive top eigenvalue of A_SS, "
                       "null where they equal the stable supports (eq_cliques / eq_components = 1)"),
        diagrams=site), sig=4)
    edge_sum = []
    for e in edges:
        if e["family"] == "hier16_drd":
            continue
        edge_sum.append(dict(family=e["family"], level=r3(e["level"]), where=e["where"], n_stable=e["n_stable"],
                             eq_cliques=e["eq_cliques"], stable=e["stable"], cliques=e["cliques"],
                             marginal_examples=[m for m in e["marginal_examples"] if m["gap"] == "inf" or m["gap"] > 0][:2]))
    rs = dict(n_trees=len(rnd), B=[r["B"] for r in rnd],
              unit=dict(n_theta=sum(r["unit"]["n"] for r in rnd), n_fail=sum(len(r["unit"]["fail"]) for r in rnd),
                        fails=[dict(seed=r["seed"], **f) for r in rnd for f in r["unit"]["fail"]][:20]),
              ultra_var=dict(n_theta=sum(r["var"]["n"] for r in rnd), n_fail=sum(len(r["var"]["fail"]) for r in rnd),
                             fails=[dict(seed=r["seed"], **f) for r in rnd for f in r["var"]["fail"]][:20]))
    cm.write_json(OUT / "c1_summary.json", dict(families=summary, edge_cases=edge_sum, random_ultrametric=rs), sig=5)
    return summary, edge_sum, rs




# ----------------------------------------------------------------------------- C2 ramps
LEVELS = (0.8, 0.5, 0.25, 0.05)
RAMP_KEYS = ("theta", "steps", "residual", "converged", "K_major", "n_dead", "sizes", "supports", "perron_cos",
             "matches", "populated", "populated_sizes", "n_stable", "n_populated", "coverage", "frac_on_stable",
             "lateral_between_max", "n_live_lateral", "n_lateral_components")


def transitions(rec):
    out = []
    for a, b in zip(rec[:-1], rec[1:]):
        if sorted(map(tuple, a["supports"])) != sorted(map(tuple, b["supports"])):
            out.append(dict(theta_from=a["theta"], theta_to=b["theta"],
                            supports_from=a["supports"], sizes_from=a["sizes"],
                            supports_to=b["supports"], sizes_to=b["sizes"],
                            levels_between=[L for L in LEVELS if min(a["theta"], b["theta"]) <= L <= max(a["theta"], b["theta"])]))
    return out


def ramps():
    files = sorted(f for f in glob.glob(str(RAW / "ramp_g*.json")) if "down2" not in f)
    out = dict(meta=dict(script="hebb/perron2/ramps.py", sigma="hier16", N=300, levels=LEVELS,
                         grid="80 values linspace(0.03, 0.95); up from a cold start, down from the end of up",
                         kick="relative 1e-3 noise on all above-floor weights at each new theta",
                         stop="projected residual < 1e-7 (cap 40000 steps); converged = residual < 1e-6"),
               runs=[])
    for f in files:
        d = json.loads(open(f).read())
        run = dict(gamma=d["gamma"])
        for dr in ("up", "down"):
            rec = d[dr]
            run[dr] = {k: [r[k] for r in rec] for k in RAMP_KEYS}
            run[dr]["lateral_between"] = [np.round(np.array(r["lateral_between"]), 5).tolist() for r in rec]
            run[dr]["transitions"] = transitions(rec)
            run[dr]["n_not_converged"] = sum(not r["converged"] for r in rec)
        # hysteresis: same theta, populated set up vs down
        up = {r["theta"]: r for r in d["up"]}
        hyst = []
        for r in d["down"]:
            u = up[r["theta"]]
            hyst.append(dict(theta=r["theta"], up=u["populated"], down=r["populated"],
                             up_supports=u["supports"], down_supports=r["supports"],
                             same=sorted(map(tuple, u["supports"])) == sorted(map(tuple, r["supports"])),
                             cov_up=u["coverage"], cov_down=r["coverage"]))
        run["hysteresis"] = sorted(hyst, key=lambda h: h["theta"])
        out["runs"].append(run)
    out["runs"].sort(key=lambda r: -r["gamma"])
    # second down ramp (cold start at 0.822, then down)
    for run in out["runs"]:
        f = RAW / f"ramp_g{run['gamma']}_down2.json"
        if f.exists():
            rec = json.loads(f.read_text())["down2"]
            run["down2"] = {k: [r[k] for r in rec] for k in RAMP_KEYS}
            run["down2"]["lateral_between"] = [np.round(np.array(r["lateral_between"]), 5).tolist() for r in rec]
            run["down2"]["transitions"] = transitions(rec)
            run["down2"]["n_not_converged"] = sum(not r["converged"] for r in rec)
    # split tests (hebb/perron2/floor_test.py)
    tests = []
    for name in ("floor_test.json", "floor_test2.json"):
        f = RAW / name
        if f.exists():
            for c in json.loads(f.read_text())["cases"]:
                tests.append({k: c[k] for k in ("gamma", "omega", "kick", "theta_from", "theta_to", "sizes_before",
                                                "sizes", "supports", "populated", "n_stable", "steps", "residual",
                                                "converged")})
    out["split_tests"] = dict(
        script="hebb/perron2/floor_test.py",
        what="restart from the up-ramp equilibrium at theta_from, set the floor to omega / N (weights at the old "
             "floor moved to the new one), kick (relative, all above-floor weights), integrate at theta_to to "
             "residual < 1e-7 (cap 200000 steps)", cases=tests)
    cm.write_json(OUT / "ramps.json", out, sig=4)
    return out


# ----------------------------------------------------------------------------- C3 cold starts
def cold(*srcs):
    U = [json.loads(open(f).read()) for src in srcs for f in sorted(glob.glob(str(src) + "/*.json"))
         if "MANIFEST" not in f]
    from hebb.perron2 import cold as cd
    S = cd.sigmas()
    table = []
    for sig in S:
        k0 = float(S[sig].mean())
        for g in sorted({u["gamma"] for u in U if u["sigma"] == sig}, reverse=True):
            for th in sorted({u["theta"] for u in U if u["sigma"] == sig}):
                us = [u for u in U if u["sigma"] == sig and u["gamma"] == g and abs(u["theta"] - th) < 1e-9]
                if not us:
                    continue
                c = us[0]["stable_c"]
                order = np.argsort(-np.array(c))
                rank = {tuple(us[0]["stable_parts"][i]): j for j, i in enumerate(order)}
                sets = sorted({json.dumps(sorted(u["populated"])) for u in us})
                table.append(dict(
                    sigma=sig, gamma=g, theta=float(th), kappa0=k0, gamma_kappa0=g * k0, n_seeds=len(us),
                    n_stable=us[0]["n_stable"],
                    n_populated=[u["n_populated"] for u in us], K_major=[u["K_major"] for u in us],
                    coverage_mean=float(np.mean([u["coverage"] for u in us])),
                    frac_on_stable=[u["frac_on_stable"] for u in us],
                    bound_one_cluster=[u["K_major"] == 1 for u in us],
                    n_distinct_populated_sets=len(sets), populated_sets=[json.loads(s) for s in sets],
                    populated_c_ranks=[sorted(rank.get(tuple(p), -1) for p in u["populated"]) for u in us],
                    supports=[u["supports"] for u in us], sizes=[u["sizes"] for u in us],
                    converged=[u["converged"] for u in us], residual_max=max(u["residual"] for u in us),
                    changed_by_kick=[u.get("changed_by_kick") for u in us],
                    pre_kick_n_populated=[u.get("pre_kick", {}).get("n_populated") for u in us]))
    # C4: up ramp vs cold starts at the same theta and gamma (hier16)
    c4 = []
    for f in sorted(glob.glob(str(RAW / "ramp_g*.json"))):
        if "down2" in f:
            continue
        d = json.loads(open(f).read())
        up = {round(r["theta"], 6): r for r in d["up"]}
        for t in table:
            if t["sigma"] != "hier16" or t["gamma"] != d["gamma"] or round(t["theta"], 6) not in up:
                continue
            r = up[round(t["theta"], 6)]
            c4.append(dict(gamma=d["gamma"], theta=t["theta"], n_stable=t["n_stable"],
                           ramp_n_populated=r["n_populated"], ramp_sizes=r["sizes"], ramp_supports=r["supports"],
                           cold_n_populated=t["n_populated"], cold_sizes=t["sizes"],
                           ramp_set_in_cold_sets=sorted(r["populated"]) in t["populated_sets"]))
    cm.write_json(OUT / "cold_starts.json", dict(c4_ramp_vs_cold=c4,
        meta=dict(script="hebb/perron2/cold.py (+ modal_app.py)", N=300, seeds="0-4 (block345: 0-7)", gammas="1, 0.5; hier16 also 0.25, 0.1 (for C4)",
                  grid=cd.GRID, kick_test="after convergence: kick all above-floor weights by (1 + 1e-3 xi) and "
                                          "re-equilibrate (<= 3 times) until clusters stop changing",
                  populated="major cluster (>= 1% of neurons) whose support is a stable part and centroid cosine > 0.98",
                  c_rank="0 = stable part with the largest Perron eigenvalue c"),
        table=table), sig=4)
    print("cold table rows", len(table), "c4 rows", len(c4))
    return table


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w in ("c1", "all"):
        c1()
    if w in ("ramps", "all"):
        ramps()
    if w in ("cold",):
        cold(*sys.argv[2:])
