"""Write site/data/sweeps/index.json listing every sweep JSON with a one-line description."""
import json

from hebb.sweeps.common import OUT

DESC = {
    "parts": "Q1: exact stable parts (thresholded Perron vectors of Sigma - theta 11^T) per covariance and theta, vs. clusters the network populates (gamma 0/1, wmin 1e-5/1e-3, 4 seeds); eigenvector comparisons.",
    "phase": "Phase diagrams: number of populated clusters K over theta x gamma, theta x wmin, theta x ii_scale, and N=60/240 (block12, hier8; seeds 0-2), with kappa0 for the predicted collective boundary theta = gamma*kappa0.",
    "alloc_limit": "Allocation in the gamma=0, N->inf limit on Sigma=diag(v): fraction per input vs theta and init distribution; exponent beta in n ∝ v^beta.",
    "alloc_ingredients": "Which ingredient changes the allocation exponent (gamma=0 limit): init log-weight spread, nonlinear weight-only regularizers, shared minibatch noise over time.",
    "alloc": "Planted-family allocation experiment (network N=150, 8 paired seeds): vary one block's variance or size; fraction of neurons in that block, missing modes, fitted beta; plus descriptive size-vs-energy scatter on powerlaw_pos12 and ring16.",
    "diversity": "Within-cluster diversity (L1 spread, 1 - voltage correlation, participation ratio) vs wmin, gamma, minibatch size, N, and (labelled comparison) sparse mask p.",
    "hysteresis": "Slow theta sweeps up and down with state carried over (block12, hier8; gamma 1 and 0): K, dead fraction, mean energy along the loop.",
    "dynamics": "Convergence time vs theta; mean-field vs minibatch (batch 8-512) on block12; init grid ii_scale x ib_scale.",
    "readout": "Weight-space readout of Sigma across theta (4 covariances, 2 seeds) in the three SPEC §7.3 settings, by edge family, with baselines; plus readout along training.",
    "tours": "Verified explorer presets (chapters 1, 2, 4): exact parameters and per-seed outcomes.",
}


def main():
    items = []
    total = 0
    for p in sorted(OUT.glob("*.json")):
        if p.name == "index.json":
            continue
        d = json.loads(p.read_text())
        size = p.stat().st_size
        total += size
        items.append(dict(file=p.name, bytes=size, description=DESC.get(p.stem, d.get("meta", {}).get("description", "")),
                          script=d.get("meta", {}).get("script")))
    idx = dict(meta=dict(description="Index of sweep outputs (hebb/sweeps/*.py; regenerate with hebb/sweeps/run_all.sh). "
                                     "Findings: docs/findings.md.", total_bytes=total), files=items)
    (OUT / "index.json").write_text(json.dumps(idx, indent=1))
    print(f"{len(items)} files, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
