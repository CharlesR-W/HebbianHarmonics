"""Validate the fast integrators in hebb/big/dist_engine.py against hebb/onsite.py on small cases.

1. lin_run vs onsite.run (linear, N = 30, B = 12): final W, steps, residual.
2. equilibrate_t vs onsite.equilibrate (rectifier klo = 3, khi = 0 and klo = 3, khi = 0.5,
   random W, N = 25): max |U diff|.
3. rect_run vs onsite.run (rectifier, N = 20, B = 6, two b values): final W, steps, residual.

Run: uv run --with torch python -m hebb.big.validate [device]    -> prints and writes
     scratch/out/big/validate_<device>.json
"""
import sys

import numpy as np

from hebb import onsite as o
from hebb import onsite_analysis as oa
from hebb.big import common as cm
from hebb.big import dist_engine as de


def main(device="cpu"):
    out = {}
    # 1. linear
    Sigma = oa.sigma_blocks(12)
    G, _ = oa.banks(Sigma, 0)
    rows = []
    for b in (0.02, 0.08, 0.3):
        net = oa.make_net(30, 12, 0)
        ref = o.run(net, G, b, 1.0, dt=0.5, T=40000, tol=1e-9)
        fast = de.lin_run(net.A, G, b, 1.0, Tmax=40000)
        rows.append(dict(b=b, dW=float(np.abs(net.W - fast["W"]).max()), steps_ref=ref["steps"], steps_fast=fast["steps"],
                         res_ref=ref["residual"], res_fast=fast["residual"]))
    out["linear_run"] = rows
    print("linear", rows, flush=True)
    # 2/3. rectifier
    import torch
    rng = np.random.default_rng(3)
    eq = []
    for khi in (0.0, 0.5):
        net = o.make_net(25, 6, rng, s=3, klo=3.0, khi=khi, sigma_t=0.7)
        Wr = rng.uniform(0, 0.4, (25, 25)) * (rng.random((25, 25)) < 0.4)
        net.W = np.triu(Wr, 1) + np.triu(Wr, 1).T
        Gs = o.sample_inputs(np.eye(6) + 0.3, 200, rng)
        Uref = o.equilibrate(net, Gs)
        f = lambda x: torch.as_tensor(np.asarray(x, float), dtype=torch.float64, device=device)  # noqa: E731
        Ut, info = de.equilibrate_t(f(o.laplacian(net)), f(net.A), f(net.t), f(net.klo), f(net.khi), f(Gs))
        eq.append(dict(khi=khi, maxdiff=float(np.abs(Ut.cpu().numpy() - Uref).max()), gmax=info["gmax"]))
    out["equilibrate"] = eq
    print("equilibrate", eq, flush=True)
    rr = []
    Sigma6 = oa.sigma_power(6)
    G6, _ = oa.banks(Sigma6, 1)
    for b in (0.05, 0.3):
        net = oa.make_net(20, 6, 1, **oa.RECT)
        ref = o.run(net, G6, b, 1.0, dt=0.5, T=40000, tol=1e-9)
        net2 = oa.make_net(20, 6, 1, **oa.RECT)
        fast = de.rect_run(net2.A, net2.t, net2.klo, net2.khi, G6, b, 1.0, Tmax=40000, device=device)
        rr.append(dict(b=b, dW=float(np.abs(net.W - fast["W"]).max()), dU=float(np.abs(ref["U"] - fast["U"]).max()),
                       steps_ref=ref["steps"], steps_fast=fast["steps"], res_ref=ref["residual"], res_fast=fast["residual"]))
    out["rect_run"] = rr
    print("rect", rr, flush=True)
    cm.RAW.mkdir(parents=True, exist_ok=True)
    cm.write_json(cm.RAW / f"validate_{device}.json", out)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cpu")
