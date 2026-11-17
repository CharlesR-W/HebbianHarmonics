"""Writes hebb/mnist/configs/final.json, the production grid (see docs/mnist.md).

Scaling conventions (decided after the pilots, see docs/mnist.md):
  w_min    = nwmin / N      (floor coupling N*w_min is what matters; default N*w_min = 0.5)
  ii_scale = 0.2 * 512 / N  (lateral init: total lateral degree ~51 independent of N)
  init     'generic' = Wib ~ Exp(mean 0.1);  'heavy' = Wib ~ Gamma(0.1, 1) * 0.1 (dense, heavy-tailed)
"""
import json
from pathlib import Path

PIX_TH = [0.3, 0.45, 0.6, 0.75, 0.9, 1.1]
RFF_TH = [0.05, 0.1, 0.15, 0.2, 0.3]
INITS = {"generic": 1.0, "heavy": 0.1}


def unit(lift, B, N, theta, init, seed=0, mode="sgd", nwmin=0.5, bw=1.0):
    return dict(lift=lift, B=B, N=N, theta=theta, ib_shape=INITS[init], seed=seed, mode=mode,
                wmin=nwmin / N, ii_scale=0.2 * 512 / N, bw_mult=bw)


U = []
for N in (512, 2048):
    for init in INITS:
        U += [unit("pixels", 196, N, th, init) for th in PIX_TH]
for init in INITS:
    U += [unit("pixels", 196, 2048, th, init, seed=1) for th in PIX_TH]
U += [unit("pixels", 196, N, th, "heavy", mode="mf") for N in (512, 2048) for th in PIX_TH]
U += [unit("pixels", 196, 2048, th, "heavy", nwmin=nw) for nw in (0.05, 5.0) for th in (0.6, 0.75, 0.9)]
for init in INITS:
    U += [unit("rff", 64, 512, th, init) for th in RFF_TH]
    U += [unit("rff", 256, 512, th, init) for th in RFF_TH]
    U += [unit("rff", 256, 2048, th, init) for th in RFF_TH]
    U += [unit("rff", 1024, 2048, th, init) for th in RFF_TH]
U += [unit("rff", 256, 2048, th, "heavy", bw=bw) for bw in (0.5, 2.0) for th in RFF_TH]
U += [unit("rff", 1024, 2048, th, "heavy", seed=1) for th in RFF_TH]
U += [unit("rff", 256, 2048, th, "heavy", mode="mf") for th in RFF_TH]
cfg = dict(defaults=dict(steps=3000, n_snap=36, n_sub=512, cg_iters=300, dt=0.1, batch=256), units=U)
p = Path(__file__).parent / "configs" / "final.json"
p.write_text(json.dumps(cfg, indent=1))
print(len(U), "units ->", p)
