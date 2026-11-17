"""Cross-check: the torch model (mean-field mode, float64) reproduces hebb.core exactly."""
import numpy as np
import torch

from hebb import core
from hebb.mnist.model import Cfg, init_weights, step


def main():
    rng = np.random.default_rng(1)
    B, N = 12, 40
    A = rng.standard_normal((B, 3 * B)); Sigma = A @ A.T / (3 * B)
    cfg = Cfg(B=B, N=N, theta=0.15, wmin=1e-3, dt=0.05, seed=3)
    P = core.Params(theta=cfg.theta, lam=cfg.lam, eta=cfg.eta, gamma=cfg.gamma, wmin=cfg.wmin, dt=cfg.dt)
    Wii, Wib, Mii, Mib = core.init_state(N, B, np.random.default_rng(cfg.seed), 1.0, cfg.ii_scale, cfg.ib_scale)
    Tii, Tib = init_weights(cfg, "cpu", torch.float64)
    assert np.allclose(Wii, Tii.numpy()) and np.allclose(Wib, Tib.numpy())
    S = torch.as_tensor(Sigma)
    for _ in range(500):
        Wii, Wib, H = core.step(Wii, Wib, Mii, Mib, Sigma, P)
        Tii, Tib, TH = step(Tii, Tib, cfg, Sigma=S)
    err = max(np.abs(Wii - Tii.numpy()).max(), np.abs(Wib - Tib.numpy()).max(), np.abs(H - TH.numpy()).max())
    print(f"max |core - torch| after 500 steps: {err:.2e}")
    assert err < 1e-9
    print("OK")


if __name__ == "__main__":
    main()
