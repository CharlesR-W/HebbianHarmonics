"""Checks of hebb/onsite.py against the statements in SPEC.md §1."""
import numpy as np
from hebb import onsite as o


def _setup(seed=0, klo=3.0, khi=0.5, sigma_t=0.7, N=14, B=6, S=64):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((B, B)); Sigma = M @ M.T / B
    net = o.make_net(N, B, rng, s=3, klo=klo, khi=khi, sigma_t=sigma_t)
    W = rng.uniform(0, 0.3, (N, N)) * (rng.random((N, N)) < 0.4); W = np.triu(W, 1); net.W = W + W.T
    return net, o.sample_inputs(Sigma, S, rng), Sigma


def test_equilibrium_is_stationary_and_minimal():
    net, G, _ = _setup()
    U = o.equilibrate(net, G)
    x = U - net.t
    g = U @ o.laplacian(net).T + np.where(x < 0, net.klo, net.khi) * x - G @ net.A.T
    assert np.abs(g).max() < 1e-9
    e0 = o.energy(net, U, G)
    rng = np.random.default_rng(1)
    for _ in range(5):
        assert o.energy(net, U + 1e-3 * rng.standard_normal(U.shape), G) > e0


def test_envelope_identity():
    """d E*/d w_ij = 1/2 E[(u_i-u_j)^2], by central differences on the minimised energy."""
    net, G, _ = _setup()
    D2 = o.dist2(o.equilibrate(net, G))
    for (i, j) in [(0, 1), (2, 7), (5, 11)]:
        h = 1e-5; es = []
        for sgn in (+1, -1):
            net.W[i, j] += sgn * h; net.W[j, i] += sgn * h
            es.append(o.energy(net, o.equilibrate(net, G), G))
            net.W[i, j] -= sgn * h; net.W[j, i] -= sgn * h
        assert abs((es[0] - es[1]) / (2 * h) - 0.5 * D2[i, j]) < 1e-6


def test_linear_case_is_harmonic():
    net, G, _ = _setup(klo=0, khi=0, sigma_t=0)
    H = o.harmonic(net)
    assert np.allclose(H.sum(1), 1) and (H > -1e-12).all()
    assert np.allclose(o.equilibrate(net, G), G @ H.T)


def test_fixed_point_and_two_neuron_bistability():
    net, G, _ = _setup()
    net.W[:] = 0
    r = o.run(net, G, b=0.1, alpha=1.0)
    D2 = o.dist2(r['U']); off = ~np.eye(net.N, dtype=bool)
    assert r['residual'] < 1e-6
    assert np.abs(net.W - np.maximum(0.1 - D2 / 2, 0))[off].max() < 1e-5
    # two neurons on two unit-variance uncorrelated inputs: D2(w) = D0^2/(1+2w)^2, D0^2 = 2
    G2 = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    b, alpha = 0.4, 0.2
    ws = []
    for w0 in (0.0, 3.0):
        n2 = o.Net(A=np.eye(2), t=0, klo=0, khi=0, W=np.array([[0, w0], [w0, 0.0]]))
        o.run(n2, G2, b, alpha, dt=0.05, T=20000)
        ws.append(n2.W[0, 1])
        assert abs(b - alpha * ws[-1] - 1.0 / (1 + 2 * ws[-1]) ** 2) < 1e-6 or ws[-1] == 0
    assert ws[0] == 0 and ws[1] > 1
