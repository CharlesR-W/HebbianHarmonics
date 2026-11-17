# `engine.js` — API reference for chapter authors

`site/js/engine.js` is a pure ES module (no DOM, no dependencies) that simulates the
model of SPEC §1 and computes the readouts of SPEC §3/§7. `hebb/core.py` mirrors it
function by function (snake_case names). `tests/test_engine_parity.py` checks that
the two agree.

```js
import * as E from './engine.js';
```

## Conventions

- **Matrices** are `Float64Array`s in row-major order, and their shapes are passed
  explicitly: `H[i*B + b]` is entry (i, b) of the N×B matrix H. Eigenvectors follow
  numpy's convention: `vectors[i*n + k]` is component i of eigenvector k, and
  eigenvalues are sorted in **descending** order.
- **State** (`createState` / `loadState`): `{N, B, Wii (N×N), Wib (N×B), Mii, Mib, H, t, resid, …}`.
  `Wii` is symmetric with a zero diagonal. The masks `Mii`/`Mib` are 0/1 and mark
  potential synapses. A synapse that isn't potential has weight exactly 0, not wmin.
  The remaining fields are internal work buffers.
- **Params**: `{theta, lam, eta, gamma, wmin, dt, nonlin}`, with defaults from
  `defaultParams()` = `{theta: 0.3, lam: 1, eta: 1, gamma: 1, wmin: 1e-3, dt: 0.05}`
  and `nonlin` = `'linear'`. Here `w0 = -eta*theta`, which both edge types share;
  `eta_IB = eta` and `eta_II = gamma*eta`.
- **Fixed point** on active edges (SPEC §1/§7.1):
  - I–B: `(HΣ)_ib = θ + (λ/η) w_ib`.
  - I–I: `γ h_iᵀΣh_j = θ + (λ/η) w_ij`, so the lateral threshold is θ/γ.
  - With γ = 0 there is no lateral plasticity, and the I–I equations carry no
    information about Σ. Nothing in the engine divides by γ.
- **Units**: weights are in units of η/λ times covariance. `resid` is a drift dw/dt;
  divide it by η to get covariance units.

## Model

| export | signature → result | notes |
|---|---|---|
| `defaultParams()` | → params | |
| `createState({N,B,seed=0,p=1,iiScale=0,ibScale=0.2})` | → state | Same recipe as `core.init_state`, using mulberry32. `core.create_state` reproduces it bit for bit. |
| `loadState({N,B,Wii,Wib,Mii,Mib})` | → state | Copies the arrays, e.g. to start from a state Python built. |
| `cloneState(s)` | → state | |
| `harmonic(s)` | → `s.H` | H for the current weights, without clipping. |
| `harmonicFrom(Wii,Wib,N,B)` | → H (N×B) | Throws if L_II is not SPD. |
| `step(s, Σ, P)` | → `s.H` | One Euler step, arithmetically identical to `core.step` (`core.step_nl` when `nonlin='relu'`). The returned `s.H` is the H of the weights **before** the update. Sets `s.resid`, `s.residII`, `s.residIB` and increments `s.t`. |
| `advance(s, Σ, P, n)` | → `s.H` | n steps. |
| `stationarity(s, Σ, P)` | → `{r, rII, rIB}` | Projected residual: max \|dw/dt\| over edges, where floor edges with negative drift count as 0. Doesn't modify the weights; overwrites `s.H`. Claim equilibrium only when `r` is small (say below 1e-6). |
| `makeSampler(Σ, B, seed)` | → `{draw(G, batch)}` | Zero-mean Gaussian draws via the eigen-factor of Σ; rank-deficient Σ is fine. |
| `sampleCov(G, batch, B, out?, center=false)` | → B×B | `GᵀG/batch` (known zero mean, works with batch = 1). With `center`, uses the batch-mean-removed covariance divided by (batch−1). |
| `stepSamples(s, G, batch, P, center=false)` | → `s.H` | Step driven by an explicit minibatch G (batch×B). |
| `stepStochastic(s, sampler, P, batch=1, center=false)` | → `s.H` | Draws a minibatch, then calls `stepSamples`. Allocation-free after the first call. |
| `dtAdvice(Σ, B, P)` | → `{dtMax, wBound}` | See the dt section below. |

### Rectified-rate mode (`P.nonlin = 'relu'`)

Kirchhoff stays linear (`u = Hg`, so the embeddings don't change). What changes is
plasticity: it sees rates `r = relu(u − E[u])` and boundary rates `relu(g_b − E[g_b])`.

- **Mean field** uses the arc-cosine kernel on the voltage covariance. With
  `v_i = h_iᵀΣh_i`, `ρ = c/√(v_i v_j)` and `φ = acos ρ`:
  `Cov(r_i, r_j) = √(v_i v_j)/(2π)·(sin φ + (π−φ)cos φ − 1)`. For I–B edges,
  `v_b = Σ_bb`.
- **Stochastic** (`stepSamples`/`stepStochastic`) computes the rates from the sampled
  voltages and uses their minibatch covariance, centred on the batch mean of the
  rates. This needs **batch ≥ 2**; batch 1 throws an error.

| export | → | notes |
|---|---|---|
| `dualRelu(vi, vj, c)` | Cov of the relus | Returns 0 if `vi·vj = 0`. |
| `dualReluInv(vi, vj, k)` | c | Inverse in the covariance argument. It is monotone. The result is clamped to `\|c\| ≤ √(vi vj)`. |
| `reluCorr(ρ)`, `reluCorrInv(t)` | | The normalised map `k(ρ) = (√(1−ρ²) + (π−acos ρ)ρ − 1)/(2π)`, which takes [−1,1] onto [−1/2π, (π−1)/2π] with slope 1/4 at 0. The inverse uses Newton's method with a bisection safeguard. It is ill-conditioned near ρ = −1, where the slope goes to 0. |

## Covariance generators

All generators are deterministic given their arguments and seed. `core.py` has the
same algorithms; the parity test holds them to 1e-12 relative.

| export | → | notes |
|---|---|---|
| `cov3({variances:[v0,v1,v2], corr:[r01,r02,r12]})` | 3×3 | An infeasible correlation matrix is clamped to PSD, keeping its unit diagonal. |
| `covBlocks({sizes, within, between, variances})` | `{Sigma, B, labels}` | `within` and `variances` can be a scalar or a per-block array. Negative `between` is allowed: this gives the signed-covariance preset of SPEC §7.2. |
| `covHierarchical({branching:[3,4], corr:[c_root, …, c_siblings], variance})` | `{Sigma, B, labels}` | `labels` is the top-level group. |
| `covPowerLaw({B, alpha, seed, nonneg=false, normalize=true})` | B×B | Random orthogonal basis × `k^-alpha`. `nonneg` adds the smallest `c·11ᵀ` that makes every correlation ≥ 0; note that this is exactly the direction the contrast score removes. |
| `covWishart({B, dof=B, seed, normalize=true})` | B×B | Rank is min(dof, B). |
| `clampPSD`, `toCorrelation`, `randomOrthogonal(B, rng)`, `makeRng(seed)` | | `makeRng` gives `{uniform, normal, exponential, int}`. |

`normalize` sets the mean variance to 1.

## Analysis

| export | → | notes |
|---|---|---|
| `clustersAuto(H,N,B,{minSep=.05,minGap=3})` | `{labels, K, tol, gap, robust, heights}` | **The headline cluster definition**; see below. |
| `clustersL1(H,N,B,tol=.1)` | `{labels, K}` | Single linkage in L1 at a fixed tolerance. Order-independent. |
| `clusterCountCurve(H,N,B,tols)` | K per tol | For plotting how robust the count is to tolerance. |
| `clustersLeader(H,N,B,tol)` | `{labels,K,centroids}` | Mirrors `core.clusters`. Order-dependent, so use it only for a quick look. |
| `clustersSynapse(Wii,Mii,N,wmin)` | `{labels,K}` | Components of the active lateral graph. Early in training every neuron is a singleton. |
| `mstL1(H,N,B)` | `{parent, weight}` | |
| `clusterStats(H,N,B,labels,K,Σ)` | `{sizes, parts (K×B), energy, spread}` | `part` is the member mean of h (the cluster's representative); `energy = partᵀΣpart`; `spread` is the mean L1 distance from members to the part. |
| `relabel(prevLabels, labels, N)` | labels | Greedy maximum-overlap matching, so colours stay stable across frames. JS only. |
| `orderByCluster(labels,N)` | permutation | For sorted similarity matrices. |
| `effectiveRank(H,N,B)` | number | `‖H‖_F⁴/‖HᵀH‖_F²`, the participation ratio of the squared singular values. K orthogonal clusters give K. |
| `similarity(H,N,B,Σ)` | N×N | `HΣHᵀ`, the voltage covariance. |
| `pca3(H,N,B,prev?)` | `{coords (N×3), basis (B×3), mean, variance}` | **Pass the previous frame's result as `prev`.** The basis is then Procrustes-rotated within the new top-3 PCA subspace to match it, so the view never flips or jumps. Without `prev`, each axis is signed so that its largest entry is positive. |
| `projectPCA(pca, X, n, B)` | n×3 | Projects other points, e.g. the vertices `e_b`. |

**Why `clustersAuto` is the default.** It is single linkage in L1. The merge heights
are the weights of the minimum spanning tree, and the cut goes in the largest
multiplicative gap among heights ≥ `minSep`. This choice has three properties:

- It is order-independent.
- It needs no tolerance knob, and `gap` reports how clear-cut the count is. At a
  fixed point, within-cluster spread is O(wmin) while clusters are O(1) apart, so the
  gap is huge.
- `robust: false` (gap < 3) honestly says "not clustered yet"; the count then falls
  back to `tol = 0.1`.

Masked nets (p < 1) show genuine sub-structure inside clusters, at L1 ≈ 0.05–0.4. To
display it, use `clusterCountCurve`.

## Weight-space readout (SPEC §3, §7.3)

| export | → | notes |
|---|---|---|
| `readoutState(s, P, opts)` | result | Computes H from the weights, then calls `readoutLS`. |
| `readoutLS(H,Wii,Wib,Mii,Mib,N,B,opts)` | `{C, theta, iters, relResidual, rmsII, rmsIB, mII, mIB, alpha, negMass, warnings}` | Masked ridge least squares over symmetric C on active edges (w > wmin(1+eps)), solved by CG on the normal equations. |
| `readoutEquations(...)` / `solveEquations(H,N,B,eq,opts)` | | The two halves of `readoutLS`, for custom equations. |
| `readoutTransient(prev, next, P, opts)` | result | Derivative-aware readout from two consecutive Euler snapshots, using `rhs += (dw/dt)/η`. It is exact away from equilibrium. |
| `readoutRelu(H,…,opts)` | result + `{outer, change, rmsRateII, rmsRateIB}` | Readout for the `nonlin = 'relu'` model; see below. θ must be known. |
| `readoutNaive(H,Wii,Mii,N,B,{theta,lam,eta,gamma,ridge})` | B×B or null | The note's `C = H⁺RH⁺ᵀ`. R has the unobserved diagonal filled with the row maximum. Returns null when γ = 0. |
| `measurementSpectrum(H, eq, N, B)` | `{ii, ib, both: {rank, cond, eig}, p}` | Rank and conditioning of the measurement operator on Sym(B), per edge family. The cost is O(m p²), so use it live only for B ≲ 20 and use `core.measurement_spectrum` offline. Families have different scales, so the numerical rank of `both` can be lower than that of one family on its own. |

`readoutLS` options:

| option | values | meaning |
|---|---|---|
| `theta, lam, eta, gamma, wmin` | | Model parameters, assumed known. |
| `use` | `'both'` \| `'ii'` \| `'ib'` | Which edge families to use. |
| `thetaMode` | `'known'` | Use the given θ. |
| | `'fit'` | θ is an extra unknown. It is separately identifiable only if γ ≠ 1, and it is unreliable on collapsed states. |
| | `'project'` | Assume θ = 0 and put the ridge only on `PCP`, so the 1-directions absorb θ. Returns `PCP`. This is exact for γ = 1 and is **the recommended mode for unknown θ**. |
| `ridge` | default 1e-6 | Relative to the mean eigenvalue of the normal operator. |
| `tol` | default 1e-10 | CG tolerance. |
| `maxIter` | default 2000 | Maximum CG iterations. |
| `x0` | B×B | Warm start. Pass the previous C in live loops. |
| `psd` | | Clip negative eigenvalues; the clipped mass is returned as `negMass`. |
| `eps` | | Margin for counting an edge as active. |
| `weightII`, `weightIB` | | Per-family weights. |

### How `readoutRelu` works

The variances `v_i = h_iᵀCh_i` are **not observed**: there are no self-edges. So
`readoutRelu` alternates between three steps:

1. Compute v from the current C.
2. Invert the dual activation on each edge.
3. Solve the resulting linear least-squares problem, warm-started from C.

It starts from the linear readout of 3×rhs. On synthetic exact equations it recovers
the planted C (this is tested). I recommend it over joint nonlinear least squares
because each step reuses the tested linear solver. Report `change` and `rmsRate*`
alongside the result.

## Metrics (SPEC §3, §7.4)

`compareCov(Chat, Σ, B, {setting, k=3})` returns
`{scale, cosine, relErr, bw, negMass, negFrac, subspace, eigHat, eigTrue}`.

The three settings must be kept apart (§7.3):

- `'known'`: no projection, and the scale is fixed at 1.
- `'scale'`: the best scale is fitted.
- `'contrast'`: P·P is applied to both matrices, with `P = I − 11ᵀ/B`, and the best
  scale is fitted. Label this the "contrast-subspace score". It also discards `1aᵀ`
  terms.

The fields:

| field | definition |
|---|---|
| `cosine` | Frobenius cosine. |
| `relErr` | `‖sA − S‖_F/‖S‖_F`. |
| `bw` | `BW(sA₊, S)²/tr S`. A₊ is A with negative eigenvalues clipped, and the clipped mass is reported in `negMass`/`negFrac`. Under `'scale'`/`'contrast'`, s minimises the distance, which gives `1 − F²/(tr A₊ tr S)`. An all-zero A₊ gives 1. |
| `subspace` | `‖U_kᵀV_k‖_F²/k`, the mean squared cosine of the principal angles. |

`baselines(Σ, B, k)` returns `{identity, diag, rankK}`. The "untrained network"
baseline is `readoutState(initialState, P)`.

Other helpers: `project1(C, B)` gives `PCP`; `psdProject(A, n)` returns the clipped
matrix with `.negMass`.

Caveat on BW: BW heavily penalises rank-deficient estimates. On a 12-input block Σ,
the rank-3 truncation of the true Σ scores `bw` = 0.52 under the contrast setting,
while the identity scores 0.12. Show cosine and subspace overlap next to BW, never
BW alone.

## Linear algebra

| export | notes |
|---|---|
| `cholesky(A,n,out?)` | Lower triangle; returns null if A is not SPD. |
| `cholSolve(L,n,R,m,X?)` | |
| `eigh(A,n)` | Householder plus QL, accurate to about 1e-14. Takes 4 ms at n = 64 on the laptop, and about 100–900 ms at n = 256. |
| `symFn(A,n,f)` | |
| `sqrtm(A,n)` | |
| `pinvSym(A,n,rcond)` | |
| `psdProject(A,n)` | |
| `matmul(A,B,m,k,n,out?)` | |
| `matmulAtB(A,B,k,m,n,out?)` | |
| `matmulABt(A,B,m,k,n,out?)` | |
| `transpose(A,m,n,out?)` | |

## Usage: a live loop with readouts

```js
import * as E from './engine.js';
const B = 24, N = 150;
const { Sigma, labels: planted } = E.covBlocks({ sizes: [8, 8, 8], within: 0.5, between: 0.05 });
const P = { ...E.defaultParams(), theta: 0.2, dt: 0.2 };
const s = E.createState({ N, B, seed: 1, p: 0.5 });
let pca = null, colours = null, Chat = null, frame = 0;
function tick() {
  E.advance(s, Sigma, P, 5);                        // ~5 steps per frame at 60 fps
  const H = E.harmonic(s);                          // H of the current weights
  pca = E.pca3(H, N, B, pca);                       // stable 3D view
  const cl = E.clustersAuto(H, N, B);
  colours = E.relabel(colours, cl.labels, N);       // stable cluster colours
  const st = E.clusterStats(H, N, B, cl.labels, cl.K, Sigma);
  draw(pca.coords, colours, st, s.resid / P.eta);   // resid = stationarity, covariance units
  if (++frame % 30 === 0) {                         // readout twice a second, warm-started
    const r = E.readoutState(s, P, { x0: Chat, maxIter: 100 });
    Chat = r.C;
    const m = E.compareCov(Chat, Sigma, B, { setting: 'known' });
    showReadout(m, s.resid < 1e-6 ? '' : 'equilibrium assumption violated');
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
```

## Performance

Measured in node 26 on the user's laptop (12 threads; V8 is the same engine as
Chrome), fully dense. Figures are steps per second.

| N, B | mean-field `step` | `relu` | stochastic, batch 32 |
|---|---|---|---|
| 100, 16 | 1335 | 963 | 1311 |
| 100, 64 | 467 | 413 | 401 |
| 200, 16 | 281 | 230 | 256 |
| 200, 64 | **111** | 105 | 115 |

Per-frame analysis costs at N = 200, B = 64:

| call | time |
|---|---|
| `clustersAuto` | 2.3 ms |
| `pca3` | 4 ms |
| `readoutState`, cold | 30–300 ms (1–32 CG iterations) |
| `readoutState`, warm-started | ≈ 30 ms |

### Cost of one step

One step costs:

- a Cholesky factorisation (N³/6),
- two triangular solves (N²B),
- HΣ (NB²),
- and HΣHᵀ, of which only the upper triangle is computed (N²B/2).

There is no generic N³ matmul in the step, and nothing is allocated in the hot loop.
p < 1 skips the dot products for masked pairs.

### dt and stability

The decay term alone is monotone for `dt·λ < 1`. Weights stay below
`max(w(0), max(1,γ)·η(max_b Σ_bb − θ)/λ)`, because `Cov ≤ max Σ_bb` for convex
combinations.

Measured on a 12-input block Σ:

| `dt·λ` | outcome |
|---|---|
| ≤ 1.2 | Same attractor, converges to 1e-15. |
| 1.6 | A different attractor (K = 1 instead of 3). |
| 1.9 | Oscillates. |
| 2.1 | Diverges. |

Use `dt·λ ≤ 0.2` for live sims: 4× faster than the default 0.05, with the same
fixed points. Use 0.05 when the transient path itself is the subject.

### Numerical notes

- **L_II conditioning.** L_II stays SPD as long as wmin > 0. At trained states its
  condition number was 40–90, for both wmin = 1e-3 and 1e-6. Consider a neuron whose
  weights all sit at the floor: its diagonal is wmin·(N−1+B), so the conditioning
  degrades like max deg/(wmin·B). That is harmless in float64 down to wmin ≈ 1e-10.
  However, the neuron's h is then set by the floor (it sits near the centroid of its
  neighbours), not by the data, and `clustersAuto` shows it as a singleton.
- **BW precision.** BW takes square roots of the eigenvalues of a singular matrix
  (P S P has rank B−1), so it is accurate only to about 1e-8 absolute. Every other
  metric is accurate to about 1e-12.
