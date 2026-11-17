# `onsite_engine.js` — API reference for chapter authors (v2 model)

`site/js/onsite_engine.js` is a pure ES module (no DOM, no imports) that simulates the
v2 model of SPEC §1 live in the page. `hebb/onsite.py` is the numpy reference.
`tests/test_onsite_parity.py` (driven by `tests/onsite_runner.mjs`) checks that the two
agree, and `tests/onsite_bench.mjs` measures speed.

```js
import * as O from './onsite_engine.js';
```

## Model

    E_W(u; g) = 1/2 Σ_{i<j} w_ij (u_i − u_j)² + 1/2 Σ_{i,b} a_ib (u_i − g_b)² + Σ_i V_i(u_i)
    V_i(u)    = klo_i/2 (u − t_i)_−² + khi_i/2 (u − t_i)_+²
    node eq.  d_i u_i + V_i'(u_i) = Σ_j w_ij u_j + Σ_b a_ib g_b,    d_i = Σ_j w_ij + Σ_b a_ib
    rule      dw_ij/dt = b − α w_ij − ½ D²_ij,  D²_ij = mean_s (u_i − u_j)²,  projected at w ≥ 0
    fixed pt  w_ij = (b − D²_ij/2)_+ / α

Expectations are plain (uncentred) means over the S rows of the input bank G, exactly
as in the learning objective. Population quantities (`clusterStats` d_eff,
`populationModes`) centre over the bank, as the reference does.

## Conventions

- **Arrays.** Every matrix is a `Float64Array` stored row-major, and its shape is passed
  explicitly. The layouts are the reference's:

  | array | shape | entry |
  |---|---|---|
  | `A` | N×B | `A[i*B + b]` |
  | `W` | N×N | `W[i*N + j]` (symmetric, zero diagonal) |
  | `G` | S×B | `G[s*B + b]` (one input sample per row) |
  | `U` | S×N | `U[s*N + i]` (response of neuron i to sample s) |
  | `D2`, `dW` | N×N | |
  | `H` | N×B | |

- **Net.** A net is the plain object `{N, B, A, t, klo, khi, W}`, where `t`, `klo` and
  `khi` are length-N `Float64Array`s. Build one with `makeNet` (random) or `makeNetFrom`
  (from given arrays). In `makeNetFrom`, scalars broadcast to length N and arrays are
  copied.
- **Eigenvectors** follow numpy: `vectors[i*n + k]` is component i of eigenvector k, and
  eigenvalues are sorted in **descending** order.
- **RNG.** `makeRng(seed)` is mulberry32, bit-identical to `engine.js` and
  `hebb.core.JsRng`. Every function that takes `rng` also accepts an integer seed. The
  stream is not numpy's, so a page will not reproduce a Python sweep sample for sample.
  To match a Python result exactly, export its arrays (A, t, klo, khi, W, G) and load
  them with `makeNetFrom`.

## Equilibrium solver (the design choice)

The node equations are piecewise linear. On a fixed **activity pattern** (which neurons
sit at or above threshold), they reduce to the SPD linear system
`(L + diag k_P) u = R + k_P t`, where `L = diag(d) − W` and `k_P` is `khi` above
threshold and `klo` below it.

The reference applies semismooth Newton to each sample separately: an N×N Cholesky per
sample per iteration. At N = 80 and S = 256 that is too slow for a live loop. Measured on
this VPS, the line-by-line port (`exact: true`) manages 52 steps/s at N = 40 and 4.5
steps/s at N = 80.

`equilibrate` (the default, "fast") runs the **same semismooth Newton iteration**, made
cheaper in four ways. None of them introduces any approximation:

1. **Components.** Coupling components of `W ≠ 0` decouple exactly, so each one is solved
   on its own. An isolated neuron has a closed form.
2. **Shared factors.** Within a component, the samples that share an activity pattern
   share one factorisation.
3. **Rank-1 updates.** The distinct patterns are visited in a greedy nearest-neighbour
   (Hamming) order. Moving to the next pattern means a few rank-1 Cholesky
   updates/downdates, at O(n²) each, instead of a fresh O(n³/3) factor. A fresh factor is
   taken when more than n/4 neurons flip, after 2n accumulated updates, or if a
   downdate would lose positive definiteness.
4. **Warm start.** Each solve starts from the previous equilibrium, which is also what
   the live loop does. After a small `dt·dW` almost every sample's pattern is already
   right, and Newton stops after one solve.

**Acceptance (the stated tolerance).** A sample is accepted when the solution has the same
pattern it was solved with. That solution is then the exact equilibrium, up to
roundoff. A sample is also accepted when `max_i |dE/du_i| ≤ tol`, with `tol = 1e-10` by
default, in units of current. Samples whose pattern changed take another Newton step,
using the reference's backtracking safeguard on `max|dE/du|`. That safeguard matters only
when concave and convex neurons are mixed (some `klo < khi`, others `klo > khi`).

**Reported residual.** Every call returns `residual`: the achieved `max |dE/du|` over the
whole bank, computed after the solve. Observed values:

- 1e-15 to 5e-13 in the benchmarks (it grows with the size of W).
- At most 1.2e-12 over 300 random stress configurations: slopes up to 100, conductances
  up to 20, concave/convex mixes, and warm starts from a different W.

In those stress configurations, the fast and exact paths differed by at most 2e-13 in U.

**Slow path for tests.** `equilibrate(net, G, S, {exact: true})` (or
`equilibrateExact`) is the line-by-line port of `hebb.onsite.equilibrate`: per-sample
dense Newton with the same cold start and the same backtracking.

The fast solver is *not* a relaxation. Gauss–Seidel on the node equations (the literal
fast neuronal dynamics) was rejected because its convergence rate degrades as clusters
fuse (cond(L) grows with b/α). It could also only ever be exact "to a tolerance".

## API

### PRNG, linear algebra

| export | signature → result | notes |
|---|---|---|
| `makeRng(seed)` | → `{uniform, normal, exponential(scale), int(n), laplace}` | `normal` is Box–Muller (2 uniforms). `laplace` is a unit-variance Laplace: a sign (1 uniform) times Exp(1) (1 uniform), divided by √2. |
| `cholesky(M, n)` | → bool | In place, lower triangle. |
| `cholSolveVec(L, n, x, off=0)` | → x | In place. |
| `cholRank1(L, n, p, delta, x)` | → bool | Updates the factor in place to `L Lᵀ + delta e_p e_pᵀ`. `x` is an n-length workspace. |
| `solveSPD(M, n, R, m)` | → X (n×m) | |
| `eigh(A, n)` | → `{values, vectors}` | Same tred2/tql2 as `engine.js`. |
| `symFn(A, n, f)`, `sqrtm(A, n)` | → n×n | |

### Covariances (Σ_g presets)

These use the same recipes and argument names as `engine.js`, and the parity test checks
them against `hebb.core`.

| export | → | notes |
|---|---|---|
| `cov3({variances, corr:[r01,r02,r12]})` | 3×3 | The correlation matrix is clamped to PSD (unit diagonal kept). |
| `covBlocks({sizes, within, between, variances})` | `{Sigma, B, labels}` | `within` and `variances` can be scalars or per-block arrays. |
| `covHierarchical({branching, corr, variance})` | `{Sigma, B, labels}` | `corr[l]` is the correlation of two inputs whose deepest common ancestor is at level l. |
| `covPowerLaw({B, alpha, seed, nonneg, normalize})` | B×B | λ_k ∝ k^−α in a random orthogonal basis. `normalize` gives mean variance 1. |
| `covFromSpectrum(lambdas, B, Q=null)` | B×B | Q's columns are the eigenvectors (identity by default). |
| `covRankOne({B, v=1s, scale=1})` | B×B | The scalar-drive input of Q-diversity. |
| `covDiag(variances)` | B×B | |
| `randomOrthogonal(B, rng)` | B×B | Columns form an orthonormal basis. |

### Inputs and network

| export | signature → result | notes |
|---|---|---|
| `sampleInputs(Σ, B, S, rng, kind='gauss')` | → G (S×B) | `G = Z Σ^{1/2}`, with Z drawn row-major. `'laplace'` gives the same covariance with heavy tails (kurtosis 6). Training and evaluation banks should be **two calls** (SPEC §3). |
| `makeAttachments(N, B, s, rng, deg=1, equal=false)` | → A | Each neuron is attached to `s` distinct inputs with total conductance `deg`. Shares are Dirichlet(1), or equal. |
| `makeNet({N, B, rng, s=3, deg=1, klo=0, khi=0, sigmaT=0, equal=false})` | → net | Draws the attachments first, then `t = sigmaT·normal`. `klo` and `khi` can be scalars or arrays. |
| `makeNetFrom({N, B, A, t, klo, khi, W})` | → net | Copies its inputs. |
| `cloneNet(net)` | → net | |
| `laplacian(net)` | → L (N×N) | |
| `inputCurrents(net, G, S)` | → R = G Aᵀ (S×N) | |
| `harmonic(net)` | → H (N×B) | The linear zero-potential response, `u = H g`, with rows summing to 1. |

### Equilibria, geometry, rule

| export | signature → result | notes |
|---|---|---|
| `equilibrate(net, G, S, {U0, R, tol=1e-10, maxIter=100, exact=false, out})` | → `{U, residual, iters, factorizations, updates, components}` | `U0` is a warm start. `out` may alias `U0`. Throws if a component has no attachment and no stiffness. |
| `equilibrateExact(net, G, S, opts)` | same | The slow reference port. |
| `nodeResidual(net, U, G, S)` | → max \|dE/du\| | An independent check of any U. |
| `energy(net, U, G, S)` | → mean E over the bank | |
| `dist2(U, S, N, V=null, M=N, out=null)` | → D2 (N×N, exact zero diagonal) | Computed from differences, so there is no cancellation. Passing V (S×M) gives the cross matrix. |
| `driftFromD2(W, D2, N, b, α)` | → dW | Zero diagonal. |
| `projectedResidual(W, dW, N)` | → r | Edges at w ≤ 0 with negative drift count as 0. |
| `drift(net, G, S, b, α, U=null)` | → `{dW, U, D2}` | |
| `residual(net, G, S, b, α, U=null)` | → r | The projected stationarity residual. |
| `step(net, G, S, b, α, dt, U=null)` | → U used | One projected Euler step, in place. |
| `run(net, G, S, b, α, {dt=0.2, T=5000, tol=1e-8, callback, solver})` | → `{steps, residual, U, eqResidual}` | As `onsite.run`: stops when `max\|ΔW\|/dt < tol`. |
| `twoNeuronD2(w, D0²=2)` | → D0²/(1+2w)² | The analytic Q-fusion calibration curve. |

### Live loop

```js
const rng = O.makeRng(1);
const { Sigma } = O.covBlocks({ sizes: [6, 6, 6, 6], within: 0.8, between: 0, variances: [2, 1, 0.5, 0.25] });
const net = O.makeNet({ N: 40, B: 24, rng, s: 3, klo: 3, khi: 0.5, sigmaT: 0.7 });
const G = O.sampleInputs(Sigma, 24, 256, rng);           // training bank
const sim = O.createSim({ net, G, S: 256, b: 0.3, alpha: 1, dt: 0.2 });
function frame() {
  sim.stepFor(12, Infinity, 1e-8);                       // ≈12 ms of steps per frame
  draw(sim.net.W, sim.D2, sim.U, sim.residual, sim.eqResidual);
  requestAnimationFrame(frame);
}
```

`createSim({net, G, S, b, alpha, dt=0.2, solver={}})` returns a `sim`. After every call
it maintains this invariant: **`sim.U` is the equilibrium at the current `sim.net.W`, and
`sim.D2`, `sim.dW` and `sim.residual` are consistent with it.**

| member | effect |
|---|---|
| `step(n=1)` | Takes n projected Euler steps, with one warm-started solve per step. |
| `stepFor(ms=12, maxSteps=∞, stopTol=0)` | Takes as many steps as fit in `ms`, at least one. Stops early once `lastChange < stopTol`. Returns the step count. |
| `setParams({b, alpha, dt})` | Recomputes `dW` and `residual` without re-solving. Hysteresis and continuation work because W is kept. |
| `refresh(cold=false)` | Call this after editing `net.W`, `A`, `t`, `klo`, `khi` or `G` in place. Edits to `A` or `G` are not seen until you do. |
| `setInputs(G, S)` | Switches to a new bank. The solve is warm-started if S is unchanged. |
| `energy()`, `components(eps)` | |
| fields | `U`, `D2`, `dW`, `residual` (projected max\|dw/dt\|), `eqResidual` (max \|dE/du\| of the last solve), `lastChange` (max\|ΔW\|/dt of the last step), `steps`, `lastSolve {iters, factorizations, updates, components}` |

### Analysis

| export | signature → result | notes |
|---|---|---|
| `components(W, N, eps=1e-9)` | → `{labels: Int32Array, K}` | Components of the live graph (w > eps). Labels are ordered by first member, as in the reference. |
| `clusterStats(U, S, N, labels, K?)` | → `[{members, size, proto (S), diamMax, diamMean, dEff}]` | Diameters are the D of each member to the prototype (SPEC §3: report them next to membership). |
| `populationModes(U, S, N)` | → `{q, V, Psi (S×r), r, dEff, C}` | `q` is descending and clipped at 0. Each `psi_a = q_a^{-1/2} Ũ V_a` has unit mean square. r is the number of modes with q > 1e-12·q₀. |
| `mds(D2, n, dim=2, prev=null)` | → `{X (n×dim), values, explained, negMass}` | Classical MDS. `prev` Procrustes-aligns the result to the previous frame, so points don't flip or rotate between frames. Use it to draw nonlinear response functions as points. `negMass` is 0 for D² from responses, which are exactly Euclidean. `explained` is the fraction of the positive spectrum captured. |
| `procrustes(X, P, n, dim)` | → X | In place. |

## Speed

Times are for `node tests/onsite_bench.mjs`, which uses the same V8 as Chrome, on this
4-vCPU VPS. The setup is S = 256, B = 24, dt = 0.2, α = 1, block covariance, and s = 3.
"Transient" is the first 60 steps from W = 0; "fixed pt" is 60 steps near stationarity.

| N | potential | b | transient steps/s | near fixed pt steps/s | components at fixed pt | max eq. residual |
|---|---|---|---|---|---|---|
| 40 | linear | 0.05 / 0.3 / 2 | 690 / 430 / 370 | 960 / 420 / 370 | 21 / 2 / 1 | 2e-13 |
| 40 | klo=3, khi=0.5, σ_t=0.7 | 0.05 / 0.3 / 2 | 290 / 270 / 300 | 350 / 300 / 310 | 9 / 1 / 1 | 2e-13 |
| 40 | klo=10, khi=0, σ_t=1 | 0.05 / 0.3 / 2 | 550 / 250 / 320 | 550 / 320 / 270 | 17 / 1 / 1 | 3e-13 |
| 80 | linear | 0.05 / 0.3 / 2 | 290 / 95 / 96 | 260 / 100 / 100 | 27 / 1 / 1 | 4e-13 |
| 80 | klo=3, khi=0.5, σ_t=0.7 | 0.05 / 0.3 / 2 | **57** / 78 / 87 | 58 / 94 / 95 | 6 / 2 / 1 | 3e-13 |
| 80 | klo=10, khi=0, σ_t=1 | 0.05 / 0.3 / 2 | 122 / 72 / 89 | 134 / 102 / 95 | 23 / 2 / 1 | 7e-13 |

The exact per-sample path (`exact: true`, the reference port) runs at 52 steps/s at N = 40
and 4.5 steps/s at N = 80 (klo=3, khi=0.5, b = 0.3). A cold first solve takes 1–4 ms.
The linear N = 80 cost is dominated by the 256 triangular solves, the residual check and
D², not by factorisation. A laptop should be at least this fast.

## Things a chapter author must know

- **Speed depends on fusion and nonlinearity.** The solve cost grows with the size of
  the largest component and with the number of distinct activity patterns inside it.
  - The slowest regime is many neurons in partially fused components with
    heterogeneous thresholds, during the transient.
  - Isolated neurons, linear networks and fully fused clusters are cheapest.
  - Use `sim.stepFor(budget)`, not a fixed `step(n)`, so a frame never stalls.
  - `sim.lastSolve` shows the work done.
- **Two banks.** The fixed-point identities hold for the training bank `G` (to
  `eqResidual`). For population claims, solve once per frame (not per step) on an
  independent evaluation bank:
  `O.equilibrate(sim.net, Geval, Seval, {U0: prevEvalU}).U`. Keep `prevEvalU` to
  warm-start it.
- **State any equilibrium claim with its tolerance.** Fixed-point statements need
  `sim.residual` (the plasticity residual) small, say below 1e-6, *and* `sim.eqResidual`
  (the neuronal residual) small. Show both live (SPEC §1).
- **dt.** This is explicit projected Euler on W. Keep `dt·α ≲ 0.5`. The bistable
  two-neuron calibration also needs dt small compared with the time spent near the
  separatrix; the tests use dt = 0.05.
- **Degenerate nets throw.** A neuron or component with no attachment and zero stiffness
  has no unique equilibrium, and `equilibrate` throws in that case. With `makeNet`
  (s ≥ 1, deg > 0) it cannot happen.
- **Two different "components".** The solver couples neurons through every `w ≠ 0`.
  The *displayed* clusters use `components(W, N, eps=1e-9)`. Projected Euler sets dying
  edges to exactly 0, so the two agree except for edges still decaying through (0, 1e-9].
- **Ties at threshold.** V is C¹, so a neuron sitting exactly at `t_i` is fine with
  either slope. The solver handles this case.
- **Cost of the analysis calls.** `populationModes` and `mds` run an O(N³) eigensolve
  (about 1 ms at N = 80). Call them per frame, not per step.
