# HebbianHarmonics — spec v2 (on-site potentials, distance rule)

v1 (`docs/spec_v1.md`) studied the raw covariance rule `dw = w0 − λw + η Cov(u_i,u_j)` on a
linear resistor network. Its results were largely degenerate: one-cluster collapse unless
pruned, exact clones inside a cluster, allocation inherited from the transient, rectified
output adding nothing structural. v2 rebuilds the explorer around a model in which the
objects being clustered are whole neuronal response functions with an exact geometry.
The author's decisions (2026-11-17): stick to the basic model; synthetic inputs simulated
live in the browser only; condense all v1 material into one chapter; the outcomes that
matter are (T1) within-cluster diversity, (T2) an allocation law n(λ), (T3) clusters vs
eigenmodes, (T4) weight-space readout. The page stays honest: where a target is not
reached by the basic model, it says so, shows the evidence, and names the missing ingredient.

## 1. Model (canonical)

- Nodes: `B` boundary neurons clamped to the input `g ~ p_g` (zero-mean, covariance
  `Sigma_g = sum_a lambda_a v_a v_a^T`); `N` interior neurons with activities `u`.
- Energy (each undirected edge once):

      E_W(u; g) = 1/2 sum_{i<j} w_ij (u_i − u_j)^2 + 1/2 sum_{i,b} a_ib (u_i − g_b)^2 + sum_i V_i(u_i)

  `w_ij >= 0` plastic interior–interior conductances; `a_ib >= 0` FIXED input attachments
  (each neuron attached to `s` random inputs, total input conductance `deg`, default 1);
  `V_i` convex on-site potentials.
- Fast dynamics: `tau du_i/dt = −dE/du_i`. Equilibrium (unique, since E is strictly convex
  when every neuron has an attachment): `u = F^W(g)`, the **response functions**. Node
  equation: `d_i u_i + V_i'(u_i) = sum_j w_ij u_j + sum_b a_ib g_b`, i.e.
  `u_i = (d_i id + V_i')^{-1}(input current)`: V' is the inverse transfer function.
- On-site family (default): two-slope quadratic `V_i(u) = k_lo/2 (u − t_i)_-^2 + k_hi/2 (u − t_i)_+^2`
  with per-neuron threshold `t_i` (heterogeneity `sigma_t`) — a leaky-rectifying transfer,
  piecewise linear so equilibria are exact. `k_lo = k_hi = 0` is the linear case, where
  `u = H g` and rows of `H` are the v1 harmonic embeddings. A smooth alternative
  (`V' = k * sinh`-type saturating transfer) is an option, not the default.
- Geometry: `<F_i, F_j>_p = E_g[F_i F_j]`, `D_ij^2 = E_g[(F_i − F_j)^2]`. Expectations are
  over a fixed sample of `S` inputs (default 256, seeded; Gaussian by default, a
  non-Gaussian option with the same covariance). The same sample is used for learning and
  for measurement, so stated fixed-point identities hold to solver tolerance.
- Slow learning = gradient descent on `J(W) = E_g[min_u E_W] + R(W)`,
  `R = sum_{i<j} (alpha/2 w^2 − b w)`. By the envelope theorem

      dw_ij/dt = b − alpha w_ij − 1/2 D_ij^2,   projected at w >= 0
      fixed point: w_ij = (b − D_ij^2/2)_+ / alpha

  Knobs: `b` (fusion scale: edges live iff `D^2 < 2b`), `alpha` (stiffness: `b/alpha` is the
  maximal conductance, to be compared with `deg`), `k_lo, k_hi, sigma_t` (nonlinearity),
  `s` (attachment sparsity), `N`, `B`, `Sigma_g`.
- Stationarity: projected residual `r = max |dw/dt|`; equilibrium statements only when `r`
  is below a stated tolerance, shown live.

## 2. What the prototypes already show (scratch/onsite_proto.py, onsite_proto2.py, alloc_proto.py) — verify, don't assume

1. Fixed attachments: sweeping `b` takes the network from N singletons through a few
   fused clusters to one blob; clusters are connected components of the live graph and
   contract as they fuse (self-consistent: fusing moves the points).
2. PLASTIC attachments under the same energy rule with a fixed input degree all migrate
   to the lowest-variance inputs (blocks with variances 2/1/0.5/0.25: 35–40 of 40 neurons
   on the 0.25 block). Pure energy minimisation prefers quiet inputs. Hence fixed
   attachments in the base model; allocation needs a further principle.
3. Function-space similarity matching with fixed per-neuron variance `nu`
   (`min 1/2 ||T − S||^2`, `||F_i||^2 = nu`) gives the water-filling mass law
   `q_a = (lambda_a − mu)_+` exactly (8 modes, to 3 digits), but neurons are generically
   MIXED across modes (rotation degeneracy): mass law yes, neuron-count law only with a
   specialising ingredient. Adding the finite-range fusion term purifies only partly.

## 3. Questions each chapter must answer (with a live figure and, where needed, a precomputed sweep)

Conventions for every experiment (from the plan review, 2026-11-17):
- **Two input banks.** A training bank `G` (S samples) defines the learning expectation;
  fixed-point identities are exact for its empirical UNCENTRED second moment
  `C_hat = G^T G / S`, not for `Sigma_g`. An independent evaluation bank is used for every
  population claim (diversity, geometry, readout vs `Sigma_g`). Report the three errors
  separately: fixed-point/solver, sampling (`C_hat` vs `Sigma_g`), inverse-problem.
- **A connected component is not a mode.** Clusters are connected components of the live
  graph; always report their diameter (max and mean `D` to the prototype) next to
  membership, because chains can be wide.

- **Q-fusion.** Is the fixed point the stated one (w vs D^2 on the curve; envelope identity
  checked numerically)? Calibration: the analytic two-neuron case, `D^2(w) = D0^2/(1+2w)^2`
  for unit attachments, `dw/dt = b − alpha w − D^2(w)/2`, which is BISTABLE (e.g. b = 0.2,
  alpha = 0.1: w = 0 and w ≈ 1.754 both stable). Then N neurons: cluster count, diameter
  vs `b` and `alpha`; continuation in b up and down (hysteresis); is refinement nested?
- **Q-resolution (eq. 10–11).** Three separate statements, not to be conflated:
  (1) the frozen-response anisotropic law `D^2 = sum_a lambda_a (Delta h_a)^2` (exact);
  (2) edge activation from the uncoupled state, for planted pairs differing along one
  mode only: the edge first lives when `|Delta h_a| < sqrt(2b/lambda_a)`;
  (3) self-consistent fusion and its loss of stability under continuation, which
  feedback shifts away from (2). `Delta h` is always the uncoupled (attachment-defined)
  separation unless stated.
- **Q-diversity (T1).** Decisive controlled experiment: rank-one (scalar-drive) input,
  matched attachments, three arms: heterogeneous nonlinear potentials, homogeneous
  nonlinear, linear. Measures: held-out error of an affine fit of each response from the
  FULL input vector (nonlinearity, not linear mixing); held-out residual of a 1-D
  nonlinear fit from the planted scalar (single-valued function of the shared mode);
  within-cluster `d_eff = (Tr C)^2/Tr(C^2)` and member-vs-mean plots as descriptive
  only. Then the same measures on multi-mode inputs, and their dependence on `b`, `alpha`
  (stronger fusion pulls members together).
- **Q-eigen (T3).** Population response operator `S f = sum_i F~_i <F~_i, f>`; its
  eigenfunctions `psi_a = q_a^{-1/2} sum_i (v_a)_i F~_i` vs (i) cluster prototypes,
  (ii) eigenmodes of `Sigma_g`. When are cluster prototypes near-orthogonal (then
  `q_a ≈ nu n_a`) and when are the modes shared component + contrasts (the
  `F_A = s + c, F_B = s − c` example, built explicitly)?
- **Q-allocation (T2).** (a) Base model: neurons per cluster is set by attachments and the
  fusion scale; show it, state plainly that this is not n(lambda). (b) Plastic attachments
  under the energy rule go to quiet inputs. (c) An ADDITIONAL principle, fully specified
  and labelled as not derived from the conductance energy: linear response vectors
  `x_i in R^B` (whitened coordinates, `F_i = x_i · Sigma^{-1/2} g`), target `T = Sigma_g`,
  objective `1/2 ||Sigma − X^T X||_F^2`, hard amplitude constraint `|x_i|^2 = nu`
  (so `sum_a q_a = N nu`). Its law is fixed-trace water-filling `q_a = (lambda_a − mu)_+`
  (for a soft penalty r instead: `q_a + r'(q_a) = lambda_a`; say which is plotted). Three
  experiments kept apart: (c1) verify the mass law; (c2) measure specialisation
  (per-neuron purity, counts `n_a`) under each added ingredient — fusion term,
  non-negativity/rectification; (c3) measure how much that ingredient moves `q_a` off the
  baseline law. Report honestly if no tested ingredient gives `n_a ≈ q_a/nu`.
- **Q-readout (T4).** From weights alone: live edges give `D_ij^2 = 2(b − alpha w_ij)`,
  dead edges `D^2 >= 2b`. Linear zero-potential case only: `H 1 = 1`, every
  `d = h_i − h_j` has `d^T 1 = 0`, so `C + 1 a^T + a 1^T` is invisible for ANY vector a.
  Target is therefore `P C P`, `P = I − 1 1^T/B`. Report rank and singular values of the
  live-edge measurement operator on symmetric contrast-space matrices (outer products
  `d d^T` must span it), recover on the identifiable subspace (masked ridge LS, optional
  inequality hinge from dead edges, PSD projection), score against `P C_hat P` first and
  `P Sigma_g P` second (Frobenius cosine, Bures–Wasserstein on the contrast space,
  spectra). Any full-covariance completion is prior-dependent and labelled so. With
  on-site potentials (even quadratic leak) `H 1 != 1`: state what changes. Nonlinear case:
  the weights determine a partial response-distance geometry; show that (MDS of the
  recovered distances vs the true ones) and stop there. Baselines: identity, diag,
  untrained network.

## 4. Chapters (tabs)

0. **idea** — the note retold for the new model: energy, on-site potential as transfer
   function, response functions, the rule as an envelope-theorem gradient, a boxed
   "what we found" keyed to T1–T4.
1. **fusion** — live, B = 3 (linear case drawn in the simplex; nonlinear case by classical
   MDS of `D`, stated on the page) and small N. Sliders: b, alpha, nonlinearity,
   heterogeneity, attachment sparsity; Sigma presets. Linked: point cloud with live edges,
   `D^2` matrix with the `2b` contour, `w` vs `D^2` scatter on the theory curve, residual.
2. **many** — B up to ~24, planted blocks / hierarchy / power-law spectra. Cluster count
   and radius vs b (live + precomputed sweep with seeds), up/down hysteresis, the
   resolution law.
3. **diversity** — Q-diversity and Q-eigen.
4. **allocation** — Q-allocation (a)–(c).
5. **readout** — Q-readout.
6. **old rule** — one chapter condensing v1: the covariance rule, why it collapses, parts
   not eigenvectors, clones, transient allocation, rectified output; with the corrections
   of `docs/review_corrections.md` applied. Reuses at most one live v1 figure.
7. **method** — engineering choices table and caveats; what was not done (stability
   Hessian eq. 12, lagged statistics, generative phase, real data).

## 5. Form and code

- Keep the existing shell (`site/index.html`, `main.js`, `ui.js`, `charts.js`,
  `view3d.js`, CSS, vendored KaTeX/fonts) and `site/DESIGN.md` conventions; replace
  chapter content. v1 chapter files, MNIST/nonlin data and code leave the tree (git
  history keeps them) except what chapter 6 needs.
- Reference implementation `hebb/onsite.py` (numpy); browser engine
  `site/js/onsite_engine.js` (pure ES module, no deps); node-vs-numpy parity test.
  Precomputed sweeps: `hebb/sweeps2/*.py` → `site/data/sweeps2/*.json`, seeds fixed,
  each JSON carries its config. Everything runs on CPU in minutes; no Modal.
- Findings go in `docs/findings_v2.md`, each claim citing script and JSON. Page text
  quotes only numbers present in the JSON or computed live.
- Done: all tabs render with zero console errors (desktop + mobile screenshots looked
  at); parity test passes; an independent GPT review of maths, code and page claims has
  no unresolved material finding; published with `results publish`.
