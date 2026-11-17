# HebbianHarmonics — spec

A visual, pedagogical explorer for the user's note "Hebbian learning on a resistor
network" (verbatim in `docs/note.md`). Audience: the note's author and technically
literate friends. It must be honest: where the note's conjectures fail or need
amending, the page says so and shows the evidence.

## 1. Model (canonical; every chapter uses exactly this)

- Nodes: `B` boundary neurons with clamped voltages `g` (the lifted stimulus,
  `g = phi(x)`), `N` interior neurons with voltages `u`. Synapses are symmetric
  conductances `w_ij >= 0`.
- Potential synapses: every interior–interior pair and every interior–boundary pair,
  optionally thinned by a fixed random mask (connection probability `p`, default 1).
  No boundary–boundary synapses (both ends are clamped, so they never affect a
  voltage; they would trivially store `Sigma_g`).
- Voltages: Kirchhoff. `u = H g`, `H = L_II^{-1} W_IB`, `L_II = diag(deg) - W_II`,
  `deg_i = sum_j W_II[i,j] + sum_b W_IB[i,b]`. Rows of `H` are probability vectors
  (harmonic measure): `h_i` is the **harmonic embedding** of neuron `i`, a point in
  the simplex over boundary neurons. Boundary neuron `b` has `h_b = e_b`.
- Plasticity (the user's rule, regularizer a function of the weight only):

      dw_ij/dt = w0 - lambda * w_ij + eta_ij * Cov(u_i, u_j),   clipped at w_ij >= w_min

  with `Cov(u_i,u_j) = h_i^T Sigma_g h_j` (so I–B edges see `(H Sigma_g)_ib`).
  `eta_IB = eta`, `eta_II = gamma * eta` (lateral gain, default 1).
  `w_min > 0` is the "term that stops weights getting too small" (keeps `L_II`
  invertible; a neuron with all weights at the floor floats to the centroid).
- Dimensionless knobs: `theta = -w0/eta` (in covariance units), `w_min*lambda/eta`,
  `gamma`, `p`, and the initial condition. `lambda/eta` only sets the weight scale and
  the timescale. Mean-field mode integrates the ODE with the exact `Sigma_g`;
  stochastic mode uses minibatch covariance estimates.
- Fixed point on unclipped ("active") edges (w0 = -eta*theta is shared by both edge
  types, so gamma rescales the lateral threshold too):
  I–B: `(H Sigma)_ib = theta + (lambda/eta) w_ib`;
  I–I: `h_i^T Sigma h_j = theta/gamma + (lambda/(gamma*eta)) w_ij`.
  Clipped edges satisfy the corresponding `<=` with `w = w_min`. gamma = 0 means
  no lateral plasticity (handle explicitly, never divide by it).
- Stationarity: the projected residual `r = max |dw/dt|` over edges, where edges at
  the floor with negative drift count as 0. Equilibrium statements and the
  equilibrium readout are only claimed when `r` is below a stated tolerance; the page
  shows `r` live. Away from equilibrium the exact relation is
  `h_i^T C h_j = (dw_ij/dt - w0 + lambda w_ij)/eta_ij`; a readout applied mid-training
  or to the untrained net is labelled "equilibrium assumption violated" (control).

## 2. What the prototype already shows (scratch/proto*.py) — to be verified, not assumed

1. `w0 >= 0` (theta <= 0): the interior collapses to ONE cluster. Hebb on a resistor
   net is self-reinforcing (coupled neurons co-vary more) and bounded (voltages are
   convex combinations, so `Cov <= max Sigma`). Multiple clusters need `w0 < 0`,
   i.e. a constant pruning drift: synapses survive only where `Cov > theta`.
2. `theta` is a resolution knob. On a block covariance (12 inputs, 3 blocks, within
   0.5, across 0.1) the cluster count goes 1 -> 3 -> 12 as theta crosses the across-
   and within-block correlations. Looks like correlation clustering at level theta.
3. Cluster embeddings are NOT signed eigenvectors (h lives in the simplex). Within a
   cluster the fixed point is `h ∝ (Sigma h - theta)_+`: a thresholded Perron vector
   of that block — a nonnegative, localized "part", closer to NMF / sparse PCA than
   to PCA. This amends the note's "clusters = eigenmodes of g".
4. With all-to-all synapses the fixed point is multistable (depends on init) and all
   members of a cluster share the same `h` up to O(w_min): redundancy, not a
   reservoir of distinct random functions. Candidates for diversity: sparse mask
   `p < 1` (each neuron reads a random subset — the natural "limited read-write"
   version), minibatch noise, larger `w_min`, smaller `gamma`. To be measured.
5. Neurons per cluster in the prototype just follows basins of the initial condition
   (roughly equal). Whether allocation tracks mode energy is an open question the
   explorer must answer empirically.

## 3. Weight-space readout (the note's `C = H^+ R H^{+T}`), made precise

- Observed: weights only. Compute `H` from them. On active edges the fixed point gives
  linear equations in `C`: `h_i^T C h_j = theta + (lambda/eta_II) w_ij` and
  `(H C)_ib = theta + (lambda/eta) w_ib`. Clipped edges give only the inequality
  `<= theta + (lambda/eta) w_min`. There are no self-edges, so the diagonal of
  `H C H^T` is unobserved.
- Estimator: masked ridge least squares over symmetric `C` on active edges (I–I and
  I–B both; report each alone too), optional hinge penalty for clipped edges, optional
  PSD projection. Also report the note's naive pseudo-inverse formula as a baseline.
- Ambiguities: unknown `lambda/eta` rescales `C`; unknown `theta` adds
  `theta * 1 1^T` (because `H 1 = 1`). So `C` is identified at best up to scale and a
  shift along `1 1^T`, and only on the row space of `H` / on supra-threshold entries.
- Metrics (compare `C_hat` with the true lifted `Sigma_g` after projecting out `1`
  from both and fitting the best scale): (a) normalized Bures–Wasserstein distance
  (the natural metric on PSD matrices that tolerates rank deficiency), (b) Frobenius
  cosine (matrix alignment), (c) eigenvalue spectra overlaid and principal-angle
  overlap of the top-k eigenspaces. Baselines: `C = identity`, `C = diag(Sigma)`, a
  rank-k truncation of `Sigma`, and readout from the untrained random network.

## 4. Chapters of the explorer

0. **The idea** — the note retold with figures: resistor network, harmonic embedding,
   the rule, what was conjectured, and a boxed "what we found" summary with amendments.
1. **Three inputs** — live in-browser simulation, B = 3, N ≈ 60–150. A 3D view of the
   simplex with the embeddings moving over time (play/pause/scrub/reset), synapses
   drawn as lines with opacity ∝ weight, a linked view of `H Sigma H^T`. Controls:
   Sigma_g (variances + correlations, with presets), theta, w_min, gamma, init spread.
   Reader should be able to say: "they collapse to one point unless there is pruning;
   with pruning they go to vertices or to an edge midpoint when two inputs correlate."
2. **Many inputs, random statistics** — B ∈ {8..48}, Sigma_g random with a tunable
   spectrum (power-law decay) or planted blocks/hierarchy. 3D projection of embeddings
   (state the projection), similarity matrix sorted by cluster, cluster parts shown as
   bars over boundary neurons vs eigenvectors of Sigma, live weight-space readout of
   Sigma with the metrics above.
3. **Real data (MNIST)** — precomputed (Modal). Lifts: raw pixels (14×14, linear
   kernel — embeddings are then literally images) and random Fourier features of a
   Gaussian kernel with D ∈ {64, 256, 1024-if-feasible} (tunable param count), with
   stochastic minibatch training; theta swept. Shows: training curves; similarity
   matrix `H Sigma H^T` over training (slider + filmstrip); embeddings as images
   (pixel lift: `h_i` directly; RFF lift: response-weighted average stimulus and top
   activating digits); 2D map of embeddings over training (UMAP, warm-started and
   Procrustes-aligned; coloured by cluster / preferred digit); the learned graph;
   readout `C_hat` vs `Sigma_g` with metrics and baselines.
4. **Homeostasis explorer** — live sim on an assumed covariance AND on the real MNIST
   lifted covariance (embedded, small D). Knobs: theta, w_min, gamma, p, N. Readouts:
   number of clusters, cluster sizes vs "mode energy" (`h^T Sigma h` of the cluster
   part), within-cluster diversity, effective rank of H, readout quality. Plus
   precomputed phase diagrams over (theta, second knob). Must answer: what do the
   knobs actually trade off? Is the note's "resolution vs. expressivity, neurons ∝
   energy" picture right?
5. **Engineering choices & caveats** — one table: decision, default, alternative, how
   it would change the picture (symmetric weights, no B–B edges, mean-field vs
   stochastic, clipping floor, centered covariance, the lift, readout assumptions).

## 5. Form

Modelled on the CoherentModes explorer (`~/code/CoherentModes/explore/cifar_coherent/site/`,
its `DESIGN.md` is the pattern spec): one page, sticky hash-routed tabs, warm
off-white light theme, serif prose column (Source Serif 4) with Inter for UI,
hand-drawn SVG/canvas figures (no plotting library), vendored KaTeX and fonts (no
CDN), definitional captions ("Plotted: ... Values: ..."), `details` for maths,
key/caveat callouts, sidenotes, state in the URL, named "tours" (preset states with a
note). Served over HTTP (the laptop's page server), not file://.

Difference from CoherentModes: chapters 1, 2, 4 SIMULATE LIVE in the browser (shared
engine `site/js/engine.js`, pure ES module, mirrored by `hebb/core.py` and
cross-checked by a node-vs-numpy test). Chapter 3 is precomputed:
`site/data/mnist/*` (JSON + binary + PNG sprites). No build step: `site/index.html`
loads `site/content/<tab>.html` fragments and `site/js/ch_<tab>.js` modules at
runtime, so each chapter is a disjoint set of files. Synthetic covariances are
labelled as such. Tone: first person, exploratory, title plus a joke "or, ..." subtitle.

## 6. Done means

All chapters load with zero console errors; screenshots of each chapter inspected;
engine JS/Python cross-check passes; MNIST numbers on the page match the saved
artifacts; GPT review rounds addressed; published with `results publish`.

## 7. Amendments after GPT review (round 1) — these override anything above

1. Lateral fixed point uses `theta/gamma` (see §1). Readout equations likewise.
2. "theta <= 0 gives one cluster" holds only when Sigma has no negative entries.
   Negative covariances are pruned by the clip at any theta (e.g. Sigma = I - 11^T/3
   gives three vertex clusters at theta = 0). Chapters must include a signed-covariance
   preset and distinguish "pruning by drift (w0 < 0)" from "pruning by anticorrelation".
   Chapter conclusions follow verified results in docs/findings.md, never the reverse.
3. Readout identifiability: I–B equations observe `H C` (not only `H C H^T`), so they
   constrain more than the row space of H. With gamma != 1 an unknown theta is NOT a
   pure `11^T` shift for the combined system. Report, per edge family, the rank /
   conditioning of the measurement operator on symmetric matrices. Keep three settings
   apart: parameters known (score the full covariance, no projection); unknown scale;
   unknown theta (score `P C P`, labelled "contrast-subspace score", P = I - 11^T/B —
   note this also discards `1 a^T + a 1^T` terms).
4. Bures–Wasserstein needs PSD: always clip negative eigenvalues of C_hat before it,
   and report the negative spectral mass removed. Normalisation:
   `BW(A,B)^2 / tr(B)` with A scaled to minimise it; define the all-zero case as 1.
5. Allocation protocol: besides descriptive scatter plots, a controlled planted-family
   experiment — fixed supports and init protocol, vary one planted block's energy
   (variance) or size, match learned clusters to planted supports, paired seeds, report
   unresolved/missing modes and seed variability. "Cluster energy" h^T Sigma h is an
   outcome; planted energy is the controlled input. Cluster definition must be stated
   and shown robust to tolerance/order (leader clustering in core.clusters is a quick
   look, not the headline definition).
