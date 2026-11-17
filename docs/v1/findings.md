# What the plasticity rule actually does — experimental findings

> Archived v1 document (moved to `docs/v1/` on 2026-11-17). Code and data paths it cites (`hebb/sweeps/`, `hebb/mnist/`, `hebb/nonlin/`, `site/data/...`) refer to the tree at commit `f613d68`; git history keeps them.

Experimentalist's notes for the explorer. Every claim cites the script that produced it (`hebb/sweeps/*.py`, seeds fixed; `hebb/sweeps/run_all.sh` regenerates all) and its JSON (`site/data/sweeps/*.json`). Canonical model: `hebb/core.py` (imported, never edited).

**Conventions.** `lam = eta = 1`, Euler `dt = 0.05`, mean-field unless stated. `A = Sigma − theta 11^T` (theta-shifted covariance); `omega = N·wmin·lam/eta` (floor leak, F5); `kappa0 = 1^T Sigma 1 / B^2` (mean covariance of two random initial embeddings, F3). *Cluster* = average-linkage clustering of embeddings in L1, cut 0.1 (order-independent); K excludes all-dead clusters and, where stated, clusters with < max(2, 2%N) members. Cuts 0.05/0.2 give the same K except within one grid step of a transition (`phase.json` `K05`, `K20`). "Fixed point" only if the projected residual max|dw/dt| < 1e-4; it was < 1e-5 in every mean-field run here unless a number is given. Default init: `W_IB ~ Exp(0.2)`, `W_II` at the floor; dense random `W_II ~ U(0, ii_scale)` studied in F3.

## F0. Everything is the theta-shifted covariance
Since 1^T h = 1: I–B drift `eta (A h_i)_b − lam w_ib`; I–I drift `eta h_i^T(gamma Sigma − theta 11^T)h_j − lam w_ij` (lateral threshold theta/gamma, as SPEC §7.1; core.step does this). A synapse survives iff its A-inner product exceeds lam·wmin/eta, so pruning by drift (theta > 0) and by anticorrelation (negative Sigma entries) are one mechanism. `Sigma = I − 11^T/3`, theta = 0.02: three vertex clusters, 3/3 seeds (tour `anticorrelated`); signed power-law has 3 stable parts at theta = 0 (`parts.json`).

## F1. A cluster is a "part": a nonnegative Pareto eigenvector of A, not an eigenvector of Sigma
**Claim (exact).** In a cluster sharing h (support S) all members are equipotential for every stimulus, so lateral synapses inside it carry no current and the harmonic condition reduces to h ∝ (eta/lam)(A h)_+, i.e. (wmin → 0) `A_SS h_S = c h_S, c > 0, h_S > 0, (A h)_b ≤ 0 off S`. N, gamma and lateral weights drop out. `c = (lam/eta)·sum_b w_ib` = a member's summed input conductance = (h^T Sigma h − theta)/|h|^2.
**Stability (derived).** Lone neuron on S: dw/dt = eta(Sigma_SS w/s − theta 1) − lam w, s = 1^T w; at the FP s = eta c/lam; Jacobian = (lam/c) A_SS (I − h1^T) − lam I (using 11^T(I − h1^T) = 0). In A_SS's eigenbasis, A_SS − c h1^T is triangular with diagonal (0, mu_2, …), so eigenvalues are −lam and lam(mu_k/c − 1); off S, w_b grows at eta(A h)_b. **Stable iff c is strictly the top eigenvalue of A_SS and (A h)_b < 0 strictly off S** ⇔ h is a strict local max of the nonnegative Rayleigh quotient h^T A h/h^T h. The cluster's collective mode obeys the same equation.
**Evidence** (`s1_parts.py`, `parts.json`: exact enumeration B ≤ 16, 6 covariances × 13 theta; network N = 120, 4 seeds, gamma ∈ {0,1}, wmin ∈ {1e-5, 1e-3}). gamma = 0, wmin = 1e-5: matched fraction (cos > 0.98 to a stable part) = 1.00 in every cell except where neurons die (power-law, theta ≥ 0.55: 0.69–0.95); K = n_stable in 71/78 cells (exceptions: parts with tiny basins, c ≈ 0.1). gamma = 1: still matched 1.00, but a subset populated (F3). Vectorised lone-neuron flow (2000 starts) finds the same parts.
**Corollaries.** (1) theta < min_bc Sigma_bc ⇒ A > 0 entrywise ⇒ exactly one part (full-support Perron vector). This is the correct form of "theta ≤ 0 ⇒ one cluster". (2) max_{b≠c} Sigma_bc < theta < min_b Sigma_bb ⇒ exactly the B vertices (A_SS a Z-matrix). (3) vertex b exists iff Sigma_bb > theta; theta > max_b Sigma_bb ⇒ no parts, network dies. (4) Uniform block (size m, corr rho, across rho_a): block part stable iff rho_a < theta < rho: exact correlation clustering. block12: 1 part (theta < 0.1), 3 (0.1–0.5), 12 (> 0.5).
**vs eigenvectors:** block12/hier8 parts lie in the top-K eigenspace (norm fraction 1.00) but are its nonnegative localized rotation (best single eigvec cos 0.71–0.74). ring16 (Fourier eigvecs, all corr > 0): localized bumps of width 10 → 1 as theta 0.07 → 0.85, one per input; best eigvec cos 0.76 → 0.46 (place fields, not Fourier modes). Signed power-law: 0.61–0.78 of norm in top-K eigenspace, best eigvec cos 0.54–0.68. Nonneg factor model at low theta: main part ≈ top eigvec (cos 0.96). Always ≈ Perron vector of Sigma_SS (cos ≥ 0.97), exactly that of A_SS.
**Amendment:** each cluster is a nonnegative, localized part: the Perron vector of the theta-shifted covariance on a support (a local maximum of nonnegative PCA of Sigma − theta 11^T); theta sets part size; parts are eigenvectors only when the top eigenvector is nonnegative.

## F2. n_stable is a function of theta; the number of clusters is not
n_stable(theta) depends only on Sigma, theta. Populated K also depends on gamma, init, omega, noise. block12, theta = 0.2: 3 stable parts; gamma = 0 → K = 3 (4/4), gamma = 1 → K = 1 (4/4).

## F3. Lateral plasticity gates resolution: theta_c = gamma·kappa0, applied coarse-to-fine
**Mechanism.** At t = 0 the lateral drift is eta(gamma h_i^T Sigma h_j − theta); for exchangeable init E[h_i^T Sigma h_j] = kappa0. If theta < gamma·kappa0 lateral synapses grow, the population binds into one equipotential blob and lands in ONE part (the loudest when energies differ). Otherwise neurons choose independently. Recursively, neurons heading to a coarse part with self-covariance E_parent stay bound while gamma·E_parent > theta and pick one finer part together.
**Prediction vs data** (`phase.json`, N = 120, 3 seeds, theta grid avoiding exact correlation levels; block12 kappa0 = 0.275, E_block = 0.625; hier8 kappa0 = 0.30, E_half = 0.55, E_pair = 0.80):

| cov | gamma | pred. 1→coarse | obs. | pred. coarse→fine | obs. |
|---|---|---|---|---|---|
| block12 | 0, 0.25 | 0.10 | 0.12 | 0.50 | 0.53 |
| block12 | 0.5 | 0.14 | 0.17 | 0.50 | 0.53 |
| block12 | 0.75 | 0.21 | 0.22 | 0.50 | 0.53 |
| block12 | 1 | 0.275 | 0.27 | 0.625 | 0.57 (K=8), 0.63 (12) |
| block12 | 1.5 | 0.41 | 0.42 | 0.94 | never (K=3 to 0.63) |
| block12 | 2 | 0.55 | 0.57–0.63 | 1.25 | never |
| hier8 | 0.5 | 0.15 | 0.17 | 0.30; 0.60 | 0.32; 0.63 |
| hier8 | 1 | 0.30 | 0.27 | 0.55; 0.80 | 0.47; 0.70 |
| hier8 | 1.5 | 0.45 | 0.42 | 0.83 | never |

All transitions are within 0.03–0.08. The boundary is N-independent (N = 60/120/240 within a grid step) and insensitive to ib_scale (0.02–1). At large gamma and high theta the bound population can die (block12 gamma = 3, theta ≥ 0.7).
**Dense random lateral init** (axis `ii_scale`; `dynamics.json` init grid): kappa0 is unchanged, but the 1→3 boundary moves from 0.27 (ii ≤ 0.03) to 0.32 (0.1), 0.37 (0.3) and 0.53 (1.0). ii = 1 also kills theta ≥ 0.53 (dead 0.5–0.9). Large ii with small ib = 0.02 gives K = 1 at theta = 0.3 (3/3). ii_scale ≤ 0.03 is indistinguishable from the floor init.
**Packing condition (derived, verified).** Coexisting clusters need gamma h_k^T Sigma h_l ≤ theta + lam·wmin/eta. For gamma ≤ 1, disjoint-support parts are automatically compatible. Ring16, theta = 0.4: gamma = 0 populates all 16 width-4 bumps (adjacent cross-cov 0.645 > theta); gamma = 1 populates only 3–4 non-overlapping bumps (max cross-cov 0.04–0.26 < 0.4). In every gamma = 1 tour, max cross-cov < theta/gamma.
**Amendment:** theta sets which parts exist; gamma sets how many a densely connected population resolves. Lateral plasticity merges the population at every level whose self-covariance exceeds theta/gamma, and forbids coexisting clusters whose cross-covariance exceeds theta/gamma.

## F4. Any allocation is a fixed point
The F1 equation does not contain n_k. The only n-dependence is the floor: h_k(|w_kB| + wmin(N − n_k)) = w_kB + wmin Σ_{l≠k} n_l h_l, an O(omega/c) pull toward the population mean. This holds for any weight-only regularizer (w = f(cov) ⇒ h ∝ f(Sigma h)) and for per-neuron synaptic scaling (h is scale-invariant). So allocation is inherited from the transient (init, gating, noise), never selected by the equilibrium. Converged allocations differ across seeds with residual < 1e-5 (e.g. `three-corners` 34/30/26, 34/32/24, 35/28/27).

## F5. The floor is a knob: omega = N·wmin·lam/eta
Every clipped synapse conducts wmin, pulling clusters toward the population mean with relative strength ~omega/c.
- **Extinction** (block12, N = 120): all dead at theta ≥ 0.42 (wmin 3e-3), ≥ 0.57 (1e-3), 0.8 (3e-4), never ≤ 0.8 (≤ 1e-4). Across N ∈ {60, 120, 240} the threshold roughly collapses on omega: 0.024 → ~0.8, 0.036 → ~0.75, 0.06 → ~0.63, 0.12 → ~0.6, 0.24 → ~0.5, 0.36 → ~0.42, 1.2 → ~0.35. Mechanism: above every initial drive, boundary weights decay; a neuron survives only if its embedding concentrates before its boundary conductance falls to ~omega.
- **Leak-stabilised wrong states:** wmin = 1e-3, theta = 0.45 (block12) sits on vertices that are not stable parts at wmin → 0.
- **Default wmin = 1e-3 is too large for N ≥ 100:** block12 dies for theta ≥ 0.57, signed3 for theta ≥ 0.43 (`parts.json`, `g1_w0.001`). Recommend omega ≤ 0.02 (wmin = 1e-4 for N ≤ 200).

## F6. Allocation: transient spectral filter n_k ∝ lambda_k^beta(theta) with a hard cutoff
**Planted family** (`s3_alloc.py` B, `alloc.json`): 4 blocks of 3 (within 0.6, across 0); block 0's variance v0 or size m0 varied; N = 150; 8 paired seeds; match by ≥ 80% centroid mass; residual ≤ 1e-6; no unassigned/dead neurons. Fraction in block 0:

| family, theta, gamma | values (x: frac) | beta |
|---|---|---|
| var, 0.25, 0 | 0.5: .048±.02, 0.7: .10±.03, 1: .25±.05, 1.4: .49±.05, 2: .72±.04, 2.8: .87±.03 | 2.9 |
| var, 0.15, 0 | .017, .061, .25, .61, .85, .95 | 4.2 |
| var, 0.05, 0 | 0, .001, .25, .90, .98, .99 | 6.8 |
| var, 0.25, 1 | .048, .10, .25, .53, .76, .90 | 3.1 |
| var, 0.15 / 0.05, 1 | collective: all 150 in ONE block (the loudest when v0 ≠ 1) | – |
| size m0 (var 1), 0.25, 0 | 1: .036, 2: .11, 3: .25, 4: .38, 6: .64 | 2.2 |
| size m0 at fixed top eigenvalue 2.2, any theta, gamma 0 | 1: .28, 2: .25, 3: .25, 4: .22, 6: .20 | −0.2 |

The controlling energy is the block's top eigenvalue lambda_k = lambda_max(Sigma_SkSk). At equal lambda, bigger blocks get slightly fewer neurons. Below the collective threshold allocation is winner-take-all. A mode goes missing only through gating or at very large beta (v0 = 0.5, theta = 0.05: 0 neurons, 8/8 seeds).
**gamma = 0, N → ∞** (part A, `alloc_limit.json`; Sigma = diag(0.5, 0.75, 1, 1.5, 2, 3), 3000 neurons): beta = 7.6, 5.8, 4.3, 2.8, 2.0, 1.6, 1.5 at theta = 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45; theta = 0: all to the largest variance. Descriptive check (`powerlaw_pos12`, gamma 0): n ∝ lambda_top^beta, beta = 4.0, 3.0, 2.1 at theta = 0.2, 0.3, 0.5 (r = 0.99). Intuition: d log h_b/dt ⊃ v_b/s − theta/w_b. The first term rewards energy (the loudest wins as theta → 0); the second rewards the largest initial weight, so at higher theta the init decides. Hard cutoff at c ≤ 0 (tour `partial-extinction`).
**Ingredients, rho weight-only** (part D, `alloc_ingredients.json`):
- Init scale: no effect (ib_scale 0.02/0.2/2: same beta ±0.1).
- Init log-spread (a "temperature"): at theta = 0.2, log-sd 0.5/1/2/3 gives beta 5.9/2.4/1.2/0.76; uniform init gives 4.9 vs exponential 2.8. beta ≈ 1 is reachable only by tuning the init spread against theta.
- Shared minibatch noise makes allocation steeper: batch 8 sends all neurons to the loudest input. Batch 32 leaves only the two loudest at theta 0.1–0.2, and drains the v = 1 part at theta = 0.3 (0.057 → 0 between steps 4000 and 40000). No hopping toward proportionality.
- Nonlinear rho: −w^p (p = 2, 3) leaves 35–97% of neurons in mixed states with beta 2–4. −w^0.5 gives beta 2.0/1.6/1.3 at theta 0.1/0.2/0.3. A soft floor +eps/w is steeper or lethal. None is robust, as F4 predicts.

**Amendment:** the spectral filter exists but belongs to the transient, not the equilibrium. Parts with non-positive supra-threshold eigenvalue get nothing. The rest get n_k ∝ lambda_k^beta, with beta ≈ 7 → 1.5 as theta rises from 0 to the smallest variance (also set by the init weight spread; gamma = 0 or above the collective threshold). Lateral plasticity below threshold, low theta and minibatch noise push toward winner-take-all. Any allocation, once reached, is a fixed point.

## F7. Members are exact clones (dense init); a small reservoir only with sparse wiring and gamma = 0
`s4_diversity.py`, `diversity.json` (block12, theta = 0.3, 3 seeds). With dense synapses, member-to-centroid L1 < 1e-5 for every wmin (1e-5–1e-2), gamma (0–2), N (30–240) and batch size (8–512). Cluster and block-level voltage participation ratio = 1.000. This is exact: the dynamics is permutation-symmetric and minibatch noise is shared, so identical neurons stay identical forever.
*Labelled comparison, sparse mask p < 1:*
- gamma = 1: members differ in h by L1 0.04–0.09, but voltage correlation is > 0.997; block-level PR 1.001–1.06 (max 4).
- gamma = 0: the population fragments into 14–22 clusters of exact clones. Block-level PR is 1.15/1.36/1.74/2.06 at p = 0.7/0.5/0.3/0.15, and 1 − mean voltage correlation goes 0.07 → 0.33.
**Amendment:** with dense connectivity every member of a cluster computes the same function (n-fold redundancy, not a reservoir). Sparse random wiring without lateral plasticity yields a few distinct functions per mode, bounded by the mode's dimension.

## F8. Multistability and hysteresis
**Deterministic slow loops** (`s5_dynamics.py` (a), `hysteresis.json`): theta 0.02 → 0.80 → 0.02, steps of 0.02, state carried over, N = 120; both seeds identical.
- block12, gamma = 1: K = 1 all the way up and dies at 0.66. The symmetric block state stays on its unstable part until c = 2.5 − 4·theta hits 0 at 0.625. Coming down it stays dead until theta = 0.26 < max_b(Sigma m)_b = 0.275.
- block12, gamma = 0: up 1 → 3 (0.18) → 8 (0.58) → 7; down 7 persist to 0.50, then 3, then 1 at 0.06.
- hier8, gamma = 1: K = 1 throughout. hier8, gamma = 0: up 1 → 2 (0.10) → 4 (0.46); down 4 to 0.30, 2 to 0.06.
- The hier8 half part (unstable above 0.3) persists to 0.42 going up: delayed bifurcation of an exactly symmetric state with ~0 residual. **A small residual does not certify stability.**

**With private multiplicative jitter** (`s5b_jitter.py`, `hysteresis_jitter.json`, w ← w·exp(sigma·xi) each step, sigma 1e-3/1e-2, 1500 steps per theta):
- gamma = 0 loops nearly close and K tracks n_stable(theta) both ways within 1–2 grid steps. block12: up 1→3 at 0.14 and 3→12 at 0.54; down 12→3 at 0.46 and 3→1 at 0.06. hier8: up 2→4 at 0.34 and 4→8 at 0.62; down 8→4 at 0.58 and 4→2 at 0.26. So the deterministic gamma = 0 hysteresis was mostly symmetry artefact.
- gamma = 1: K = 1 along the entire loop for both covariances and both sigmas, with no extinction. **The gamma = 1 memory is real:** a laterally bound population never splits as theta rises, while a fresh start at the same theta gives 3 or 12 clusters.
- sigma = 1e-2 produces transient spikes (K 9–14 at single steps).

**Init dependence** (`dynamics.json` init grid; block12, theta ∈ {0.3, 0.45}, ii ∈ {0, 0.03, 0.1, 0.3, 1} × ib ∈ {0.02, 0.2, 1}, 3 seeds): K = 3 whenever ii ≤ 0.03 or ib ≥ 1. Large lateral relative to boundary init gives K = 1, or 0 (theta 0.45, ii 0.3, ib 0.02, seed 1). The ratio of initial lateral to boundary conductance is what matters.

## F9. Timescales; mean-field vs stochastic
**Convergence** (block12, N = 120, 4 seeds, steps of dt = 0.05 to max|dW/dt| < 1e-5): 300–600 steps away from transitions. Critical slowing near them: 850–950 at theta 0.08 and 900–1650 at 0.12 (transition at 0.1); 1250–2650 at 0.48–0.5 (transition at 0.5).
**Minibatch vs mean-field** (gamma = 1, 3 seeds, 16000 steps): batch ≥ 128 reproduces mean-field K and allocation (theta 0.3, seed 0: MF .37/.30/.33; b128 .38/.28/.33; b512 .37/.29/.34). Small batches enlarge the collective regime: theta 0.3 with batch 8 or 32 gives K = 1 (3/3); theta 0.45 with batch 8 gives K = 1, batch 32 gives K = 3. Zero neurons changed block between snapshots at steps 4000–16000: no basin hopping.

## F10. Readout: equilibrium weights store a parts-sketch, not Sigma
**Identifiability (derived).** At a clustered FP the weights are a function of {(h_k, c_k, n_k)} plus clipped-edge inequalities. Active I–B edges give (Sigma h_k)_b = theta + c_k h_kb on S_k; I–I equations add nothing (h_k^T Sigma h_k is implied). Measured rank of the measurement operator: 5–24 of 78 (block12); the I–I family alone has rank K. A natural estimator satisfying all active-edge equations for disjoint supports is the **parts-sum** C = theta 11^T + Σ_k c_k h_k h_k^T/|h_k|^2, with c_k = a member's summed input conductance: a nonnegative rank-K sketch.
**Scores** (`s6_readout.py`, `readout.json`, N = 120, gamma 1, wmin 1e-4, 2 seeds; contrast-subspace cosine after PCP, scale fitted):

| cov, theta (K) | masked LS | parts-sum | identity | rank-K trunc. | untrained |
|---|---|---|---|---|---|
| block12, 0.03 (1) | −0.79 | 0.63 | 0.79 | 0.08 | 0.78 |
| block12, 0.35 (3) | 0.79 | 0.89 | 0.79 | 0.89 | 0.77 |
| block12, 0.65 (12) | 0.79 | 0.79 | 0.79 | 1.00 | 0.78 |
| hier8, 0.55 (4) | 0.79 | 0.90 | 0.82 | 0.95 | 0.81 |
| powerlaw_pos12, 0.25 (2) | 0.84 | 0.95 | 0.61 | 0.95 | 0.52 |
| powerlaw12 signed, 0.18 (3–4) | 0.76 | 0.83 | 0.71 | 0.95 | 0.69 |

Bures–Wasserstein (PSD-clipped, scale-fitted, contrast setting) never beats identity at equilibrium (block12, 0.35: 0.52 vs 0.10): the sketch misses all sub-threshold variance. In the known setting, masked LS has relative Frobenius error 0.43–0.88 (identity 0.60–0.83).
**Transient is better.** Applying the equilibrium formula along training (residual 1e-3–4e-3 at the peak), block12 reaches contrast cosine 0.90–0.95 and BW 0.03–0.28 at steps 120–160 (theta 0.05, 0.35, 0.45; 3/3 seeds each), then falls to −0.79/0.79/0.79. No peak for theta 0.65 or the signed power-law (max 0.78).
*Labelled comparison:* mask p = 0.5 raises the equilibrium rank to 42 and gives contrast cosine 0.94, BW 0.03 (block12, 0.35, 2/2 seeds); p = 0.25 and the signed power-law do not improve.
**Verdict:** only the supra-threshold structure of populated parts is recoverable. Quality peaks at intermediate theta *above the collective threshold* and transiently during training. For nonnegative block-like Sigma the parts-sum matches the best rank-K truncation; for signed Sigma it is well below it.
**Insist on in the canonical readout:** lateral equations gamma h_i^T C h_j = theta + (lam/eta) w_ij; theta fitted as a free offset in the contrast setting; report the measurement-operator rank (clones inflate the equation count, not the rank); offer the parts-sum estimator (it beats min-norm LS in every contrast row); edge active iff w > 1.5 wmin.

## Changes I would make to SPEC.md
1. §2.1: one stable part iff theta < min_bc Sigma_bc (F1.1); one cluster also whenever theta < gamma·kappa0 (F3).
2. §2.2: correlation clustering is exact for uniform blocks; in general, stable parts = strict local maxima of nonnegative PCA of Sigma − theta 11^T, populated subject to the gating and packing rules.
3. §2.3: add the stability criterion and c_k = summed input conductance.
4. §2.4: diversity is exactly zero with dense synapses.
5. §2.5: allocation is not an equilibrium property; report the transient law.
6. §1: state the floor as omega = N·wmin; default wmin 1e-4 for N ≤ 200; show kappa0 and gamma·kappa0 by the theta slider.
7. Stationarity: near-zero residual can be an unstable symmetric state; offer private jitter (~1e-3); shared minibatch noise cannot split clones.
8. §3: identifiable = {(h_k, c_k)}; add parts-sum and rank report; best readout mid-training.

## What the explorer should let the reader discover
**Defaults:** lam = eta = 1, dt = 0.05, gamma = 1, wmin = 1e-4 (N ≤ 200), W_IB ~ Exp(0.2), W_II ~ U(0, 0.03), mean-field, run to residual < 1e-5 (300–2600 steps). Show kappa0, gamma·kappa0, residual, dead count, c_k per cluster. All tours are in `tours.json` (N 90–200, seeds 0–2, residual < 1e-5 unless noted; K in 3/3 seeds unless stated).

**Chapter 1 (B = 3).** The reader should find: consensus without pruning; parts are vertices or edge midpoints; splitting depends on gamma; anticorrelation prunes without drift; unequal variances tilt allocation and theta above a variance extinguishes it; too large a floor kills everything.

| tour | Sigma | theta | gamma | wmin | N | outcome |
|---|---|---|---|---|---|---|
| consensus | 1 diag, 0.3 off | 0.1 | 1 | 1e-4 | 90 | K=1 at (1/3,1/3,1/3) |
| neutral-identity | I | 0 | 1 | 1e-4 | 90 | K=1 at population mean, e.g. (.28,.35,.37) |
| three-corners | I | 0.4 | 1 | 1e-4 | 90 | K=3, ≈34/30/26 |
| corners-collective | I | 0.2 | 1 | 1e-4 | 90 | K=1 at one vertex (theta < kappa0 = 1/3) |
| edge-midpoint | [[1,.6,0],[.6,1,0],[0,0,1]] | 0.5 | 1 | 1e-4 | 90 | K=2: (.5,.5,0) with 57–64 + vertex 2 |
| midpoint-gated | same | 0.65 | 1 | 1e-4 | 90 | K=2: ONE of vertices 0/1 + vertex 2 |
| midpoint-splits | same | 0.65 | 0.5 | 1e-4 | 90 | K=3 |
| anticorrelated | I − 11^T/3 | 0.02 | 1 | 1e-4 | 90 | K=3 |
| winner-takes-all | diag(1,1.5,2.5) | 0.2 | 1 | 1e-4 | 90 | all 90 at vertex 2 |
| independent-tilt | diag(1,1.5,2.5) | 0.2 | 0 | 1e-4 | 90 | ≈72–76 / 9–17 / 1–5 |
| partial-extinction | diag(1,1.5,2.5) | 1.2 | 1 | 1e-4 | 90 | K=2 (no vertex 0), ≈62/28 |
| floor-too-high | I | 0.6 | 1 | 1e-2 | 90 | all dead |

**Chapter 2.** The reader should find: theta is a resolution knob on parts (1→3→12; halves→pairs→vertices); parts are localized place fields on the ring while eigenvectors are Fourier modes; lateral plasticity delays splitting and enforces non-overlap.

| tour | Sigma | theta | gamma | N | outcome |
|---|---|---|---|---|---|
| blocks-one | block12 | 0.05 | 1 | 150 | K=1 |
| blocks-collective | block12 | 0.2 | 1 | 150 | K=1 (3 parts exist) |
| blocks-three | block12 | 0.35 | 1 | 150 | K=3, 41–59 each |
| blocks-twelve | block12 | 0.7 | 1 | 150 | K=12 (2–4 dead) |
| hier-halves | hier8 0.6/0.3/0.05 | 0.22 | 0.5 | 150 | K=2 halves |
| hier-pair-per-half | hier8 | 0.4 | 1 | 150 | K=2, one pair per half |
| hier-pairs | hier8 | 0.57 | 1 | 150 | K=4 (res ≤ 6e-6) |
| ring-tiling-independent | ring16, ell 1.5 | 0.4 | 0 | 160 | K=16 width-4 bumps (res ≤ 5e-6) |
| ring-packing | ring16 | 0.4 | 1 | 160 | K=3–4 non-overlapping |

**Chapter 4.** The reader should find: allocation follows a steep theta-dependent power of part energy only in the independent regime; the collective regime is winner-take-all; the floor matters through N·wmin; nothing makes members diverse.

| tour | Sigma | theta | gamma | wmin | N | outcome |
|---|---|---|---|---|---|---|
| filter-steep | diag(.5,.75,1,1.5,2,3) | 0.05 | 0 | 1e-4 | 200 | 182–190 on v=3, rest v=2 |
| filter-soft | same | 0.4 | 0 | 1e-4 | 200 | 5 inputs, ≈106–119/40–51/16–25/14–15/5–12 |
| filter-collective | same | 0.2 | 1 | 1e-4 | 200 | 193–196 on v=3 |
| floor-leak | block12 | 0.65 | 1 | 1e-3 | 150 | all dead |
| floor-leak-ok | block12 | 0.65 | 1 | 1e-5 | 150 | K=12 |

Precomputed figures for chapter 4: `phase.json` (K over theta×gamma with the line theta = gamma·kappa0; theta×wmin; N), `alloc.json` (planted curves with seed bands), `alloc_limit.json` (beta vs theta), `diversity.json`, `hysteresis.json` + `hysteresis_jitter.json`, `readout.json`.

## Inconclusive / not done
- The extinction threshold's dependence on omega is empirical (no derivation).
- The coarse-to-fine gating rule is a heuristic that predicts all tested transitions within 0.03–0.08; it was not tested quantitatively on power-law covariances.
- Allocation under private per-neuron noise was not measured (only shared minibatch noise).
- Readout using known dw/dt (the exact non-equilibrium relation) was not tried.
