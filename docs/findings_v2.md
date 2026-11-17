# Findings v2: on-site model, distance rule (answers to SPEC.md §3)

Model: SPEC.md §1; reference implementation `hebb/onsite.py` (not modified). Every number below is in a JSON under `site/data/sweeps2/`, produced by the named script. Run one with `uv run python -m hebb.sweeps2.<name>`, or all six with `hebb/sweeps2/run_all.sh` (about 16 min on 4 vCPU). Helpers are in `hebb/onsite_analysis.py`.

## Conventions

- **Two banks.**
  - Learning uses a training bank of S = 256 samples; fixed-point identities are exact for its uncentred `C_hat = GᵀG/S`.
  - Population claims use an independent evaluation bank of 4096 samples, or `Sigma_g` analytically in the linear case.
  - Seed s fixes the rng streams: attachments `[s,0]`, thresholds `[s,3]`, training bank `[s,1]`, evaluation bank `[s,2]`. Random covariance presets use a fixed seed (12345), so seeds vary the network and the bank, not the statistics.
- **Equilibrium.**
  - Claimed only when the projected residual `max|dw/dt| < 1e-6`. Runs stop at `max|ΔW|/dt < 1e-9`.
  - Euler step `dt = min(0.5, 1/α)`, which is stable at stable fixed points. The Hessian of J is αI plus a negative semidefinite part, because E*(W) is a minimum of functions linear in W.
- **Three error types** are reported separately: fixed-point/solver, sampling, and inverse-problem.
- **Clusters** are connected components of the live graph (w > 1e-9), always reported with diameters:
  - `diam_max` / `diam_mean`: max / mean RMS distance D from a member to the prototype;
  - `span`: the largest pairwise D inside the component;
  - `span/sqrt(2b) > 1` marks a chain, i.e. a component wider than one edge's range.
- **Exactness labels:**
  - **derived**: algebra checked numerically;
  - **identity**: verified to solver or rounding precision;
  - **empirical**: measured on the tested settings, with seeds and spread.
- **Default rectifying potential:** klo = 3, khi = 0, σ_t = 0.7 (a leaky rectifier with a 4:1 gain ratio).

---

## Q-fusion (`fusion.py` → `fusion.json`)

**G1. The fixed point is the stated one** (identity).

Settings: N = 40, B = 12, s = 3; covariances `blocks` (variances 2/1/0.5/0.25, within-block correlation 0.8) and `power`; linear and rectifying potentials; α ∈ {0.1, 1}; 15 values of b from 0.005 to 2; three protocols (cold start from W = 0, continuation up, continuation down); 5 seeds for linear, 3 for rectifying.

- 1440 of 1440 runs converged; the largest residual was 1.0e-9.
- Live edges satisfy `αw = b − D²/2` to 1e-9.
- Dead pairs satisfy `D² − 2b ≥ 1.4e-5`.
- A central-difference `dE*/dw` matches `D²/2` to 6e-10.

Field: `summary`.

**G2. Two-neuron calibration** (derived, verified).

Setup: `D²(w) = D0²/(1 + 2w/deg)²`. Write c = D0²/2 and κ = α·deg.

- w = 0 is stable for b < c and unstable for b > c. At b = c the case is degenerate.
- The saddle-node sits at `y* = (4c/κ)^{1/3}`, with `b_sn = 3·2^{-4/3}(cκ²)^{1/3} − κ/2`.
- Bistable window: `b_sn < b < c`, non-empty iff **c > κ/4**.

Checks:
- On 120 (D0², α, b) grid points, runs from w0 = 0 and from w0 = b/α end on the analytic branches to within 2.7e-8.
- A continuation loop (D0² = 2, α = 0.2) switches on at b = 1.02 (theory 1.0) and off at 0.32 (theory 0.307), on a grid of step 0.02.

**The SPEC example depends on an unstated D0².** For unit-variance uncorrelated inputs, D0² = 2, and b = 0.2, α = 0.1 is monostable (b_sn = 0.2065 > 0.2). The stated pair w = 0 and w = 1.754 (unstable point at 0.378) needs D0² = 1, or b = 0.4, α = 0.2 with D0² = 2; the latter is what `tests/test_onsite.py` uses.

Fields: `pair.*`.

**G3. N neurons: no regime of a few clean clusters** (empirical).

Cold-start K falls from about N (b ≲ 0.008) to 1 (b ≳ 0.3). In between, multi-member components peak at 6–8 while many neurons stay singletons, and then a giant component grows.

`blocks`, linear, α = 1 (5 seeds, mean ± sd):

| b | 0.018 | 0.028 | 0.042 | 0.065 | 0.10 | 0.15 | 0.24 |
|---|---|---|---|---|---|---|---|
| K | 31.8±2.1 | 26.2±1.6 | 20.6±1.4 | 12.4±2.2 | 4.6±1.7 | 3.4±0.8 | 1.4±0.8 |
| multi-member K | 4.8±1.3 | 6.4±0.8 | 7.2±1.0 | 5.4±1.4 | 3.0±1.7 | 2.4±0.5 | 1.2±0.4 |
| largest | 4.2±1.3 | 6.8±0.8 | 10±1.9 | 16.8±6.9 | 27.8±8.7 | 29.4±4.8 | 39.4±1.2 |
| diam_max | 0.13 | 0.23 | 0.29 | 0.48 | 0.76±0.19 | 0.36 | 0.09 |
| span/√(2b) | 1.2 | 1.6 | 1.6 | 2.1 | 2.7±0.6 | 0.96 | 0.22 |

- **Chains.** Components reach 3.65 edge-ranges on the training bank and 3.60 on the evaluation bank.
- **Diameter is non-monotone:** it peaks at percolation, then contracts.
- **Generality.** `power` and the rectifying potential look the same. With random s = 3 attachments the 4-block structure gives no 4-cluster plateau.
- **Sampling.** Over multi-member components, diam_max differs between the training and evaluation banks by a median of 2.3% (90th percentile 8.4%, max 17%). Pairwise `|ΔD²|/max D²` is 6–14%.

**G4. α sets tightness and hysteresis, not onset** (empirical).

- **Onset.** At small b, cold-start K is the same for α = 0.1 and α = 1; activation depends only on `D0² < 2b` (G7).
- **Tightness.** At α = 0.1 components are tight: `blocks` linear at b = 0.1 has diam_max 0.062 ± 0.007, against 0.76 ± 0.19 at α = 1.
- **Hysteresis at α = 0.1 is large.** The down-continuation stays one component down to b ≈ 0.03, where the cold and up branches have 20–30 components. Up/down partitions differ for b ∈ [0.028, 0.24] (`blocks`) and [0.028, 0.55] (`power`).
- **Hysteresis at α = 1 is weak:** b ∈ [0.10, 0.36], pair-mismatch ≤ 0.44.
- **Pair criterion.** At fixed b, the two-neuron rule says hysteresis needs b > κ/4. That predicts b > 0.025 at α = 0.1 (matches), but b > 0.25 at α = 1, whereas the network shows it from 0.10.
- **Which branch is lower.** Within the window the fused branch usually has the lower J (by up to 80 units, `power`, linear, α = 0.1). At b = 0.028 it is the reverse (−2.0 and −3.4). The cold-start state is generally not the minimiser of J.

**G5. Refinement is nested except near percolation** (empirical).

Measure: the fraction of pairs co-clustered at the smaller of two consecutive b but split at the larger.

- Away from percolation it is 0.
- Near percolation the cold-start value reaches 0.48 (`blocks` linear, α = 1) and 0.28 (rectifying).
- Cold-start K is non-monotone in b in 3 of 16 α = 1 seed-runs.
- Along the down-continuation it is ≤ 0.10 (0.20 for `power` rectifying, α = 0.1).

Fields: `nest_cold`, `nest_down`, `part_diff_up_down`.

---

## Q-resolution (`resolution.py` → `resolution.json`)

Inputs: B = 8 with a Hadamard eigenbasis. The uniform mode has λ = 1; contrast modes have λ = 4, 2, 1, 0.5, 0.25, 0.125, 0.0625. Contrast modes are needed because Δh·1 = 0 in the linear model.

**G6. Statement (1), the frozen law, is exact for C_hat** (identity).

- `D² = Σ_a λ_a (Δh·v_a)²` with C_hat's eigenpairs holds to 1.4e-15 relative to max D² (W = 0 and random W; 2 covariances; 5 seeds).
- With Σ_g instead, the error is sampling only: 17.5% at S = 256, and 4.9% against evaluation-bank D².
- `max|ΣΔh| = 8e-16`.

Field: `p1`.

**G7. Statement (2), activation from the uncoupled state, is exact** (derived, verified).

The edge from W = 0 lives iff `δ < sqrt(2b/λ̂_a)` with `λ̂_a = v_aᵀ C_hat v_a`. Test: 595 isolated planted pairs (7 modes × 17 separations × 5 seeds, b = 0.005).

- The λ̂ threshold misclassifies 0 pairs; the Σ_g threshold misclassifies 30.
- A homogeneous rectifier scales the uncoupled D0² by 0.42–0.53, which moves the δ threshold out by about 1.45×.

Field: `p2`.

**G8. Statement (3a): a pair loses fusion at the saddle-node** (derived, verified).

At fixed b, hysteresis requires b > κ/4. A fused pair then survives up to `δ_off = sqrt(32(b+κ/2)³/(27κ²λ))`, while it first fuses at `δ_on = sqrt(2b/λ)`.

The ratio `δ_off/δ_on = sqrt(16(b+κ/2)³/(27κ²b))` does not depend on λ:
- 1.41 at α = 0.1, b = 0.1;
- 8.28 at α = 0.01, b = 0.1;
- 1 at α = 1, b ≤ 0.1.

Simulated planted pairs (modes 1 and 4; c = 0.01; α = 0.005 / 0.01 / 0.02 / 0.04):
- switch-on at b = c, within one grid step (1.4e-4);
- switch-off simulated at 0.0051 / 0.0070 / 0.0090 / 0.0100, against theory 0.0050 / 0.0069 / 0.0089 / 0.0100;
- identical for both modes.

Field: `p3a`.

**G9. Statement (3b): feedback shifts the resolution both ways** (empirical, plus one exact scaling).

Setup: 30 linear neurons spread uniformly along one mode (λ̂ ≈ 3.8 or 1.0); 14 values of b; α ∈ {0.01, 0.1}; 5 seeds.

- **Cold start is finer than (2).** Self-consistent K is never below the prediction from the positive-initial-drift graph (280 of 280 cases), and is up to 6 above it (mode 1, α = 0.01, b ≈ 0.006: 6.4 vs 3.2).
- **Feedback breaks chains, never bridges them.** No component bridges an uncoupled gap larger than δ_on (max ratio 0.997).
- **Condensation.** Where the uncoupled graph is nearly connected (predicted K of 1–1.8), the α = 0.01 fixed point has 2–5 tight components (span/√(2b) ≤ 0.5 on both banks). This is the only setting where discrete clusters nucleated from a continuum.
- **Down-continuation is coarser.** For mode 3, α = 0.01, it gives K = 1 for b ≥ 0.006, where the cold start gives K = 3.4.
- **Exact scaling.** On a single-mode line the fixed point depends only on (b/λ̂, α/λ̂); runs on modes 1 and 3 give the same W to 3e-8.

Spans on the evaluation bank are within 6.5% of the training-bank values. Fields: `pop`, `scaling`.

---

## Q-diversity, T1 (`diversity.py` → `diversity.json`)

Setup:
- Rank-one drive g = z·v, with v positive. N = 40, B = 12, s = 3, matched attachments.
- Three arms that differ only in the potential:
  - hetero: klo = 3, khi = 0, σ_t = 0.7;
  - homo: the same with t = 0;
  - linear.
- 7 values of b from 0.003 to 0.3; α ∈ {0.1, 1, 10}; 5 seeds. 570 of 570 runs converged.
- Measures on the evaluation bank (fit on one half, score on the other):
  - `affine_err`: held-out error of an affine fit of each response from the full input;
  - `spline_err`: held-out error of a 1-D spline fit from z;
  - `proto_err`: held-out error of `member ≈ a + c·prototype`;
  - d_eff and numerical rank, descriptive only.

**G10. Each response is a function of the shared mode** (derived). With rank-one drive, the unique equilibrium is a function of z. The spline residual is ≤ 0.004 everywhere, which is approximation error.

**G11. Linear and homogeneous potentials give no diversity** (derived, verified).

- **Linear:** responses are exactly c_i·z (affine error 1e-29, rank 1).
- **Homogeneous:** the node equations are positively homogeneous, so the response family has rank ≤ 2 (span{z₊, z₋}); at W = 0 it is exactly c_i·φ(z). Each neuron is nonlinear (affine error 0.116), but members nearly collapse onto the prototype (proto_err ≤ 1.5e-4, d_eff = 1.000).

**G12. Heterogeneous thresholds give real but very small diversity, and fusion suppresses it.** This is the decisive test of T1, and it largely fails.

- Per neuron, the affine error is 0.096 ± 0.009 at W = 0 and falls with fusion (0.05 at α = 0.1, b = 0.3).
- Within clusters:
  - proto_err ≤ 0.023 (seed mean; single runs ≤ 0.028);
  - d_eff 1.00–1.03;
  - numerical rank up to 16.

  So members are many slightly different functions whose differences carry about 2% of variance.
- **Fusion strength matters.** At α = 0.1, proto_err falls from 0.014 (b = 0.014) to 1e-5 (b = 0.3). At α = 10 (weak fusion) it stays near 0.02.
- **Stronger or reshaped potentials** (default b = 0.03, α = 1; field `strength`):
  - klo = 30 raises per-neuron nonlinearity to 0.25 and proto_err to 0.068 ± 0.017 (σ_t = 0.7). Individual seeds reach 0.16 (klo = 30, σ_t = 1.4); d_eff ≤ 1.08.
  - A compressive-above-threshold potential (klo = 0, khi = 3) gives 0.011–0.028.
- **Verdict.** The "many random features per mode" picture is not delivered by the basic model.

**G13. Multi-mode inputs** (empirical; 4 planted block scalars).

- **Block attachments, noiseless blocks (ρ = 1).** Each neuron is a function of its own block's scalar (spline residual 1e-4 for b ≤ 0.055). Merging across blocks raises it to 0.35 at b = 0.3.
- **Random attachments.** Neurons mix modes from the start (0.19), and fusion increases mixing (0.43).
- **Within-cluster d_eff of 1.3–2.8** at intermediate b comes from cross-mode mixing in chains; the linear arm shows it too (2.0).
- **Private noise (ρ = 0.8)** adds a floor of about 0.12.

---

## Q-eigen, T3 (`eigen.py` → `eigen.json`)

**G14. Linear case: the population operator is Σ^{1/2} HᵀH Σ^{1/2}** (derived, identity).

On linear functionals θ·g the operator acts as K·Σ with K = HᵀH.
- The eigenfunctions are Σ's modes iff [K, Σ] = 0 (with the usual basis caveat for degenerate eigenspaces); then q_a = λ_a·k_a.
- The spectra agree to 3e-15.
- The evaluation-bank spectrum is within 1–4% of the analytic one.
- Trained networks never make K commute with Σ: relative commutator 0.11–0.42.

**G15. The explicit shared + contrast example** (derived, verified).

Setup: g1 = s + c, g2 = s − c, var s = 1, var c = sc2. The two clusters have prototypes s ± c with cosine ρ = (1 − sc2)/(1 + sc2). The spectrum `ν/2 (n_A + n_B ± sqrt((n_A − n_B)² + 4n_A n_B ρ²))` matches the analytic HΣHᵀ exactly when members are identical.

- **Equal sizes (12/12).** The eigenfunctions are exactly s and c (|cos| = 1.000), which are Σ's modes, and not the prototypes (best match 0.95 / 0.31 at sc2 = 0.1).
- **Unequal sizes.** ψ₁ tilts toward the larger cluster's prototype: cos(ψ₁, s) = 0.975 at 20/4, sc2 = 0.1, and 0.85 at sc2 = 0.5.
- **Orthogonal prototypes (sc2 = 1) with unequal sizes.** The eigenfunctions are the prototypes and q = ν·n exactly: (32, 16) and (40, 8).

**G16. Trained networks** (empirical; N = 48, 3 seeds; diameters on the evaluation bank).

- **Linear, block attachments on uncorrelated blocks, b = 0.048–0.23.**
  - 4 clusters (diam_max 0.11–0.38 against prototype sd ≈ 0.84).
  - Prototypes orthogonal (|cos| ≤ 0.024).
  - Eigenfunctions match both the prototypes (0.999) and Σ's modes (0.997).
  - q ≈ ν·n to 0.2–2.7%.
  - This is the note's picture, but the attachments build it in.
- **Linear, cross-block correlation 0.4, b = 0.048 and 0.105 (4 clusters; K = 3 by b = 0.23).**
  - Prototype |cos| ≈ 0.47.
  - Eigenfunctions match Σ's modes (0.996) and not the prototypes (0.78).
  - q ≈ ν·n fails by 30%; the cluster-Gram spectrum fits to 1–3%.
- **Random attachments.**
  - At intermediate b the eigenfunctions match neither the prototypes (0.60–0.81) nor Σ (0.72–0.87).
  - At percolation even the cluster-Gram prediction fails (25–110%).
- **Rectifying arm.** Cluster counts shift (no 4-cluster interval). The top eigenfunctions are 1–25% nonlinear.

**Answer.** Prototypes are near-orthogonal, and then q_a ≈ ν·n_a, only when clusters draw on uncorrelated input groups. Otherwise the eigenfunctions are shared + contrast mixtures, and for tight clusters q is the spectrum of `(νn)^{1/2} PᵀP (νn)^{1/2}`.

---

## Q-allocation, T2 (`allocation.py` → `allocation.json`)

**G17. (a) In the base model, allocation is set by attachments and the fusion scale, not n(λ)** (empirical; block attachments; α = 1; 5 seeds).

- Each block keeps exactly its attached neurons; no component spans two blocks.
- Raising one block's variance from 0.25 to 8 fragments it rather than recruiting neurons: at b = 0.02 it goes from 1 component of 12 to 8.2 ± 1.9 components of mean size 1.5.
- Changing the number of neurons attached (4–24) changes cluster size one for one.
- With random attachments, component size is anti-correlated with prototype variance (Spearman −0.70 at b = 0.1, −0.31 at b = 0.05). This is a resolution effect, not an allocation law.

**G18. (b) Plastic attachments under the energy rule drain to quiet inputs** (empirical; confirms SPEC §2.2).

Setup: block variances 2/1/0.5/0.25, dense initial rows, fixed degree, N = 40, 5 seeds.

- **α_A ≥ 1, with or without lateral learning or the rectifier.** 37–40 of 40 neurons have a majority of their mass on the 0.25 block (mean share 0.76–0.97); all runs converged.
- **α_A = 10.** The ridge keeps rows spread over all 12 inputs: the quiet block is every neuron's largest share, but only 0.33 of the mass, and no neuron has a majority there.
- **α_A = 0.1.** Converged runs have 20–27 majorities; one run did not converge (32, residual 2.6e-5).
- **Mechanism** (derived; W = 0, linear). The attachment energy is (deg/2) times the attachment-weighted variance of the input mixture. Pure attachments cost 0, the ridge pays for spreading, and spreading is cheapest over quiet inputs.

**G19. (c1) The added principle obeys the water-filling mass law** (derived, verified).

The principle: `½‖Σ − XᵀX‖²` with hard |x_i|² = ν, labelled as not derived from the conductance energy.

- **Derivation.** Relax to M ⪰ 0 with Tr M = Nν. The optimum is q_a = (λ_a − μ)₊. The relaxation is tight by Schur–Horn when rank ≤ N, which holds here (N = 60 or 80 > B = 8 or 16). The hard constraint is what is plotted.
- **Numerics** (two spectra; ν ∈ {0.05, 0.1, 0.2}, i.e. μ = 1.17, 0.42, −0.43; 4 inits):
  - ‖q − q_pred‖₁/(Nν) ≤ 6e-9;
  - off-diagonal ≤ 3e-11;
  - objective excess ≤ 2e-10.
- **Neurons are mixed:** only 1–29% have purity > 0.9.

**G20. (c2/c3) No tested ingredient gives n_a ≈ q_a/ν** (empirical; 4 inits).

| lam8, ν = 0.1 | pure frac | count error (L1/N) | mass-law dev | objective excess |
|---|---|---|---|---|
| none | 0.09 | (mixed) | 0 | 0 |
| fusion b = 0.01 / 0.03 | 0.09 / 0.28±0.16 | 0.18 / 0.18 | 0 | 0 |
| fusion b = 0.1 | 0.48±0.22 | 0.38 | 0.34±0.08 | 0.69 |
| nonneg (eigenbasis) | ≥ 0.983 | 0.19±0.05 | 0.19±0.05 | 0.23±0.10 |
| nonneg (rotated basis) | 0.00 | — | 0.51 | 1.61 (off-diag 0.43) |
| nonneg + fusion 0.03 | ≥ 0.983 | 0.19 | 0.19 | 0.24 |

- **Fusion purifies only partly while it keeps the law** (b ≤ 0.03). At b = 0.1 it breaks the law and empties modes 4–8, where the law assigns 5.8 / 1.8 / 0 / 0 / 0 neurons.
- **Non-negativity in the eigenbasis.** If XᵀX is diagonal, X ≥ 0 forces disjoint supports, so pure neurons follow. The optimiser gets nearly all-pure neurons (one run has 98.3% pure, off-diagonal 0.048, and is stationary), but in local minima that over-populate weak modes: counts 20–24 / 14–16 / 10–11 / 5–6 / 3–5 / 1–3 / 0–2 / 0–1, against the integer optimum 26 / 16 / 11 / 6 / 1 / 0 / 0 / 0.
- **Basis dependence.** Nonneg is close to the law only at ν = 0.2 (error 0.04), and fails completely in any other basis.
- **pow16:** count error 0.18.

---

## Q-readout, T4 (`readout.py` → `readout.json`)

**G21. Linear, zero potential: P C_hat P is read out exactly whenever the live-edge operator has full rank. Rank is set by the live-edge count, not by fusion** (identity for the recovery; empirical for when it applies).

Setup: N = 40 and 80, B = 12, `blocks` and `power`, 8 values of b, 3 seeds, all converged.

- **The blind subspace is exact:** d·1 = 0 to 8e-16.
- **Rank tracks live edges.** The rank on the 66-dimensional contrast space follows the live-edge count until it saturates, though not always: one run has 122 edges and rank 64.
- **Errors, reported separately:**
  - fixed point ≤ 1.9e-9;
  - sampling ‖PC_hatP − PΣP‖/‖PΣP‖ = 0.09–0.10;
  - inverse problem on the identifiable subspace (singular values > 1e-4) ≤ 4e-3 overall and ≤ 1e-3 at full rank.

`blocks`, N = 40, Frobenius cosine vs P C_hat P:

| b | 0.003 | 0.016 | 0.036 | 0.083 | ≥ 0.19 |
|---|---|---|---|---|---|
| K (components) | 39 | 31.7 | 21 | 6 | 1–1.3 |
| live edges | 1 | 10 | 31 | 110 | 722–780 |
| ridge + PSD | 0.06 | 0.34 | 0.62 | 0.985 | 1.000 |
| hinge + PSD | 0.48 | 0.57 | 0.70 | 0.986 | 1.000 |
| identity baseline (fitted scale) | 0.36 | 0.54 | 0.54 | 0.54 | 0.54 |
| diag baseline (NNLS) | 0.10 | 0.38 | 0.47 | 0.51 | 0.57–0.60 |
| untrained random network | 0.20 | 0.26 | 0.32 | 0.37 | 0.46–0.47 |
| shuffled weights | 0.20 | 0.26 | 0.27 | 0.32 | 0.36–0.39 |

- **Full rank in fragmented networks.** At N = 80, full rank and cosine 1.000 arrive while the network is still fragmented: `blocks`, b = 0.036, 24–31 components; `power`, b = 0.083, 18–23 components, with chains 2.4–3.1 edge-ranges wide.
- **N = 40 needs near-percolation.** It reaches full rank only at K ≲ 7.
- **Against Σ:** cosine 0.996 (`blocks`) and 0.991 (`power`), limited by sampling.
- **Too few live edges.** The readout is no better than a fitted multiple of the identity (0.54 / 0.69); the dead-edge hinge helps a little.
- **W = 0** gives no measurements at all.
- **Weight noise** (multiplicative, with the whole readout rebuilt from the noisy weights):
  - fully fused regime, b ≥ 0.44: 1% noise gives cosine 0.22–0.63, because w ≈ b/α and the information sits in tiny deviations;
  - sweet spot just above percolation: N = 80, b = 0.083 keeps 1.000 (`blocks`) and 0.985 (`power`) under 1% noise.

**G22. Quadratic leak: H1 ≠ 1 is not enough** (derived, verified).

(L + K)1 = A1 + k, so if k_i/deg_i is constant, H1 = (1 + k/deg)⁻¹·1 and d·1 = 0 still.

- **Uniform leak k = 0.1 or 1.** |d·1| ≤ 6e-16. The rank is 66 of 78: the same blind subspace 1aᵀ + a1ᵀ. Full-C cosine plateaus at 0.796, while the contrast part is exact.
- **Heterogeneous leak k_i ~ U(0, 0.2) or U(0, 1).** |d·1| up to 0.16 / 0.49. Rank 78 of 78, and full C_hat is recovered (cosine 1.000) for b ≥ 0.083.
- **Fusion shrinks |d·1|** (to 0.004 at b = 1), so the 1-directions become noise-sensitive: 1e-4 weight noise gives cosine 0.63 at b = 1.

**G23. Nonlinear case: a partial distance geometry and nothing more** (rectifying, N = 40, 3 seeds).

- Live edges give the training-bank D to ≤ 4e-9. Against the evaluation bank they differ by 11–13%, which is sampling.
- Pairs in different components satisfy D ≥ √(2b): 100% on the training bank and ≥ 99.7% on the evaluation bank.

| b | 0.02 | 0.05 | 0.1 | 0.3 |
|---|---|---|---|---|
| exactly known | 0.012 | 0.053 | 0.31 | 0.95 |
| shortest-path upper bound | 0.005 | 0.13 | 0.63 | 0 |
| lower bound only | 0.98 | 0.82 | 0.07 | 0.05 |

- Shortest paths overestimate D by a median factor of 1.28–1.39 (max 3.3).
- Classical MDS of the largest component, completed vs true, has Procrustes disparity 0.19–0.22 on either bank (≤ 0.004 at b = 0.3).

---

## Contradictions with, or refinements of, SPEC.md

1. **Two-neuron example.** It requires D0² = 1, or b = 0.4, α = 0.2. With unit-variance inputs (D0² = 2), b = 0.2, α = 0.1 is monostable (G2).
2. **Quadratic leak.** "H1 ≠ 1" does not remove the blind subspace; that requires k_i/deg_i to vary across neurons (G22).
3. **Resolution (3).** Feedback makes the resolution finer from a cold start and coarser along the down-continuation. Statements (1) and (2) are exact only for C_hat (G6, G7, G9).
4. **§2.1.** The middle regime is many singletons plus chains plus a giant component, not "a few fused clusters" (G3).
5. **§2.2 and §2.3 are confirmed** (G18, G19, G20).

## What the explorer should let the reader discover

- **fusion.**
  - Two-neuron drift curve with the analytic window shaded, and two presets: "bistable" (D0² = 2, b = 0.4, α = 0.2: w = 0 and w = 1.754 both attract) and "SPEC example, unit variance" (b = 0.2, α = 0.1: only w = 0).
  - Live N-neuron view (B = 3, N ≈ 30), with presets "singletons" (b = 0.005), "chains at percolation" (b ≈ 0.07–0.1, α = 1) and "one blob" (b = 0.3).
  - A "hysteresis" preset at α = 0.1: drag b from 0.3 down to 0.03 and it stays fused; reset to W = 0 and it has about 25 components.
  - Always show the w-vs-D² scatter on the line αw = b − D²/2, and the residual.
- **many.**
  - K(b), largest(b) and diam_max(b) with cold/up/down branches and seed bands; nestedness bars; J for both branches.
  - A "condensation" preset: a line along one mode, α = 0.01, b ≈ 0.006 (about 3 components predicted, about 6 tight ones at the fixed point).
- **diversity.**
  - Member curves u_i(z) for the three arms (`presets`, b = 0.03, α = 1).
  - A toggle that maps each member affinely onto its prototype: hetero collapses to within about 2%, homo almost exactly.
  - proto_err vs b for α ∈ {0.1, 1, 10}, and the `strength` table.
- **eigen.**
  - Live s ± c example with sliders for var c and n_A/n_B: ψ₁ swings between s and the larger cluster's prototype; q shown against ν·n and the cluster Gram.
  - Two presets: "uncorrelated blocks" (ψ = prototypes = Σ modes) and "correlated blocks" (ψ = Σ modes ≠ prototypes).
- **allocation.**
  - A variance slider for one block: its neuron count stays fixed while its cluster count grows.
  - The drain of plastic attachments to the quiet block.
  - Water-filling plot with the μ line next to pure-neuron counts per ingredient: nonneg gives pure neurons with the wrong counts, rotated nonneg fails.
- **readout.**
  - Rank vs b against the 66 line, for N = 40 and N = 80; cosine vs b for each method and baseline.
  - A weight-noise slider showing the fused-regime fragility and the sweet spot at N = 80, b ≈ 0.08.
  - A uniform vs heterogeneous leak toggle.
  - An MDS overlay for the nonlinear case.

## Inconclusive / not done

- **Sizes.** N ≤ 80 and B ≤ 12; no B up to 24, no N-scaling.
- **Potentials and inputs.** Only the two-slope potential (no sinh-type saturating transfer) and only Gaussian inputs (no Laplace).
- **Knobs not swept:** s = 3 and deg = 1 were fixed.
- **Grid resolution.** Hysteresis windows are resolved only to the 15-point b grid; nestedness is checked only between consecutive b.
- **Allocation (c).** Only projected gradient from random inits. The integer-optimal pure allocation exists under nonneg but was not found; a global search might find it. NSM-style rectified similarity matching was not tried.
- **Eigen comparisons.** The q ≈ ν·n errors at small b are partly definitional (singletons count as clusters).
- **Training-bank geometry.** Fusion chain metrics are primarily training-bank values (evaluation-bank spans are reported beside them). The nonlinear readout's "true" D is the training bank, with evaluation-bank versions reported.
- **Not done (SPEC §4.7 list):** the stability Hessian (eq. 12), lagged statistics, a generative phase, real data.
