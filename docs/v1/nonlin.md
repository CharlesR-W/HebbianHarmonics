# Linear circuit, rectified output

> Archived v1 document (moved to `docs/v1/` on 2026-11-17). Code and data paths it cites (`hebb/sweeps/`, `hebb/mnist/`, `hebb/nonlin/`, `site/data/...`) refer to the tree at commit `f613d68`; git history keeps them.

*(Code: `hebb/nonlin/`. Data: `site/data/nonlin/` (format in its README.md). Runs:
Modal `crw-hebbian-nonlin`, run ids `20261116-231220-v2_14`, `-231222-v2_28`, `-232401-v2_g0`.)*

## 0. Summary: what rectified output adds and what it does not

**Adds, on real pixels: a preference for heavy-tailed projections (projection pursuit).**
At the same θ quantile, the rectified rule converges to a different part from the linear
rule. On MNIST, the linear rule's dominant part is the central slanted stroke. The
rectified rule's is two vertical bars at the left and right, the sides of a "0".

| | cos(part, linear part), q = 0.946 / 0.967 / 0.980 | skewness of the part's voltage | excess kurtosis |
|---|---|---|---|
| linear rule | 1 | 0.34 to 0.49 | −0.23 to −0.98 |
| rectified, τ = 0 | 0.65 / 0.50 / 0.11 | 1.37 to 1.54 | 1.5 to 1.8 |
| rectified, τ = 0.5 | 0.52 / 0.34 / 0.05 | 1.6 to 1.9 | 2.4 to 3.1 |
| rectified, τ = 1 | 0.40 / 0.24 / 0.03 | 1.9 to 2.1 | 3.4 to 4.4 |

The skewness and kurtosis ranges are over all three q.

- The effect replicates across seeds (τ = 0, seed 1: cosines 0.64 / 0.48 / 0.09).
- It replicates with γ = 0.
- It replicates at 28×28, N = 3920: skewness 1.56–1.58 vs 0.43–0.50, cosine to the linear part 0.05–0.10.
- It vanishes on the Gaussian surrogate. The same rectified rule on Gaussian samples with the pixels' mean and
  covariance, at its own θ quantile, learns the linear part (cosine 0.997–0.998, skewness 0.37–0.53).

This is the classical result that nonlinear Hebbian rules find non-Gaussian directions,
here appearing inside the resistor-network model. It is a genuine use of higher moments,
which no kernel lift was needed for.

**Adds: much larger rate covariances than the Gaussian theory predicts, so more binding and
later extinction.** On real digits the rate variance of a learned part is 1.6–1.7× (τ = 0),
2.2–2.4× (τ = 0.5) and 3.7–4.1× (τ = 1) the Gaussian dual-activation value for its voltage
variance. That is because the part was chosen to be skewed. The excess drives stronger
lateral binding:
- the rectified network stays ONE cluster up to q = 0.98, where the linear rule has 3–6
  clusters;
- at τ = 1 all 980 neurons are still alive at q = 0.987, where the linear rule has lost
  962–968 of them;
- the Gaussian surrogate with identical hyperparameters loses 931 of 980 neurons at
  q = 0.967 and dies entirely at q ≥ 0.98.

**Does not add: a better code, more clusters, or a different allocation law.**
- Every trained network, under either rule, is a highly redundant code with 1–8 distinct
  features. The rectified network is the more redundant one.
- Linear probes on N = 980 rates score 0.20–0.25 (rectified, one cluster) and 0.38–0.57
  (linear, 3–8 clusters). For comparison: the raw pixels 0.925, the UNTRAINED network's
  rates 0.62, and 980 random ReLU features 0.97.
- The rectified population is not a more useful code here; the Hebbian redundancy
  dominates.
- Allocation (neurons per cluster vs energy) cannot be compared. The rectified rule never
  populates two or more major clusters on MNIST, and the linear rule's major clusters have
  nearly equal energies (0.97–1.12), so the fitted slopes (−7 to +3.6) mean nothing.

**For Gaussian inputs it adds nothing beyond Σ_g (exact).** The rule becomes
⟨h_i,h_j⟩_Σ → ‖h_i‖_Σ‖h_j‖_Σ k_τ(cos_Σ(h_i,h_j)) with a known convex k_τ. That changes the
fixed-point geometry (synthetic: more and narrower clusters at the same pairwise threshold,
§2.1), not the information available.

**Readout: survives under the Gaussian assumption, and is biased on real data by exactly the
measured gap.** Inverting the dual activation recovers Σ on synthetic Gaussian data
(contrast cosine 0.9995 when the clusters span the blocks). On MNIST the recovered energy of
the learned part, c_kᵀĈc_k/c_kᵀΣc_k, compares with the measured rate-variance excess
Var_emp(r)/(k_τ(1) s²) as follows:

| | recovered energy ratio | measured excess |
|---|---|---|
| τ = 0 | 1.64 / 1.68 / 1.73 | 1.61 / 1.65 / 1.71 |
| τ = 0.5 | 2.22 / 2.33 / 2.43 | 2.19 / 2.29 / 2.38 |
| τ = 1 | 3.72 / 3.91 / 4.17 | 3.65 / 3.85 / 4.10 |

Controls: networks trained on the Gaussian surrogate give 0.998–1.03 (most within 0.2%); the
linear rule's own readout gives 0.99–1.01. Inverting the exact expected rate covariance instead of the
noisy SGD weights gives the same bias (1.61 / 1.65 / 1.70 at τ = 0), so the bias comes from
the closure, not from non-equilibrium. Matrix-level scores (contrast cosine 0.20–0.34 for
the rectified rule vs 0.38–0.43 for the linear rule, with the diag baseline at 0.52) mainly
reflect identifiability: a single cluster observes little beyond its own part.

The base model sees only second moments: interior voltages are linear in the clamped
inputs, u = H g, so Cov(u_i, u_j) = h_iᵀ Σ_g h_j and the Hebb rule needs a kernel lift
to reach anything beyond Σ_g. This extension keeps Kirchhoff linear, so u_i = h_i·g exactly
and the harmonic embedding is still a function of the weights only. What changes is that
plasticity, and any downstream reader, sees the firing rate r_i = f(u_i − b_i) with
f = ReLU instead of the voltage:

    dw_ij/dt = w0 − λ w_ij + γη Cov(r_i, r_j)        (I–I)
    dw_ib/dt = w0 − λ w_ib +  η Cov(r_i, s_b)        (I–B),   s_b = f(g_b − b_b)

The kernel lift is dropped: g is the raw stimulus (MNIST pixels, 14×14 so B = 196; a
28×28 check uses B = 784). The question is whether the rectification itself gives the rule
access to higher moments, and whether that changes what the network learns.

## 1. Engineering choices (the controversial ones, made explicit)

| choice | default | alternatives | why / what it changes |
|---|---|---|---|
| where the threshold sits | b_i = E[u_i] + τ·sd(u_i), τ ∈ {0, 0.5, 1}; τ = 0 is the default | b_i = 0; b_i = median; a fixed b | Voltages are convex combinations of the inputs (maximum principle), so with nonnegative pixels f(u) = u and **nothing changes at b = 0**. The nonlinearity only bites relative to a threshold. Rectifying deviations from the mean (τ = 0) is the minimal version; τ > 0 is a sparsity knob: a neuron fires for a minority of stimuli (fraction 50%/31%/16% for a Gaussian voltage at τ = 0/0.5/1). Setting the threshold in units of the neuron's OWN standard deviation is what makes the Gaussian theory a single scalar function of the correlation (§2). |
| what the threshold is | intrinsic-excitability homeostasis | — | b_i depends on the neuron's own activity statistics, so it is homeostasis of excitability, not a synaptic regularizer. The synaptic regularizer ρ(w) = w0 − λw stays a function of the weight only, as the user wants. We use the fast-homeostasis limit: b_i is computed from the exact population moments of u_i = h_i·g at the current weights (E u_i = h_i·μ, Var u_i = h_iᵀΣh_i). A slow running average would add a third timescale; not studied. |
| boundary neurons | treated the same: s_b = f(g_b − μ_b − τ σ_b) | linear boundary rates | Symmetry, and so that the I–B and I–I families measure the same kind of quantity. |
| rate gain | one global constant c per τ, chosen so the mean boundary-rate variance on the pixel data is 1 (c = 1.46, 1.99, 2.91 for τ = 0, 0.5, 1) | none | Same convention as the sibling's pixel lift (mean pixel variance 1). It keeps θ, w_min and the init on the same numerical scale for both rules. |
| θ for the rectified rule | the same quantile q of the off-diagonal boundary covariance the rule sees: Σ_g for the linear rule, c²Cov(s_b, s_c) for the rectified rule | same raw θ | The linear θ values are the sibling's pixel grid {0.3, 0.45, 0.6, 0.75, 0.9}. These are the quantiles q = 0.946, 0.967, 0.980, 0.987, 0.992 of Σ_g's off-diagonal. Each rectified θ is the same q of its own boundary rate covariance (table: `hebb/nonlin/configs/theta_table14.json`). |
| Gaussian surrogate (B) | g ~ N(μ_g, Σ_g); identical rule, c and θ as A | θ at the surrogate's own quantile (B', also run for τ = 0) | B holds the rule fixed and changes only the input beyond its second moment. Gaussian pixels have smaller and less skewed rate covariances (q-matched θ is about half: 0.17 vs 0.29 at the first grid point, τ = 0), so B' is the "same relative threshold" control. |
| network | N = 5B = 980, dense generic random init (W_IB ~ Exp(0.1), W_II ~ U(0, ii_scale)), γ = 1, λ = η = 1, dt = 0.1, 3000 steps, batch 256, weights time-averaged over the last 20% | — | As the sibling's pixel runs. Their scaling conventions are kept: leak ω = N·w_min = 0.5 (w_min = 5.1e-4) and N·ii_scale = 102.4 (ii_scale = 0.104). A first grid with w_min = 1e-3 (ω ≈ 1) went from one cluster straight to extinction (§6). |
| linear rule on the same inputs | rule = linear, input = pixels, same code path, same init seeds | reuse sibling runs | The run is cheap (40 s), and running it here gives paired seeds and identical analysis. It was checked to reproduce the sibling's step exactly (test `test_linear_rule_matches_sibling_step`). |
| stationarity residual | max |E[dw/dt]| with the EXACT expectation: full 60k training set (pixels), closed form (Gaussian surrogate), H Σ Hᵀ (linear). Edges with w ≤ 2w_min and negative drift count 0 | strict floor w_min(1+1e-6) (also stored as `res_max_strict`) | Time-averaged SGD weights leave edges that sit at the floor most of the time slightly above it. Counting their full drift (≈ −θ) reports floor-toggling noise as non-stationarity. 2w_min is also the readout's "active" cutoff. |
| clusters | the sibling's headline definition: single-linkage components of L1(h_i, h_j) < 0.1. "Dead" neurons have every synapse ≤ 2w_min. Live clusters exclude a dead-majority cluster; "major" = live and ≥ max(2, 1% N) | tol 0.05 / 0.2 also stored | Same as `site/data/mnist`. |

## 2. Gaussian theory: the dual activation

**Claim (exact).** Let (u_i, u_j) be jointly Gaussian with standard deviations s_i, s_j and
correlation ρ. With rates r = f(u − E u − τ sd(u)),

    Cov(r_i, r_j) = s_i s_j k_τ(ρ),

where k_τ is ONE scalar function (the same for every pair, because the threshold is in units
of each neuron's own sd). With X, Y standard normal with correlation ρ, k_τ(ρ) =
E[(X−τ)+(Y−τ)+] − m_τ², where m_τ = E(X−τ)+ = φ(τ) − τΦ(−τ). It satisfies:

- k_τ'(ρ) = P(X > τ, Y > τ) = Φ₂(−τ, −τ; ρ) (Price's theorem): the slope is the probability that both neurons fire.
- k_τ''(ρ) = φ₂(τ, τ; ρ) > 0, so k_τ is increasing and **convex**.
- k_τ(0) = 0, k_τ(1) = Var((X−τ)+), k_τ(−1) = −m_τ².
- Integrating Price twice from ρ = 0 (and substituting s = sin t to remove the endpoint singularity):

      k_τ(ρ) = ρ Φ(−τ)² + (1/2π) ∫₀^{arcsin ρ} (ρ − sin t) exp(−τ²/(1 + sin t)) dt.

- At τ = 0 this is the degree-1 arc-cosine kernel minus the product of the means:
  k₀(ρ) = (sin φ + (π − φ) cos φ − 1)/(2π), φ = arccos ρ,
  = ρ/4 + ρ²/(4π) + ρ⁴/(48π) + …

Normalised, k̃_τ(ρ) = k_τ(ρ)/k_τ(1) is the rate correlation as a function of the voltage
correlation. It lies below the diagonal on (0, 1), and further below for larger τ: k̃(0.5) =
0.43, 0.38, 0.31 at τ = 0, 0.5, 1. Negative correlations are squashed toward k̃(−1) =
−0.47, −0.23, −0.10.

**So in the Gaussian case** the rectified rule is the base rule with the Σ-inner product
⟨h_i, h_j⟩_Σ replaced by

    ⟨h_i, h_j⟩_Σ  →  ‖h_i‖_Σ ‖h_j‖_Σ · k_τ( ⟨h_i, h_j⟩_Σ / (‖h_i‖_Σ ‖h_j‖_Σ) ),

and the I–B term (HΣ)_ib by the same expression with h_j = e_b. This is a known, monotone,
convex transform of the cosine in the Σ-geometry, rescaled by the two norms. It still
depends on Σ_g only. **For Gaussian inputs rectification gives the rule no information
beyond the second moments.** What it changes is the geometry of the fixed points.

Verified: k, k′ and the endpoints against Monte Carlo (10⁶ samples, τ ∈ {0, 0.5, 1}); the
mean-field rule against sampled rate covariances; the torch table and its exact backward
pass (`hebb/nonlin/test_nonlin.py`, 14 tests).

### 2.1 What it does to the fixed points (synthetic, mean-field, exact)

Is the Gaussian rectified rule just the linear rule with a remapped θ? Take unit-variance
inputs and a correlation threshold ρ*: the linear rule at θ = ρ* and the rectified rule at
θ = k̃_τ(ρ*) bind exactly the same input PAIRS. The run is `hebb/nonlin/synth.py`: a ring
covariance (B = 24, squared-exponential correlation of length 2), N = 96, γ = 1, mean-field,
2 seeds; residual ≤ 7e-4 everywhere except linear ρ* = 0.45, seed 1 (9e-3). Output:
`site/data/nonlin/synth_ring.json`.

| ρ* | linear K | rect τ=0 K | rect τ=1 K |
|---|---|---|---|
| 0.30 | 2, 2 | 2, 2 | 2, 2 |
| 0.45 | 3, 4 | 5, 5 | 4, 5 |
| 0.60 | 2, 3 | **10, 10** | 4, 5 |
| 0.70 | dead (1), dead | dead, dead | **5, 8** |
| ≥ 0.80 | dead | dead | dead |

It is **not** a remap. At the same pairwise threshold, the rectified rule populates more,
narrower clusters, and at τ = 1 it keeps clusters alive above the linear rule's extinction
point. The mechanism is the convexity together with the two norms. Interior neurons have
s_i < 1, because averaging lowers the variance. Partially correlated neuron pairs, which
are the bonds BETWEEN prospective clusters, are discounted by k̃(ρ)/k̃(ρ*) < ρ/ρ*. Perfectly
correlated pairs WITHIN a cluster need only s² > k̃(ρ*) instead of s² > ρ*. So the rule
prunes between-cluster bonds harder and keeps lower-variance clusters alive. This is the
same "coincidence detection" sharpening that k_τ'(ρ) = P(both fire) expresses. On the
block covariance of chapter 2 (12 inputs, 3 blocks; within 0.5, across 0.1) at θ = 0.4
with c² = 1/k_τ(1): τ = 0 gives the 3 block clusters, like the linear rule, while τ = 1
splits them into 9 single-pixel clusters.

### 2.2 The readout survives, by inverting the dual activation

On active edges (w > 2 w_min) the fixed point gives, with all parameters known:

    γ c² s_i s_j k_τ(ρ_ij) = θ + (λ/η) w_ij,    c² s_i s_b k_τ(ρ_ib) = θ + (λ/η) w_ib,
    s_i² = h_iᵀ C h_i,  s_b² = C_bb,  ρ_ij = h_iᵀ C h_j/(s_i s_j),  ρ_ib = (HC)_ib/(s_i s_b).

**What is identifiable.**

1. *Scale.* Every equation is homogeneous of degree 1 in C. With (θ, λ/η, γ, c, τ) known the scale is fixed; with λ/η or c unknown, C is identified only up to scale, as in the linear case.
2. *Variances.* There are still no self-edges, but the variances now enter EVERY equation. Linearising, d[s_i s_j k(ρ)] = k′(ρ) d(h_iᵀCh_j) + [k(ρ) − ρk′(ρ)]·(s_j/2s_i · d(h_iᵀCh_i) + s_i/2s_j · d(h_jᵀCh_j)). At τ = 0 the coefficient is (sin φ − 1)/2π, which vanishes only at ρ = 0. So the variances cannot be "filled in" afterwards, the way the naive formula fills the diagonal with a row maximum: they must be solved jointly. In practice they are pinned the same way as in the linear model. Members of one cluster share h (ρ = 1), so their lateral edges observe s_k² k_τ(1) directly, and the I–B equations observe (HC)_ib. At the cluster level the measurement rows span the same matrices {sym(c_k c_lᵀ)} ∪ {c_k e_bᵀ} as in the linear model, only reweighted by k′. **The identifiable set is unchanged: the span of the populated cluster parts and their active pixels.**
3. *θ is no longer an exact gauge.* In the linear model, C → C + a 11ᵀ (a common random offset added to every input) shifts every Cov(u_i, u_j) by a, because H1 = 1, so an unknown θ is unidentifiable. After rectification that offset changes s_i and ρ_ij nonlinearly, so formally θ becomes identifiable. In practice it is not: fitting θ jointly with C on the synthetic ring returns θ̂ = −0.15 for a true 0.24 and θ̂ = −0.01 for a true 0.53 (`readout_fit_theta` in `synth_ring.json`), with equally good misfit. The loss is nearly flat along a mixed θ/C direction.
4. *Active edges only see the convex branch.* On an active edge the rate covariance exceeds θ > 0, so ρ > 0 and k′ ≥ k′(0) = Φ(−τ)². The inversion is well conditioned there. The anticorrelated branch, where k′(−1) = 0, would be ill conditioned, but clipped edges only give inequalities anyway.

**How we solve it.** Nonlinear least squares over symmetric C, both edge families, each
family's squared residual averaged over its equations and normalised by its mean Q², plus a
ridge α‖C‖²_F (α = 10⁻⁴ × the linearised operator norm × k′(1)², the sibling's relative
ridge). L-BFGS starts from the **chord** initialisation C₀ = LS-readout(Q / k_τ(1)).
Because k_τ is convex with k_τ(0) = 0, k_τ(ρ) ≤ ρ k_τ(1) on [0, 1], so the chord
underestimates the correlations. C₀ is also exactly what you get by ignoring the
nonlinearity, and is reported as the `chord` baseline. k_τ comes from an 8001-point table
with its exact derivative in the backward pass, and is extended linearly outside [−1, 1] so
non-PSD iterates stay differentiable. The alternative suggested in the brief (guess the
variances, invert to correlations, re-solve, repeat) stalls: when a guessed variance is too
small, Q/(s_i s_j) exceeds k(1), the inverse clips at ρ = 1, and nothing pushes the
variance up. The joint fit does not have this problem.

**The ridge matters more than in the linear case.** On the block covariance (3 block
clusters, mean-field, residual 1e-10) the readout recovers Σ with contrast cosine 0.9995 at
the default ridge. At a ridge of 10⁻⁷ the data are fit to 10⁻⁶ but the cosine is −0.36.
The data do not determine C within a block. The linear CG solver from 0 returns the
minimum-norm solution implicitly; the nonlinear solver does not, so the ridge is what
selects it.

**Test (synthetic Gaussian).** Mean-field, block covariance, τ = 0 (θ = 0.4) and τ = 0.5
(θ = 0.25), N = 48. Stationarity residual < 1e-8; relative misfit < 5e-3; contrast cosine
> 0.99 (`test_meanfield_stationary_and_readout_recovers_sigma`). When the populated clusters
do not span the blocks, nothing is identifiable, exactly as for the linear rule. At τ = 1,
θ = 0.4 the clusters are single pixels and the contrast cosine is 0.65–0.77.

## 3. What is new for non-Gaussian g

With real pixels, Cov(f(u_i), f(u_j)) depends on the joint law of (h_i·g, h_j·g) beyond its
covariance. We measure the gap directly. On every trained network, the empirical rate
covariance over all 60k training digits, E_ij, is compared with the Gaussian prediction
P_ij = c² s_i s_j k_τ(ρ_ij) from the exact voltage covariance. That isolates the part of what
the rule sees that is not a function of Σ_g. Reported per family (I–I neuron pairs, I–B,
B–B pixel pairs): ‖E − P‖_F/‖E‖_F, overall and on active edges; the gap in rate CORRELATION
against k̃_τ(ρ), which removes variance mismatch; and the variance ratio
Var_emp(r_i)/(k_τ(1) s_i²).

## 4. Experiments on MNIST

Setup: §1. 14×14 (B = 196), N = 980, 3000 SGD steps with time-averaged weights. The θ grid
is the sibling's linear θ {0.3, 0.45, 0.6, 0.75, 0.9} ↔ q = {0.946, 0.967, 0.980, 0.987,
0.992}. Rectified θ are at the same q of their own boundary rate covariance:
- τ = 0: 0.29, 0.42, 0.54, 0.69, 0.81;
- τ = 0.5: 0.27, 0.38, 0.51, 0.64, 0.76;
- τ = 1: 0.22, 0.34, 0.48, 0.62, 0.76.

Seeds: 2 for linear and rectified τ = 0, 1 otherwise. The same grid was also run with γ = 0,
plus 28×28 (N = 3920) at q = 0.967 and 0.980. 89 runs in total. To make θ interpretable,
`rule_cov_quantiles` in `index.json` gives the distribution of the covariances each rule
actually correlates. Interior–pixel entries have median ≈ 0.0003–0.005 and 99th percentile
0.8–1.7; interior pairs are a single value per cluster pair (e.g. 0.75 for the one cluster
at τ = 0, q = 0.946, against θ = 0.29).

### 4.1 Structure (live clusters / dead neurons, γ = 1, seed 0 [seed 1])

| q | linear | rect τ=0 | rect τ=0.5 | rect τ=1 | B: gauss, same θ, τ=0 | B′: gauss, own q, τ=0 | gauss, linear |
|---|---|---|---|---|---|---|---|
| 0.946 | 1 / 0 [1 / 0] | 1 / 0 [2 / 0] | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |
| 0.967 | 2 / 0 [2 / 0] | 1 / 0 [1 / 0] | 1 / 0 | 1 / 0 | 5 / 931 | 1 / 0 | 3 / 0 |
| 0.980 | 3 / 0 [6 / 807] | 1 / 0 [1 / 0] | 1 / 0 | 1 / 0 | all dead | 1 / 0 | 3 / 0 |
| 0.987 | 8 / 962 [7 / 968] | 1 / 966 [3 / 970] | 1 / 925 | **1 / 0** | all dead | 6 / 960 | 7 / 959 |
| 0.992 | all dead | 1 / 979 [dead] | 1 / 973 | 1 / 954 | all dead | all dead | all dead |

The generic dense init plus γ = 1 binds the population into one blob, as in the sibling's F3.
The linear rule fragments only on the way to extinction; this replicates the sibling's
pixel runs. The rectified rule holds the blob together longer and survives to higher θ,
especially at τ = 1. γ = 0 (no lateral plasticity) does not change the picture: on raw MNIST
nearly every neuron falls into the basin of the same dominant part (linear 1 / 3 / 7 live
clusters at q = 0.946 / 0.967 / 0.980; rectified still 1). 28×28: linear 3–4 live clusters
(3796 / 83 / 41 neurons at q = 0.967), rectified 2 (3919 + a singleton).

### 4.2 What is learned (the part)

The table in §0 gives the numbers. The images are `runs/<name>/parts_final.png`. In
`neurons_over_time.png` both rules pass through the same transient "0-ring" part (the
global Perron part). The linear rule stays there until step ~500 and then condenses onto
the central "/" stroke (and two small arcs at q = 0.98). The rectified rule leaves the ring
by step ~40 for two vertical bars.

The bars' voltage fires for 35–38% of digits at τ = 0, against 47% for the linear part
rectified post hoc. Neither part is digit-selective: both prefer "0" by mean response.
As a single feature, the rectified part is a worse digit predictor (1-feature probe
0.21–0.22 vs 0.27–0.28).

A vs B: with identical rule and hyperparameters (B), the surrogate learns a non-skewed part
(skewness 0.54) of different width (14.6 vs 21.8 pixels), loses 931 of 980 neurons at
q = 0.967, and dies entirely at q ≥ 0.98. At its own
quantile (B′) it learns the linear part. Everything the rectified rule does differently on
MNIST therefore comes from moments beyond the second.

### 4.3 The higher-order gap (§3), measured

| | rel. gap ‖E−P‖/‖E‖: I–I | I–B | pixel pairs | median E/P on active I–B edges | interior variance ratio | pixel variance ratio (q10 / q50 / q90) |
|---|---|---|---|---|---|---|
| rect τ=0, q=0.946 | 0.38 | 0.44 | 0.43 | 1.72 | 1.61 | 1.0 / 2.6 / 2.9 |
| rect τ=0, q=0.980 | 0.41 | 0.46 | 0.43 | 1.77 | 1.71 | (same pixels) |
| rect τ=1, q=0.980 | 0.76 | 0.78 | 0.77 | 4.2 | 4.10 | 0.45 / 7.8 / 14 |
| linear net, τ=0 post hoc, q=0.980 | 0.09 | 0.27 | 0.43 | 1.22 | 1.09 | 1.0 / 2.6 / 2.9 |
| gauss B′ net on real digits, q=0.980 | 0.09 | 0.27 | 0.43 | 1.11 | 1.10 | |

The gap is mostly variance: rectified skewed voltages have heavier upper tails than a
Gaussian with the same variance. The rate CORRELATION, as a function of the voltage
correlation, follows k̃_τ(ρ) closely (mean absolute deviation 0.01–0.03 on I–B and pixel
pairs; the scatter is in `dual.json`).

The rectified rule's own parts have the LARGEST gap (1.6–4.1× vs 1.1–1.2× for the linear or
surrogate parts). The rule moves to the directions where the Gaussian closure is worst: the
learning dynamics select for the gap.

### 4.4 Readout and allocation

Readout: §0 and `index.runs[*].readout`, `parts[*].energy_ratio_*`.
- With θ fitted, the joint fit returns θ̂ ≈ 0 (−0.002 to −0.07) for true θ of 0.22–0.54, with
  equally good misfit. θ is not identifiable in practice.
- γ = 0 nets (I–B equations only) are biased differently: an energy ratio of 1.28–1.41 vs a
  measured excess of 1.61–1.72. Pixel rates are themselves non-Gaussian (variance ratio
  median 2.6), so the I–B equations carry a different bias.

Allocation: not testable here (§0).

### 4.5 Sample synthesis (D)

`samples.png`: Gaussian samples x = μ + Ĉ₊^{1/2} z, sharing z across rows; μ is the data
mean (assumed, not read from the weights). Every weight-derived row is a blurry blob
concentrated on the single learned part. The rectified rows put their variance on the two
bars, the linear row on the central stroke. Neither resembles digits. This is the
identifiability limit of a one-cluster network, not a property of the rectification.

## 5. Costs

- Modal: $2.75 for the app (`modal billing report --for today`), plus $0.05 for two
  torch-import test apps.
- 135 units in total, 1.64 GPU-hours of unit time, on A10/T4 (the L4 queue was full):
  - v1: 50 units, superseded and not exported;
  - v2: 50 + 4 + 35 units;
  - smoke runs.
- Each 14×14 unit takes 25–70 s (dynamics 6 s, gap + readout + 40 probes the rest); each
  28×28 unit takes 3 min (linear) or 7 min (rectified).
- Local CPU work: pilots at N ≤ 256 (~15 min in total), the synthetic sweep (~6 min), and
  the export (~6 min, including UMAP).

## 6. Failures and caveats

- **First grid at w_min = 1e-3 (leak ω ≈ 1) was uninformative.** It went from one cluster
  straight to extinction, with only the q = 0.97 point showing a few small clusters. This is
  the sibling's F5. It was rerun at their ω = 0.5. The configs are kept in
  `hebb/nonlin/configs/v1/`; the outputs are not exported.
- **The θ sweep does not span "one cluster → fragmented".** With the generic dense init and
  γ = 1, MNIST pixels give one blob, then (linear only) a few clusters plus a growing dead
  pool, then extinction. γ = 0 did not change this. Wider cluster structure would need a
  different init (the sibling's heavy-tailed `ib_shape`) or a smaller leak; not run here.
- **Stationarity.** Time-averaged SGD weights of live networks have res_max 0.07–0.34
  (res_rel 0.11–0.44; active-edge mean 0.002–0.03). That is the minibatch noise floor, not the < 1e-4 of
  mean-field. The readout conclusions do not hinge on it: inverting the exact expected rate
  covariance gives the same bias as inverting the weights (§0).
- **The fast-homeostasis threshold uses the exact population moments of u_i.** A running
  estimate was not studied.
- **Modal.** The first launch sat 20 min in the L4 queue (switched to a GPU fallback list).
  The next launch failed all 50 units in a minute with corrupted `import torch` errors in
  the container. Those were infrastructure errors: the same image imported torch fine in
  isolation, and a redeploy fixed it.
- **The "alternate variances and correlations" inversion stalls** (§2.2), and **a small
  ridge gives a wrong readout** (§2.2). The joint L-BFGS fit with the sibling's relative
  ridge works.
- **UMAP and the I–I scatter are degenerate for one-cluster nets.** All points coincide at
  ρ = 1.

## 7. Code map

`hebb/nonlin/`:
- `dual.py`: k_τ, k_τ′, tables, the Gaussian rate covariance.
- `meanfield.py`: the exact Gaussian mean-field rule (numpy).
- `readout.py`: the NLS inversion; the k_τ autograd function.
- `model.py`: data, the torch SGD rule, the exact expected drift.
- `analysis.py`: the gap, probes, part energies.
- `train.py`: one unit.
- `synth.py`: §2.1.
- `make_configs.py`: θ tables and grids.
- `modal_job.py`.
- `export.py`.
- `test_nonlin.py`: run with `PYTHONPATH=.:<dir with pytest> /var/tmp/hebb-mnist-venv/bin/python -m pytest hebb/nonlin/test_nonlin.py`.

The sibling's `hebb/mnist/{analysis,readout}.py` is imported, not copied: the cluster
definition, the linear LS readout and the Scorer. The torch step is copied from
`hebb/mnist/model.py` and tested to match it.
