# Allocation under private noise (F11–F17)

Script: `hebb/sweeps/s8_private_noise.py` (parts A R M P X DT L N). Modal runner: `hebb/sweeps/s8_modal.py` (`S8_MODAL=1`); without it the parts run locally on 4 processes. Raw output: `scratch/out/sweeps/s8_*.json`. Site data: `site/data/sweeps/alloc_private_noise.json`. Conventions are as in `findings.md`. σ is per sqrt(time unit 1/λ).

**Setup.**
- **Lone limit** (γ = 0, no lateral synapses): M = 500–1000 independent neurons per condition, stacked in one array.
  - Σ = diag(v), v = (0.5, 0.75, 1, 1.5, 2, 3); parts are vertices, with λ_b = v_b and c_b = v_b − θ.
  - Three inits: "weak" (all at the weakest vertex's fixed point), "strong" (all at v = 3), "rand" (W ~ Exp(0.2)).
  - 2000 time units; statistics over the second half.
- **Measures:**
  - A neuron is *in* part k if cos(h, part k) ≥ 0.9; otherwise it is *in transit*.
  - *alloc* = fraction of neurons whose nearest part is k.
  - Hops are counted by milestoning: a neuron's label is the last part it was in. Q_kl = (number of k→l hops) / (time spent labelled k).
- **Networks:** N = 120, γ ∈ {0, 1}, 4 blocks of 3 inputs (within-block correlation 0.6, across 0).
  - "graded": variances (0.7, 1, 1.4, 2), λ = (1.54, 2.2, 3.08, 4.4), κ0 = 0.234.
  - "mild": variances (0.85, 1, 1.15, 1.3), κ0 = 0.197.
  - θ = 0.12 (below θ_c) and 0.35 (above).
  - Inits: rand, all on block 0, all on block 3. Seeds 0 and 1, independent noise per run, T = 800.

## F11. The noise model decides everything; only additive and input noise let neurons hop
Three models; all keep ρ(w) = w0 − λw, with noise added on top.
- **(a) additive:** w += σ√dt·ξ, then clip at the floor.
- **(b) multiplicative:** w ← w·exp(σ√dt·ξ), then clip.
- **(c) private input / covariance-estimation noise (recommended):** each synapse estimates Cov(u_x, u_y) from its own independent samples, n = 1/σ² samples per unit time.
  - Gaussian (Isserlis) variance per step: σ²dt·(C_xx·C_yy + C_xy²).
  - For an I–B synapse that is q_i·Σ_bb + (Σh_i)_b²; for an I–I synapse it is γ²(q_i·q_j + C_ij²).
  - Its noise scales with the signal and does not vanish at the floor.
  - A per-neuron sample-stream variant (`inpn`, noise correlated across a neuron's synapses) agrees within 0.03.

**Claim (derived and verified).** Under (b), a floored synapse cannot regrow at the default floor.
- Its stationary law (dw = −θdt + σw·dW) is p(w) ∝ w⁻²·exp(2θ/(σ²w)), so the escape rate goes like exp(−x) with x = 2θ/(σ²·w_min).
- At w_min = 1e-4, x ≥ 111 for every σ ≤ 3.

**Evidence** (part M: M = 500, T = 500, θ ∈ {0.05, 0.2}, w_min ∈ {1e-4, 3e-3, 1e-2, 3e-2}, σ 0.5–3):
- At w_min = 1e-4, no neuron ever left the strong vertex. In part A, weak- and strong-started populations stay disjoint after 2000 time units (TV = 1.00) for σ = 0.3, 1 and 2 at all three θ.
- Exits from the strong vertex occur only for x ≲ 8 (none at x ≥ 10). log(rate) vs x has slope −1.2 (r = −0.95, 13 points across two θ and three floors).
- Multiplicative noise leaves the parts unblurred: off-part mass 0.00.

**Robustness:**
- dt = 0.0125 vs 0.05: allocation changes < 0.01 for (c), and ≤ 0.01 with ~10% more blur for (a).
- (b) stays frozen at dt = 0.0125.

**Amendment:** whether allocation is a property of the parameters depends on how a synapse fluctuates. Size-proportional ("biological") volatility pins pruned synapses at the floor, so allocation stays inherited from the transient unless the floor is large.

## F12. Lone-neuron theory: nonnegative power iteration, a Lyapunov function, no potential
**Derived.** Write w = s·h, s = 1ᵀw, A = Σ − θ11ᵀ. Then dw/dt = Ah − w, so:
- dh/dt = (I − h1ᵀ)Ah / s
- ds/dt = 1ᵀAh − s (active synapses only)

In rescaled time τ = ∫dt/s, dh/dτ = Ah − h(1ᵀAh). This is the normalised linear flow x' = Ax, clipped at the floor: the lone neuron performs continuous nonnegative power iteration on A.
- Since x' = Ax is the gradient flow of xᵀAx/2, the Rayleigh quotient R(h) = hᵀAh/hᵀh is non-decreasing.
- Its strict local maxima on the simplex are F1's stable parts.
- This is also why the transient decides allocation (F6): the outcome is that of power iteration from the initial weights.

**Verified** (part L): along the clipped flow, the largest decrease of R per step is 9e-9 (signed Σ, B = 8; clipping round-off) and 9e-16 for diagonal Σ.

**But** the drift is not a gradient field in w: the antisymmetric part of its Jacobian is 8–9% of the symmetric part (median; max 50–59%; θ = 0.05–0.4).
- With isotropic noise there is therefore no detailed balance and no Boltzmann law exp(−V/D).
- The stationary state is set by Freidlin–Wentzell quasi-potentials, and as σ → 0 all mass goes to the deepest well.

**1-D barrier (derived, verified).** From vertex k (c_k = v_k − θ), any input l with v_l > c_k offers a cheap exit.
- The floored synapse w_l sees drift −θ + (v_l/c_k − 1)·w_l, so ΔU = θ²/(2(v_l/c_k − 1)) in D = σ²/2 units.
- Predicted vs measured Arrhenius barriers (part R, 7 σ each, r ≤ −0.97):

| vertex | θ = 0.05 | θ = 0.2 | θ = 0.4 |
|---|---|---|---|
| v = 2 | 0.0023 vs 0.0026 | 0.030 vs 0.029 | 0.091 vs 0.086 |
| v = 1.5 | — | 0.0153 vs 0.0152 | 0.046 vs 0.052 |

- The loudest vertex has no such route, so the deepest well is the loudest part whenever λ_top − θ exceeds every other λ.

## F13. Allocation equilibrates, but almost one-way
**Claim.** Under (a) and (c) the stationary allocation does not depend on the start.
- TV(weak, strong) ≤ 0.004 for every (c) run with σ ≥ 0.1. The same holds for (a) with σ ≥ 0.15 (θ ≤ 0.2) or σ ≥ 0.2 (θ = 0.4).
- The slowest relaxation time of the measured rate matrix is 2.7–7.7 time units.
- It is reached mainly downhill: weak vertices empty within ~5–20 time units (out-rates 0.1–2 per unit time).

Uphill exits from the loudest part are Arrhenius:
- **(c):** rate ≈ 0.17–0.35·exp(−Δ/D), Δ = 0.11–0.13, nearly θ-independent (r ≤ −0.998).
- **(a):** Δ ≈ 0.38 / 0.46 / 0.55 at θ = 0.05 / 0.2 / 0.4 (3–8 σ, r ≤ −0.977). Less reliable, because the rate saturates above σ ≈ 0.7.

Numbers (time in units of 1/λ):

| case | uphill escape |
|---|---|
| (c), σ = 0.2 | one uphill exit per neuron per ~1100 time units |
| (a), θ = 0.2, loudest part | residence ~3e5 time units measured at σ = 0.3; Arrhenius extrapolation ~3e10 at σ = 0.2 |
| (a), θ = 0.4, σ = 0.1, v = 2 part | metastable for ~7e7 time units |

Not equilibrated within 2000 time units: (a) at θ = 0.4 with σ ≤ 0.15 (TV 0.19–0.38), and at θ = 0.2 with σ = 0.05 (TV 0.23).

Hopping is not too rare; the equilibrium itself is the problem (F14).

**Robustness:** two rate estimates agree within 0.06 at θ = 0.2, σ = 0.4 under (c). The rate matrix from vertex starts gives (.001 .004 .011 .061 .189 .735); direct occupation gives (.002 .009 .022 .082 .205 .680).

## F14. The stationary law: winner-take-all unless parts are blurred; a σ-controlled, θ-independent filter
**Claim.** With intact parts (transit ≤ 5% of neurons), ≥ 99% of neurons sit in the loudest part:
- (a), equilibrated runs with 0.15 ≤ σ ≤ 0.3: 0.999–1.000.
- (c), σ ≤ 0.2: 0.991–1.000.

The non-loudest parts hold more than 5% only when 9–23% (c) or ≥ 28% (a) of neurons are in transit. They hold more than 30% only at transit ≥ 41%.

| model | law (diag Σ, λ = v) | parameters | fit quality |
|---|---|---|---|
| (c) input | **n_k ∝ λ_k^β** | β = 7.1, 4.7, 3.2, 2.5, 2.2, 1.8 at σ = 0.25, 0.3, 0.4, 0.5, 0.6, 0.8. Spread across θ ∈ {0.05, 0.2, 0.4} is ≤ 1.0 at σ = 0.25, ≤ 0.5 at 0.3, ≤ 0.22 beyond | log-rms 0.02–0.21, r ≥ 0.995 |
| (c), σ → ∞ | β → ≈ 1 | 1.54, 1.28, 1.17, 1.05, 0.97 at σ = 1, 1.5, 2, 3, 5 (θ = 0.2; transit 77–84%) | log-rms ≈ 0.09 |
| (a) additive | **n_k ∝ exp(λ_k/T)** (Boltzmann in λ), plus an excess on the top part at σ ≤ 0.5 | T = 0.55–0.66 (σ 0.5), 1.13–1.16 (0.7), 1.9 (1), 3.2 (1.5), 4.6 (2), 7.1 (3); uniform as σ → ∞ | log-rms 0.07–0.10 at σ = 0.7 (power law: 0.25–0.28); 0.006–0.03 at σ ≥ 1 (power law: 0.03–0.14) |
| (b), w_min = 0.03, θ = 0.05 | power-law-like | β = 8.0, 2.0, 0.9, 0.1 at σ = 1, 1.5, 2, 3; off-part mass only 0.08–0.21 | lone limit only |

**Temperature:** σ², i.e. the inverse sample rate for (c) and 2D for (a).

**θ drops out under (c).** At σ = 0.4, θ = 0.05 / 0.2 / 0.4 give:
- (.002 .008 .021 .080 .203 .686)
- (.002 .009 .022 .082 .205 .680)
- (.003 .010 .025 .086 .208 .668)

θ keeps only the existence cutoff: parts with c_k ≤ 0 do not exist (a hard cutoff). Inside that cutoff the filter is soft: every existing part has nonzero occupancy.

**β ≈ 1 is not energy tracking.** At σ ≥ 1.5, the (c) allocation is the argmax statistics of noise-dominated weights with sd ∝ √v_b. P(argmax √v_b·|Z_b|) = (.038 .072 .108 .180 .244 .358) vs simulated (.036 .071 .108 .183 .249 .354) at σ = 1.5.

**Planted family of F6** (part P: M = 600 × 3 inits, T = 600; β = slope of log(frac0/fref) vs log(λ0/λ_ref)). Allocation is steeper than both the noiseless transient (F6: β = 2.9 at θ = 0.25) and the diagonal case:
- Variance family, (c): β = 18, 10.4, 6.4, 4.0 at σ = 0.25, 0.3, 0.4, 0.6, for θ = 0.15 and 0.25 (r ≥ 0.989). Transit is 46%, 66%, 85%, 96%.
- Variance family, (a): β = 7.5–10.9, 4.7–5.0, 2.7 at σ = 0.3, 0.4, 0.5 (transit 68–93%).
- Size family, (c) σ = 0.4: β = 4.2–4.9.
- TV between inits ≤ 0.023 everywhere.

**Answer to the spectral-filter question.** A stationary filter n(λ; σ) exists. It is monotone: a power law under input noise and Boltzmann-like under additive noise. But it is set by the noise level, not by θ. It is winner-take-all exactly where parts are well defined, and it reaches proportional allocation only when most neurons are between parts at any instant.

**Multiplicative noise with a large floor.** w_min = 0.01–0.03 is the only setting found with graded allocation and modest blur. But it implies ω = N·w_min ≥ 1.2 at N = 120, which F5 shows is lethal for a dense lateral network. It was tested only in the lone limit.

**Amendment:** with private noise the allocation becomes a law, and the law is "almost everything to the loudest supra-threshold part". A graded filter n_k ∝ λ_k^β(σ) (β ≈ 7 → 1 as σ grows) exists only at noise levels that keep many neurons between parts. θ decides which parts exist, not how they share neurons.

## F15. With lateral plasticity, private noise makes one bound cluster on the loudest part
**Claim** (part N, graded covariance, 2 seeds × 3 inits per cell). With γ = 1, every noisy condition ends with 100% of neurons on the loudest block (K = 1). TV between block-0- and block-3-started runs is 0.00. This covers (c) σ 0.2–0.4 and (a) σ 0.3, noise on all synapses or on I–B only, at θ = 0.12 and 0.35.
- This includes θ = 0.35 > θ_c = 0.234. There, the noiseless network from a random init populates 3–4 blocks (seeds 0, 1).
- The one exception is (a) σ = 0.1 with I–B-only noise at θ = 0.35: 87% on the loudest block, K = 2, TV 0.08, i.e. not equilibrated in 800 time units.
- A population started on block 0 leaves it within 5 time units. That is the cheap 1-D exit, not a collective move.
- So noise breaks the multi-cluster state above θ_c, not the binding below it.

**The bound cluster averages out private noise.** Under (c) at γ = 1: transit 0.00, member-to-mean L1 spread 0.010–0.033. The same noise at γ = 0 gives transit 0.05–0.80 and spread 0.29–0.37.

**The γ = 0 network matches the lone limit** (noise on I–B only):
- graded, θ = 0.35, (c) σ = 0.3: transit 0.54 vs 0.50.
- mild, θ = 0.35, σ = 0.3: 0.83 vs 0.78.

**Mild covariance:**
- Same outcome for (c) σ ≥ 0.3.
- (c) σ = 0.2 with I–B-only noise from block 0 keeps a 17% / 82% split.
- (a) σ = 0.1 does not equilibrate in 800 time units (TV 0.08–1.00).
- (a) σ = 0.3 dissolves all parts (transit 0.95–1.00, cos 0.60–0.67), as in the lone limit (blur, F17).

**Lateral noise acts as a floor.**
- Noisy floored lateral synapses sum to 1.4–13 per neuron at γ = 0 under (a), vs 0.012 without noise: an effective ω = N·E[w_lat], as in F5.
- With it, the γ = 0 population binds into one blob (graded, θ = 0.12, (a) σ = 0.3: all in block 3, transit 0, vs 0.44 transit without lateral noise).

**Amendment:** lateral plasticity plus private noise gives one cluster on the loudest part in this family, from any start. If noise reaches the lateral synapses, it adds a floor that grows like N·σ²/θ.

## F16. Private noise gives jitter, not diversity
**Claim.** Members differ instantaneously but not persistently.
- **Persistence:** correlation between a neuron's deviation from its cluster mean in the 3rd vs the 4th quarter of the run, using neurons that never left their part. It is −0.02 to 0.03 in every γ = 1 noisy cell and in γ = 0 (a) σ = 0.1. One γ = 0 cell with few stayers gave −0.28.
- **Time-averaged spread** is ~9× below instantaneous (0.002 vs 0.018; 0.011 vs 0.094). That is what averaging ~40 independent snapshots of pure jitter gives.
- **Within-cluster voltage correlation:** ≥ 0.9989 at γ = 1 and 0.97–0.9997 at γ = 0.
- **Within-cluster voltage participation ratio:** 1.000–1.02 (γ = 1) and 1.00–1.05 (γ = 0).
- Noiseless clusters remain exact clones (spread ~1e-15), as in F7.

**Amendment:** private noise does not turn a cluster into a reservoir. Members compute the same function plus independent jitter whose time average vanishes. Sparse wiring (F7) remains the only source of distinct functions found.

## F17. The cost of noise
**Graded regime** (diag Σ, (c) σ = 0.3–0.5, any θ):
- 21–69% of neurons are in transit at any moment.
- Off-part mass is 0.29–0.51, and the mean instantaneous cosine to the nearest part is 0.82–0.94.
- Averaging over time and members recovers the part: cosine 0.96–0.99 to the noiseless part.
- No neuron dies under (a) or (c) (dead = max weight < 0.05).

**Loudest-only regime** ((c) σ = 0.2; (a) σ ≤ 0.3): transit is 0–9%.

**Other cases:**
- The 12-input planted family blurs more at the same σ: 46–96% transit for (c) σ = 0.25–0.6.
- Under (b) with the default floor, blur is zero and nothing hops. With w_min = 0.03, off-part mass is 0.05–0.21 and allocation equilibrates (strong-vertex exit rate ≥ 0.066 per unit time at σ ≥ 2).
- The additive floor lift per synapse is 0.1–0.5 × σ²/(2θ) with Euler clipping.

## Inconclusive / not done
- The full Freidlin–Wentzell quasi-potential of the loudest well was not computed. Its barrier is measured (F13) but not derived; the 1-D formula covers only exits towards louder inputs.
- β(σ) for input noise is not derived.
  - β·σ ≈ 1.3 holds roughly for σ = 0.3–0.8.
  - (β − 1)·σ² is 0.33–0.49 over σ = 0.25–0.8.
  - Both are descriptive only.
- The additive Boltzmann temperature is descriptive: T/σ² falls from 2.4 (σ = 0.5–0.7) to 0.8 (σ = 3), and the top-part excess is unexplained.
- Networks covered one family (4 blocks of 3, two variance sets), N = 120, T = 800, 2 seeds. No ring or power-law covariances, and no N-scan of the lateral-noise floor.
- Multiplicative noise with a large floor was tested only in the lone limit; the network version is expected to die (F5) but was not run.
- Input noise (c) is the Gaussian (Isserlis) approximation with independent per-synapse streams. Exact finite-sample estimation and noise on the voltages themselves were not run; the per-neuron variant agrees within 0.03.
- No private-noise model was found that keeps parts intact (transit < 5%) and still gives graded allocation. None was ruled out either.
