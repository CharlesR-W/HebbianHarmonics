# Allocation under private noise (F11–F17) — revised 2026-11-17

Script: `hebb/sweeps/s8_private_noise.py`. Heavy parts run on Modal CPU via `hebb/sweeps/s8_modal.py` (`S8_MODAL=1`). `S8_REFLECT=1` selects the reflected scheme for additive/input noise and adds an `_r` suffix to raw outputs.
- Raw outputs: `scratch/out/sweeps/s8_*.json`.
- Site data: `site/data/sweeps/alloc_private_noise.json` (see `meta.revision`).
- Conventions as in `findings.md`: λ = η = 1, w_min = 1e-4, θ = −w0/η. σ is per √(time unit 1/λ).
- Additive and input noise use the reflected scheme at dt = 0.05; multiplicative uses the log scheme at dt = 0.0125. Both are defined in F11.

**Setup.**
- **Lone limit (γ = 0, no lateral synapses):** M = 500–1000 independent neurons per condition. Σ = diag(v), v = (0.5, 0.75, 1, 1.5, 2, 3); the parts are the vertices, with λ_b = v_b and c_b = v_b − θ.
- **Starting points:** "weak" (all neurons at the weakest vertex's fixed point), "strong" (all at v = 3), and "rand" (W ~ Exp(0.2)). Runs last 2000 time units (multiplicative: 1000; the large-σ part X: 600), with statistics over the second half.
- **Measures:**
  - A neuron is *in* part k if cos(h, part k) ≥ 0.9; otherwise it is *in transit*.
  - *alloc* is the fraction of neurons whose nearest part is k.
  - Hops are counted by milestoning (a neuron's label is the last part it was in). The rate matrix is Q_kl = (number of k→l hops) / (time spent labelled k).
- **Networks:** N = 120, γ ∈ {0, 1}, 4 blocks of 3 inputs (within-block correlation 0.6, across 0).
  - "graded": variances (0.7, 1, 1.4, 2), λ = (1.54, 2.2, 3.08, 4.4), κ0 = 0.234.
  - "mild": variances (0.85, 1, 1.15, 1.3), κ0 = 0.197.
  - θ = 0.12 (below θ_c = γκ0) and 0.35 (above it). Starts: rand, all on block 0, all on block 3. Seeds 0 and 1, independent noise per run, T = 800.

## F11. The noise model and its floor treatment decide everything

**Models** (all keep ρ(w) = w0 − λw; with h frozen over a step, a synapse's drift is c − w, where c = (Σh)_b − θ):

- **(a) additive:** Itô dw = (c − w)dt + σ dW, reflected at w_min.
- **(b) multiplicative:** Itô dw = (c − w)dt + σw dW, reflected at w_min.
  - Mean-neutral: noise does not change E[dw].
  - The earlier implementation (w ← w·exp(σ√dt ξ) after the drift step) integrated the Stratonovich SDE, which adds a drift of +σ²w/2.
- **(c) private covariance-estimation noise (recommended):** each synapse estimates Cov(u_x, u_y) from its own samples, n = 1/σ² samples per unit time.
  - Gaussian (Isserlis) variance: σ²dt·(C_xx·C_yy + C_xy²). For an I–B synapse that is q_i·Σ_bb + (Σh_i)_b²; for an I–I synapse it is γ²(q_i·q_j + C_ij²).
  - Its noise does not vanish at the floor.
  - A per-neuron variant (one sample stream per neuron, `inpn`) gives β within 0.2 of (c).

**Schemes** (Strang splitting, validated on the 1-D floored synapse; part V):
- **(a)/(c):** half-step of the exact flow dw/dt = c − w, sticking at w_min (exact for a deterministic flow into the floor); then the noise increment, reflected at w_min (exact for Brownian motion); then the second half-step.
  - The floored weight's analytic stationary mean is 0.0210 / 0.1666 (θ = 0.2, σ = 0.1 / 0.4). The scheme gives 0.0207 / 0.1662 at dt = 0.05 and 0.0210 / 0.1666 at dt = 0.0125.
  - The original projected Euler (clip) gave 0.0117 / 0.1336 at dt = 0.05: biased low by up to ~2×.
  - Naively reflecting the whole Euler increment is also wrong: a noiseless floored synapse gains θ·dt per step.
- **(b):** the same splitting in x = log w: exact flow of dw/dt = c − (1 + σ²/2)w over dt/2 (sticking at w_min), then x += σ√dt ξ reflected at log w_min, then dt/2 again.
  - **Stationary density** (θ = 0.05, w_min = 0.01; analytic p ∝ w^(−2−2/σ²)·exp(2θ/(σ²w))): total variation 0.012–0.019 at dt = 0.0125. The mean matches within 0.5% (0.01099 vs 0.01097 at σ = 1; 0.01732 vs 0.01723 at σ = 2).
  - **Escape rate** (dw = (−θ + aw)dt + σw dW, a = 3/1.95 − 1, absorbed at b = 3θ/a), compared with exact first-passage quadrature for x = 2θ/(σ²·w_min) = 2.5–10: within 0.4–4% at dt = 0.003, 3–18% at 0.0125, and −26% to +120% at 0.05.
  - **The old scheme** misses the escape rate by 1.5–5× in either direction (0 escapes against a predicted rate of 1.5e-4 at x = 8.3, dt = 0.05). At σ = 2 its stationary mean weight is 83–183 against an analytic 0.017.

**Claim (multiplicative, derived and verified).** A floored synapse under (b) leaves the floor at rate ~exp(−x), x = 2θ/(σ²·w_min).
- Quadrature gives ln(mean escape time) = 112, 250, 993, 3968 at θ = 0.05, σ = 3, 2, 1, 0.5, and 445–15 963 at θ = 0.2. So at the default floor, pruned synapses never return.
- Simulation, lone limit (dt = 0.0125, M = 500, T = 500):
  - No neuron left the strong vertex for any x ≥ 13.3.
  - At the default floor, weak- and strong-started populations stay disjoint after 1000 time units (TV = 1.00) in 10 of 12 cells (θ ∈ {0.05, 0.2, 0.4} × σ ∈ {0.3, 1, 2, 3}).
  - The two exceptions (θ = 0.4, σ = 2 and 3) are not floor escapes. They empty the weakest part (c = 0.1) when its single active weight collapses towards the floor and the neuron re-decides. Random starts there still differ (TV 0.59–0.60).
  - With large floors, strong-vertex exit rates fall with x at slope −0.93 per unit x (θ = 0.05, r = −0.95) and −0.70 (θ = 0.2, r = −0.96). Exiting the loudest vertex is not the 1-D problem, so x alone does not collapse the data.

**Amendment:** whether allocation becomes a property of the parameters depends on how synapses fluctuate. With size-proportional volatility and a small floor, pruned synapses never regrow and allocation stays inherited from the transient.

## F12. Lone-neuron theory: nonnegative power iteration, a Lyapunov function (zero floor only), no potential

**Derived.** Write w = s·h, s = 1ᵀw, A = Σ − θ11ᵀ. Then dw/dt = Ah − w, so:
- dh/dt = (I − h1ᵀ)Ah / s
- ds/dt = 1ᵀAh − s (active synapses only)

In rescaled time τ = ∫dt/s, dh/dτ = Ah − h(1ᵀAh). This is the normalised linear flow x' = Ax: the lone neuron performs continuous nonnegative power iteration on A.
- **Zero floor:** x' = Ax is the gradient flow of xᵀAx/2, so R(h) = hᵀAh/hᵀh is non-decreasing along the projected flow on the orthant. Its strict local maxima on the simplex are F1's stable parts.
- **Positive floor:** this is only approximate. Measured worst dR/dt is ≥ −1e-13 at w_min = 0 and −5e-12 at 1e-6. At w_min = 1e-4 it is −6e-6 (signed Σ) and −2e-5 (diagonal); at 1e-3 it is −3e-3 (part L2; B = 8, θ = 0.15).
- This explains why the transient decides allocation (F6): the outcome is that of power iteration from the initial weights.

**Not a gradient field.** The drift has a substantial curl: the antisymmetric part of the Jacobian is 8–9% of the symmetric part (median; up to 50–59%).
- With isotropic noise there is therefore no detailed balance and no Boltzmann law. The small-noise limit is governed by Freidlin–Wentzell quasi-potentials, which I did not compute.

**1-D barrier (a linearisation).** Hold s = c_k and the active weight fixed; then a floored synapse to an input l with v_l > c_k sees drift −θ + (v_l/c_k − 1)·w_l. The barrier is ΔU = θ²/(2(v_l/c_k − 1)), in units of D = σ²/2.

Measured Arrhenius barriers under additive noise (part R, 7 σ, r ≤ −0.92), measured vs predicted:

| vertex | θ = 0.05 | θ = 0.2 | θ = 0.4 |
|---|---|---|---|
| v = 2 | 0.0021 vs 0.0023 | 0.027 vs 0.030 | 0.064 vs 0.091 |
| v = 1.5 | — | 0.013 vs 0.015 | 0.045 vs 0.046 |

Agreement is within 1–30%; the gap grows with θ, as expected for a linearisation.
- The loudest vertex has no such route, and its measured barriers are much higher (F13).
- "The deepest well is the loudest part" is an empirical statement for the tested families: every stationary state observed at small σ concentrates there (F14). It is not derived.

## F13. Allocation equilibrates under (a) and (c), mostly one-way

**Claim.** The stationary allocation does not depend on the start:
- TV(weak, strong) ≤ 0.004 for every (c) run with σ ≥ 0.1, and for (a) with σ ≥ 0.15 (θ ≤ 0.2) or σ ≥ 0.2 (θ = 0.4).
- Slowest relaxation time of the measured rate matrix: 3.1–5.5 time units for (c) and 5.4–10.7 for (a).
- It is reached mostly downhill: weak vertices empty within ~5–20 time units (out-rates 0.1–1.3 per unit time).

Uphill exits from the loudest part follow Arrhenius laws (fits over σ ≤ 0.8):
- **(c):** Δ = 0.10–0.12 in D units, θ-independent (prefactor 0.13–0.43, r ≤ −0.987).
- **(a):** Δ = 0.34 / 0.46 / 0.57 at θ = 0.05 / 0.2 / 0.4. Only 4 σ values each; r ≤ −0.983.

**Timescales (units of 1/λ):**
- (c), σ = 0.2: one uphill exit per neuron per ~1200 time units (measured).
- (a), θ = 0.2: residence in the loudest part is ~7e4 (measured) at σ = 0.3, extrapolated to ~1.5e10 at σ = 0.2.
- (a), θ = 0.4, σ = 0.1: residence in the v = 2 part is ~1e6 (extrapolated).
- Not equilibrated within 2000 time units: (a) at θ = 0.4, σ ≤ 0.15 (TV 0.12–0.27), and at θ = 0.2, σ = 0.05 (TV 0.17).
- (b) at the default floor never equilibrates (F11).

**Robustness:** two estimates of the stationary state agree within 0.08 for (c) at θ = 0.2, σ = 0.4. The rate matrix from vertex starts gives (0, .001, .006, .049, .177, .766); direct occupation gives (.002, .007, .019, .078, .203, .691).

**Conclusion:** hopping is not too rare; the problem is the stationary state itself (F14).

## F14. The stationary law: winner-take-all wherever parts are intact (tested families only)

**Scope.** Only these families were tested:
- diagonal Σ with the 6 variances above (lone limit);
- the F6 planted 4×3-block family (lone limit);
- two 4-block covariances in N = 120 networks (F15).

No ring, power-law or signed covariances were tested.

**Claim.** Where parts stay intact, ≥ 99% of neurons sit in the loudest part:
- (a) with 0.15 ≤ σ ≤ 0.3 (equilibrated, transit ≤ 8%): 0.998–1.000.
- (c) with σ ≤ 0.2 (transit 2–9%): 0.991–1.000.

The other parts together hold more than 5% only when the transit fraction is ≥ 11% for (c), ≥ 45% for (a), and 8% for (b) at w_min = 0.01. They hold more than 30% only at transit ≥ 48% (c), 85% (a), or 36% (b, w_min = 0.03).

| model | law on diag Σ (λ = v) | parameters | fit quality |
|---|---|---|---|
| (c) input | **n_k ∝ λ_k^β** | β = 6.9–7.2, 4.6–5.1, 3.36–3.41, 2.73–2.76, 2.34–2.42, 1.91–2.03 at σ = 0.25, 0.3, 0.4, 0.5, 0.6, 0.8 (ranges over θ = 0.05, 0.2, 0.4) | log-rms 0.01–0.14, r ≥ 0.994 |
| (c), large σ | β keeps falling | 1.74, 1.48, 1.36, 1.25, 1.16 at σ = 1, 1.5, 2, 3, 5 (θ = 0.2; transit 83–90%) | log-rms 0.10–0.12 |
| (a) additive | **n_k ∝ exp(λ_k/T)**, plus an excess on the top part at σ ≤ 0.5 | T = 0.72–0.74 (σ 0.5), 1.31–1.40 (0.7), 2.2 (1), 3.7 (1.5), 5.0 (2), 7.9 (3); uniform as σ → ∞ | log-rms 0.056–0.066 at σ = 0.7 (power law 0.21–0.22); 0.005–0.025 at σ ≥ 1 |
| (b) multiplicative, w_min = 0.01–0.03 | steep, roughly a power law | θ = 0.05, w_min = 0.03: β = 9.0, 2.8, 1.7, 0.8 at σ = 1, 1.5, 2, 3, with transit 2%, 18%, 36%, 54%. θ = 0.05, w_min = 0.01, σ = 2: β = 3.3, loudest part 0.87, transit 8%, off-part mass 0.15 (dt 0.05 / 0.0125 / 0.003: 0.877 / 0.869 / 0.867) | r 0.96–0.997, log-rms 0.12–0.50: not a clean power law |

**What plays the temperature:** σ², i.e. the inverse sample rate for (c) and 2D for (a).

**θ drops out under (c)**, apart from the existence cutoff (parts with c_k ≤ 0 do not exist). At σ = 0.4, θ = 0.05 / 0.2 / 0.4 give:
- (.002, .007, .019, .077, .201, .695)
- (.002, .007, .019, .078, .203, .691)
- (.002, .008, .021, .080, .204, .685)

Within the cutoff the filter is soft: every existing part has nonzero occupancy.

**At large σ, (c) allocation reflects noise statistics, not energy.** At σ = 3 it is within TV 0.015 of the argmax statistics of noise-only weights with sd ∝ √v_b (argmax_b √v_b·|Z_b|: (.038 .072 .108 .180 .244 .358)). At σ = 5 it moves past them (β = 1.16 vs 1.25), and the σ → ∞ limit was not established.

**Planted F6 family** (part P: M = 600 × 3 starts, T = 600; β = slope of log(frac0/fref) vs log(λ0/λ_ref); TV between starts ≤ 0.014). Allocation is steeper than both the noiseless transient (F6: β = 2.9 at θ = 0.25) and the diagonal case:
- Variance family, (c): β = 17.5–18.0, 10.2–10.7, 6.5–6.6, 4.2–4.3 at σ = 0.25, 0.3, 0.4, 0.6 (θ = 0.15 and 0.25; r ≥ 0.984). Transit is 53–56%, 71–72%, 87–88%, 96–97%.
- Variance family, (a): β = 7.1–10.3, 3.9–4.0, 2.2–2.3 at σ = 0.3, 0.4, 0.5 (r 0.80–0.91; transit 74–96%).
- Size family, (c) at σ = 0.4: β = 4.3–5.0 (transit 97%).

**Answer to the spectral-filter question, for these families.** There is a stationary, monotone filter n(λ; σ): a power law under input noise and Boltzmann-like under additive noise. It is set by the noise level, not by θ. It is winner-take-all wherever parts are well defined. It becomes graded only when a sizeable fraction of neurons is between parts at any moment, and proportional allocation (β ≈ 1) is reached only at very high noise or by tuning.

Multiplicative noise with a large floor gives the best trade-off found: 13% outside the loudest part at 8% transit. But it needs ω = N·w_min ≥ 1.2 at N = 120, which F5 shows is lethal for dense lateral networks; only the lone limit was tested.

**Amendment:** with private noise the allocation becomes a law. In the tested families that law is "almost everything to the loudest supra-threshold part", softened into n_k ∝ λ_k^β(σ) (β ≈ 7 → 2 for σ = 0.25 → 0.8) only at noise levels that keep many neurons between parts. θ decides which parts exist, not how they share neurons.

## F15. With lateral plasticity, private noise makes one bound cluster on the loudest part

Part N, reflected scheme, 2 seeds × 3 starts per cell.

**Graded covariance, γ = 1:** every noisy condition ends with 100% of neurons on the loudest block (K = 1), and TV between block-0- and block-3-started runs is 0.00. This covers (c) σ 0.2–0.4 and (a) σ 0.1–0.3, noise on all synapses or on I–B only, at θ = 0.12 and 0.35.
- This includes θ = 0.35 > θ_c, where the noiseless network from a random start populates 3–4 blocks.
- The one exception: (a) σ = 0.1 with I–B-only noise at θ = 0.35 (random start: 75% on the loudest block, K ≈ 1.9, TV 0.07; not equilibrated in 800 time units).
- A population started on block 0 leaves it within 5 time units by the cheap 1-D exit, not collectively.
- So noise breaks the multi-cluster state above θ_c, not the binding below it.

**The bound cluster averages out private noise.** At γ = 1: transit 0.00, member-to-mean L1 spread 0.002–0.037. Under (c) at γ = 0: transit 0.06–0.84, spread 0.28–0.33.

**The γ = 0 network matches the lone limit** (I–B-only noise), comparing transit:
- graded, θ = 0.35, (c) σ = 0.3: 0.58 vs 0.56;
- mild, θ = 0.35, (c) σ = 0.3: 0.86 vs 0.87.

**Mild covariance:**
- At (c) σ = 0.2 with I–B-only noise, a split persists (block-0 start 18% / 82%; random start 32% / 68%; TV 0.18).
- At (c) σ = 0.3, all neurons end in one cluster on the loudest block.
- At (c) σ = 0.4 the bound population is blurred: transit 0.78–1.00, cos 0.73–0.88.
- (a) σ = 0.1 with I–B-only noise does not equilibrate (TV 0.17–0.35).
- (a) σ = 0.3 dissolves every part (transit 0.99–1.00, cos 0.57–0.65), as in the lone limit.

**Lateral noise acts as a floor.** Under (a), floored lateral synapses sum to 1.5–16 per neuron at γ = 0, vs 0.012 without noise: an effective ω = N·E[w_lat] in F5's sense.
- It binds the population: graded, θ = 0.12, (a) σ = 0.3 puts everyone on block 3 with transit 0, vs 0.64 transit with I–B-only noise.

**Amendment:** in these families, lateral plasticity plus private noise gives one cluster on the loudest part from any start. Noise reaching the lateral synapses adds a floor that grows like N·σ²/θ.

## F16. Private noise gives jitter, not diversity

**Claim.** Cluster members differ at any instant but not persistently.
- **Persistence** (correlation between a neuron's deviation from its cluster mean in the 3rd vs 4th quarter of the run, using neurons that never left their part): −0.02 to 0.04 in every cell with such neurons (all γ = 1 noisy cells, and γ = 0 under (a) σ = 0.1).
- **Time-averaged spread** is ~9× smaller than the instantaneous spread (0.0021 vs 0.019; 0.011 vs 0.092), as expected from averaging ~40 independent snapshots of pure jitter.
- **Within-cluster voltage correlation:** ≥ 0.9989 at γ = 1 and 0.975–0.9998 at γ = 0.
- **Within-cluster voltage participation ratio:** 1.000–1.002 (γ = 1) and 1.00–1.05 (γ = 0).
- Without noise, members are exact clones (spread ~1e-15).

**Amendment:** private noise does not make a cluster a reservoir. Members compute the same function plus jitter whose time average vanishes. F7's sparse wiring remains the only source of distinct functions found.

## F17. The cost of noise

**Graded regime** (diag Σ, (c) σ = 0.3–0.5, any θ):
- 25–73% of neurons in transit at any moment;
- off-part mass 0.35–0.54;
- mean instantaneous cosine to the nearest part 0.81–0.93;
- the time- and member-averaged embedding of each occupied part still has cosine 0.95–0.98 to the noiseless part;
- dead fraction ≤ 3e-5 (dead = max weight < 0.05).

**Winner-take-all regime** ((c) σ = 0.2; (a) σ ≤ 0.3): transit 0–9%.

**Planted 12-input family:** blurs more at the same σ (53–97% transit for (c) σ = 0.25–0.6).

**Multiplicative with a large floor** (θ = 0.05):
- w_min = 0.01, σ = 2: transit 8%, off-part mass 0.15, cosine 0.98;
- w_min = 0.03, σ = 2: transit 36%, off-part mass 0.37, cosine 0.90.

**Floored-synapse lift under (a):** follows the reflected-OU density p ∝ exp(−(2/σ²)(θw + w²/2)), e.g. mean 0.021 at θ = 0.2, σ = 0.1. The earlier "0.1–0.5 × σ²/(2θ)" combined clipping bias with a formula that ignored the −w term.

## Inconclusive / not done
- No quasi-potential (Freidlin–Wentzell) computation. Barriers out of the loudest part are measured only, and "deepest well = loudest part" is empirical.
- No derivation of β(σ) for input noise. The large-σ limit of β is not established (still 1.16 and falling at σ = 5).
- The additive Boltzmann form and its T(σ) are descriptive (T/σ² falls from ~3 at σ = 0.5 to 0.9 at σ = 3), and the top-part excess is unexplained.
- Multiplicative noise with a large floor was tested only in the lone limit. Its allocations are not clean power laws (r 0.96–0.997).
- The network runs were redone with the reflected scheme but cover only one block family, N = 120, T = 800, 2 seeds. There is no N-scan of the lateral-noise floor.
- (c) is the Gaussian approximation with independent per-synapse sample streams. Exact finite-sample estimation and noise on the voltages themselves were not run.
- No private-noise model was found that keeps parts intact (transit < 5%) and still gives graded allocation; none was ruled out.
