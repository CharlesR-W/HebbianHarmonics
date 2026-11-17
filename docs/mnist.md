# Chapter 3 — the Hebbian resistor network on kernel-lifted MNIST

What was run, why, what came out, and what the page can show. Code: `hebb/mnist/`.
Page data: `site/data/mnist/` (format in its `README.md`). Vocabulary follows
`docs/findings.md` and `docs/review_corrections.md`: **part** (a cluster's shared
harmonic embedding, a nonnegative localized Perron-type vector of Σ − θ11ᵀ on a support),
**ω = N·w_min** (the floor coupling), **dead pool** (neurons with every synapse at the
floor), the **collective threshold γκ0** (a heuristic, not a phase boundary),
**parts-sum estimator**, and **readout best mid-training**.

## 1. Setup

**Model.** This is SPEC §1 exactly, in torch (`hebb/mnist/model.py`):

- Each step clips to the floor, computes H = L_II⁻¹W_IB by Cholesky, applies dw = dt·(−θ − λw + Cov), then clips at w_min.
- Every I–I and I–B synapse is potential (p = 1), with γ = 1, λ = η = 1, dt = 0.1, and 3000 steps (300 time units).
- **Stochastic mode** (default) uses Cov = the unbiased minibatch covariance of u = Hg from 256 random training digits per step.
- **Mean-field mode** uses the exact Σ_g.
- With identical inputs, the code reproduces `hebb.core.step` to 1e-15 over 500 steps (`hebb/mnist/check_core.py`).
- Dynamics run in float32 on GPU (float64 on CPU). All readouts and the final analysis run in float64; see §7.

**Stimuli.** MNIST digits, 2×2 average-pooled to 14×14 (196 values in [0, 1]). Σ_g is the exact covariance over the 60k training set. Display, tuning and samples use the 10k test set.

**Lifts g = φ(x).**

| lift | B | kernel | notes |
|---|---|---|---|
| `pixels` | 196 | linear | h_i is literally an image |
| `rff64/256/1024` | 64, 256, 1024 | Gaussian, RFF √(2/D)cos(ωᵀx+b) | bandwidth σ = median pairwise distance = 4.47 (14×14 units); also σ/2 and 2σ at D = 256 |

No third lift was run. I considered random ReLU features and local-patch features. The pilots showed the bottleneck is the plasticity dynamics (how many parts get populated before extinction, §3), not the kernel: every RFF size and bandwidth gave the same 2-part structure. A third lift would not have been clearly better motivated than the init/ω controls I spent the budget on.

**Feature scaling.** Each lift is multiplied by ONE global constant so that the mean feature variance is 1 (tr Σ_g = B). Σ_g entries are then on a correlation-like scale, and θ can be read against them. I did not z-score each feature: that would blow up near-constant border pixels and change the kernel geometry.

**Centred covariance, not raw second moment.** The rule as written uses Cov, so the network never sees the mean of g. The off-diagonal entries are signed, as the table shows.

Distribution of off-diagonal Σ_g entries:

| lift | 1% | 10% | 50% | 90% | 95% | 99% | 99.9% | max | negative | diag (max) |
|---|---|---|---|---|---|---|---|---|---|---|
| pixels | −0.36 | −0.035 | 0.000 | 0.130 | 0.326 | 0.842 | 1.55 | 1.87 | 53% | 1.00 (2.98) |
| rff1024 | −0.48 | −0.19 | 0.000 | 0.187 | 0.279 | 0.484 | 0.741 | 1.19 | 50% | 1.00 (2.36) |
| rff256 | −0.49 | −0.19 | 0.000 | 0.187 | 0.277 | 0.488 | 0.769 | 1.04 | 50% | 1.00 (2.25) |

Conductances are nonnegative, so negative covariances can only prune: an anticorrelated pair is cut at *any* θ ≥ 0 (pruning by anticorrelation, SPEC §7.2), not just by the drift θ. For pixels, anticorrelated stroke regions (a "1" vs a "0") are therefore never wired together. For RFF features, half of all pairs are anticorrelated and a part must be a set of mutually positively correlated features. In the readout, anticorrelations appear only as clipped-edge inequalities, which the estimator does not use.

**What "recovering Σ_g" does and does not tell you.** Σ_g is the finite-dimensional version of the centred kernel covariance operator, i.e. what kernel PCA diagonalises. For a characteristic kernel, the *mean* embedding E φ(x) already determines p(x), and this network never sees the mean.

- Even a perfect Σ_g does not give p(x). Example from review_corrections §8: two-atom mixtures with weights p and 1 − p have the same centred feature covariance up to the factor p(1 − p).
- For RFF features specifically, the raw second moments are values of the characteristic function at ω_i ± ω_j. So a rich enough raw second-moment matrix would carry distributional information, but the centred matrix removes exactly the mean part.
- Recovering Σ_g gives kernel-PCA structure (which feature combinations co-vary), not a density.

**Two N-dependent scalings** were introduced after the pilots; see §7.

- **Floor:** w_min = ω/N with ω = 0.5 by default; alternatives ω = 0.05 and 5.
- **Lateral init:** W_II ~ U(0, 0.2·512/N), so the total initial lateral conductance per neuron (about 51) does not grow with N.

With the original fixed w_min = 1e-3 and U(0, 0.2), N = 2048 went fully extinct at θ ≥ 0.6, where N = 512 had live parts. This matches F5: extinction collapses on ω.

**Init.** Every synapse is potential and starts positive in both variants. This matters in high dimension.

- **Generic** (the requested default): W_IB ~ Exp(mean 0.1).
- **Heavy**: W_IB ~ Gamma(shape 0.1, mean 0.1). This is still dense, but each neuron leans on a few inputs.

With B in the hundreds, a dense iid init puts every embedding within O(1/√B) of the centroid. Pruning then races symmetry breaking (§3).

**Grid (114 runs, all completed).**

- **Pixels:** N ∈ {512, 2048} × θ ∈ {0.3, 0.45, 0.6, 0.75, 0.9, 1.1} × {generic, heavy}.
  - Seed 1 for all N = 2048 pixel runs.
  - Mean-field for pixels heavy (both N, all θ).
  - ω ∈ {0.05, 5} at θ ∈ {0.6, 0.75, 0.9}.
- **RFF:** θ ∈ {0.05, 0.1, 0.15, 0.2, 0.3} for rff64 (N = 512), rff256 (N = 512 and 2048), rff1024 (N = 2048), each with both inits.
  - Bandwidths ×0.5 and ×2 (rff256, N = 2048, heavy).
  - Seed 1 for rff1024 heavy.
  - Mean-field for rff256 N = 2048 heavy.

Raw artifacts: `/var/tmp/hebb-mnist/runs/20261116-233411-final_cpu` (N = 512, Modal CPU) and `…/20261117-001011-final_gpu` (N = 2048, Modal A10). Full weight matrices remain on the Modal Volume `crw-hebbian-mnist-results` under `<run>/weights/`.

**Measured over training** (36 log-spaced snapshots):
- H (512 stored neurons, float16)
- clusters under several definitions
- effective rank of H
- weight histograms and fraction at the floor
- the projected stationarity residual (SPEC §1, mean-field drift at the snapshot weights)
- the masked-LS readout (both families) with all metrics

At the end, everything is recomputed on the **time-averaged weights of the last 20% of steps** (600 steps). That is how the stochastic runs are read out.

**Cluster definition.** Single-linkage components of "L1(h_i, h_j) < 0.1"; this is order-free. I also report tolerance 0.05 and 0.2, `core.clusters` (leader clustering), components of the active I–I graph, "major" clusters (≥ 1% of N), and live clusters (excluding the dead pool). In all 114 runs the counts at tolerance 0.05, 0.1 and 0.2 are identical, and they equal the leader count. Members of a cluster are clones: within-cluster L1 spread is about 1e-5 (headline 9.6e-6). This agrees with F7: redundancy, not a reservoir of distinct random functions.

## 2. Headline run

`pixels_N2048_th0.9_nw0.5_heavy_sgd_s0`: pixels, N = 2048, θ = 0.9 (99.1th percentile of off-diagonal Σ), ω = 0.5, heavy init, SGD.

- **Clusters:** 13 clusters = 12 live parts + 1 dead pool of 1625 neurons. Live sizes are 101, 56, 46, 45, 39, 36, 32, 19, 14, 14, 12, 9. Seed 1 gives 11 live + 1699 dead; mean-field gives 16 live + 1629 dead.
- **Parts:** blobs of 2–3 pixels at high-variance stroke locations (`cluster_parts.png`). Their top-activating digits are mostly heavy-ink 8s, 0s and 3s (`cluster_top_digits.png`). They detect ink mass at a location; they are not digit-class detectors.
- **Stationarity residual** on the averaged weights: mean |dw/dt| = 1.9e-4 and active-edge mean 7e-3, but max 0.46 (a few noisy edges, relative 0.5). The mean-field twin reaches residual 0.
- **Energy per part:** h_kᵀΣh_k lies in 1.04–1.37. That is θ plus the within-cluster weight, while sizes vary 11× (9 to 101). Spearman ρ(size, energy) = 0.69, but the energy range is tiny. Energies are pinned just above θ, and the neuron count per part is inherited from the initial basins (F4). This does not support "neurons ∝ mode energy".

## 3. Cluster count vs θ: one part → a few parts → dead pool + small parts → extinction

All runs are SGD, ω = 0.5, seed 0. Format: live clusters (major) / dead neurons.

| lift, N, init | θ = 0.3 | 0.45 | 0.6 | 0.75 | 0.9 | 1.1 |
|---|---|---|---|---|---|---|
| pixels 512 generic | 1 / 0 | 3 (2) / 0 | 4 (3) / 480 | 0 / 512 | 0 / 512 | 0 / 512 |
| pixels 512 heavy | 1 / 0 | 3 / 0 | 3 / 0 | 10 (8) / 389 | 0 / 512 | 0 / 512 |
| pixels 2048 generic | 1 / 0 | 2 / 0 | 3 (2) / 0 | 10 (3) / 1901 | 5 (0) / 2040 | 0 / 2048 |
| pixels 2048 heavy | 1 / 0 | 3 / 0 | 3 / 0 | 7 (6) / 1191 | 12 (7) / 1625 | 0 / 2048 |

| lift, N, init | θ = 0.05 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|
| rff64 512 heavy (generic: 1 everywhere) | 2 | 2 | 2 | 2 | 2 / 415 dead |
| rff256 2048 heavy | 2 | 2 | 2 | 2 | 2 |
| rff256 2048 generic | 2 | 2 | 2 | 2 | 4 (1) / 1933 dead |
| rff1024 2048 heavy | 2 | 2 | 2 | 2 / 1098 dead | 2 / 1865 dead |
| rff256 bw×0.5, 2048 heavy | 2 | 1 | 2 / 1997 dead | extinct | extinct |
| rff256 bw×2, 2048 heavy | 2 | 2 | 2 | 2 | 2 |

The θ values span the 94th–99.3rd percentile of off-diagonal pixel Σ (RFF: 64th–95th).

What happens:

- **Low θ:** one full-support part. For pixels at θ = 0.3 every neuron converges onto a "0"-shaped ring, the Perron vector of the high-variance stroke region; see `neuron_tiles.png` of the θ = 0.3 run.
- **Moderate θ:** 2–3 large parts.
- **Higher θ:** most neurons lose all inputs before they localise and join the dead pool, a floor-coupled clump at the centroid. The survivors form 7–12 small parts.
- **θ ≈ 1.1 for pixels:** extinction, even though single-pixel vertices with Σ_bb up to 2.98 remain stable parts by F1. Starting from near-centroid embeddings, the boundary drive (Σh)_b ≈ (row mean of Σ) ≤ 0.16 is far below θ, so every weight decays before the embedding can concentrate (F5's mechanism).
- **The "fragmented" regime is never reached.** The cluster count peaks at about 12 just below extinction.
- **Heavy init vs generic:** the heavy-tailed init roughly doubles the number of populated parts and moves extinction from about θ = 0.75 to about 1.0. **Which parts are populated is a property of the transient, not of θ alone (F2).**
- **RFF lifts:** at every D and bandwidth there are exactly two large parts, then the dead pool, then extinction. The RFF Σ is symmetric about 0 (half the pairs anticorrelated), so a positive-correlation part can only gather a correlated half of the features.

**The collective threshold** γκ0 = mean initial lateral covariance ≈ mean entry of Σ (0.03 for pixels, about 0.0005 for RFF) is far below every θ used. So lateral binding at t = 0 is not what sets the count here. The dense lateral init does initially average embeddings toward the centroid, which feeds the extinction race.

**Mean-field vs stochastic (minibatch 256).**

- They agree closely at most θ: pixels 2048 θ = 0.45 (1316/541/191 vs 1344/513/191), all RFF runs, and pixels 512 θ = 0.45 and 0.75.
- They differ near transitions:
  - pixels 2048 θ = 0.6: MF 6 live + 807 dead; SGD 3 live, none dead.
  - pixels 2048 θ = 0.9: MF 16 live; SGD 12.
  - pixels 2048 θ = 0.3: MF 2; SGD 1.
- Readout scores agree within about 0.05.
- In the mean-field runs the residual reaches 0 (to printed precision).

**Seeds** (pixels 2048, seeds 0/1): live counts 3/2, 3/4, 7/7, 12/11 for heavy at θ = 0.45/0.6/0.75/0.9; generic 10/8 at θ = 0.75. Sizes differ, so the allocation is seed-dependent.

**ω (floor)**, pixels 2048 heavy:

- ω = 5 → extinct at θ ≥ 0.6.
- ω = 0.05 → fewer dead neurons and fewer, larger parts: θ = 0.9 gives 7 live / 627 dead, vs 12 / 1625 at ω = 0.5.
- ω = 0.05, θ = 0.75 has a larger residual (mean 0.039): it is not converged at 300 time units.

The floor is a real knob (F5).

## 4. Readout of Σ_g from the weights

**Estimator** (`hebb/mnist/readout.py`).

- *Masked ridge least squares* over symmetric C on active edges (w > 2·w_min):
  - I–B: (HC)_ib = θ + w_ib.
  - I–I: h_iᵀCh_j = θ/γ + w_ij/γ.
- Each family's squared residual is averaged over its equations, so the about 2M lateral equations do not swamp the feed-forward ones.
- Ridge α = 1e-4·‖A‖; conjugate gradients in float64.
- Variants: both families, I–I only, I–B only.
- The note's naive formula C = H⁺RH⁺ᵀ: R from all lateral weights (clipped ones included), diagonal filled with the row max, ridge pseudo-inverse; plus its feed-forward analogue sym(H⁺R_IB).
- No hinge penalty and no PSD projection inside the fit.

**Scores** (SPEC §7.3):

- *known* (θ, λ/η known; full C vs full Σ): relative Frobenius error and unscaled BW²/trΣ.
- *unknown scale*: Frobenius cosine, and BW with optimal scale = 1 − F²/(trC·trΣ).
- *unknown θ*, the contrast-subspace score, on PCP vs PΣP: cosine, scaled BW, top-k eigenspace overlap.

Negative eigenvalues are clipped before BW, and the clipped mass is reported (10–15% for LS).

**Baselines:** identity; diag Σ; rank-K truncation of Σ (K = cluster count; the best any K-part sketch could hope for); the same LS applied to the untrained network (the "equilibrium assumption violated" control); and the parts-sum estimator θ11ᵀ + Σ_k c_k h_kh_kᵀ/|h_k|², with c_k = a member's summed input conductance.

Final numbers, contrast cosine / contrast BW (lower BW is better):

| run | LS both | LS I–I | LS I–B | naive I–B | parts-sum | untrained | identity | diag | rank-K |
|---|---|---|---|---|---|---|---|---|---|
| pixels 2048 heavy θ0.9 (12 parts) | **0.50 / 0.67** | 0.42 / 0.74 | 0.49 / 0.68 | 0.38 / 0.74 | 0.41 / 0.74 | 0.35 / 0.64 | 0.35 / 0.63 | 0.52 / 0.35 | 0.96 / 0.38 |
| pixels 2048 generic θ0.75 (10) | 0.54 / 0.66 | 0.34 / 0.84 | 0.54 / 0.66 | 0.42 / 0.74 | 0.36 / 0.81 | 0.35 / 0.64 | 0.35 / 0.63 | 0.52 / 0.35 | 0.95 / 0.42 |
| pixels 2048 heavy θ0.45 (3) | 0.53 / 0.81 | 0.46 / 0.84 | 0.53 / 0.81 | 0.47 / 0.84 | 0.46 / 0.84 | 0.35 / 0.64 | 0.35 / 0.63 | 0.52 / 0.35 | 0.74 / 0.74 |
| pixels 2048 heavy θ0.3 (1) | 0.38 / 0.92 | 0.35 / 0.93 | 0.38 / 0.92 | 0.35 / 0.93 | 0.35 / 0.93 | 0.35 / 0.64 | 0.35 / 0.63 | 0.52 / 0.35 | 0.49 / 0.90 |
| rff1024 2048 heavy θ0.1 (2) | 0.49 / 0.92 | 0.39 / 0.93 | 0.48 / 0.92 | 0.40 / 0.93 | 0.39 / 0.93 | 0.18 / 0.70 | 0.18 / 0.70 | 0.20 / 0.69 | 0.68 / 0.84 |
| rff256 2048 heavy θ0.1 (2) | 0.51 / 0.90 | 0.41 / 0.92 | 0.48 / 0.90 | 0.43 / 0.92 | 0.41 / 0.92 | 0.34 / 0.49 | 0.34 / 0.49 | 0.38 / 0.47 | 0.71 / 0.82 |
| rff64 512 heavy θ0.1 (2) | 0.61 / 0.84 | 0.50 / 0.86 | 0.57 / 0.85 | 0.54 / 0.86 | – | 0.56 / 0.28 | 0.56 / 0.27 | 0.61 / 0.24 | 0.76 / 0.76 |

Across all live runs, the LS contrast cosine is 0.28–0.61.

- **Pixels:** it is about equal to diag Σ (0.52).
- **RFF:** it is clearly better than diag or identity (0.49 vs 0.20 for D = 1024).
- **BW:** the readout is always much worse than identity or diag (0.66–0.94 vs 0.24–0.70), because C_hat is nearly zero outside the measured parts and BW penalises the missing variance.
- **Known setting:** relative Frobenius error 0.86 (headline); identity 0.94, diag 0.86, rank-K 0.26.

**Where it works.** On the quantities the weights actually measure, the readout is essentially exact.

- Part energies: c_kᵀĈc_k matches c_kᵀΣc_k to about 3 decimals.
- Restricted to the boundary nodes the live parts cover, Ĉ reproduces Σ with cosine 0.84 (headline, 30 of 196 pixels; 0.75–0.84 across the featured pixel runs, 0.63/0.73 for rff1024/rff256). The median part-energy error is 0.3–2%.
- Ĉ puts about 0 on everything else: 77% of Ĉ's Frobenius mass sits on the support block, which holds only 30% of PΣP's mass (headline).

**What limits it.** The number of independent equations. The cluster-level measurement operator on Sym(B) has the following rank (dimension in parentheses):

| run | Sym(B) dim | I–I rank | I–B rank | both (cond.) |
|---|---|---|---|---|
| headline | 19306 | 12 (= K) | 55 | 67 (cond. 118) |
| pixels generic θ0.75 | 19306 | – | – | 93 |
| rff256 θ0.1 | 32896 | 2 | 155 | 157 |
| rff1024 θ0.1 | 524800 | 2 | 646 | 648 (cond. 7200) |

Clones multiply the equations (about 22k active lateral equations in the headline) but not the rank (F10). The I–I family alone gives only the K part energies; almost everything comes from the I–B rows (Σh_k)_b on each part's support. The weights store a parts-sketch, not Σ.

**Agreement with the other findings.**

- **Readout best mid-training: confirmed.**
  - Headline: the contrast cosine rises 0.35 → 0.58 at step 55, then falls to 0.45–0.50 as pruning proceeds.
  - rff1024: 0.18 → 0.67 at step 55 → 0.48.
  - Mid-training readouts violate the equilibrium assumption and are labelled as such.
- **Parts-sum: disagrees with F10.** On MNIST the parts-sum is *worse* than masked LS in every run (headline 0.41 vs 0.50; rff1024 0.39 vs 0.49), where F10 found it better on synthetic block covariances. MNIST parts overlap and Σ is signed. The LS fit uses every row (Σh_k)_b on the support, a rank-55 sketch, while parts-sum keeps only rank K + 1. (Parts-sum here uses c_k from the final instantaneous I–B row sums of the stored members, i.e. weights only.)
- **I–B vs I–I:** the I–B family carries almost all the information (LS I–B ≈ LS both). The naive pseudo-inverse is worse than LS in every run. The untrained-network readout equals the identity baseline.

## 5. Moment-based sample synthesis (export item 6)

Second moments do not determine p(x). Each generator adds a model, and I say which.

**Pixels: Gaussian completion.** x ~ N(x̄, C/scale²).

- x̄ is the training mean. It is **not** in the weights; it is outside information.
- C is the readout in the parameters-known setting, in absolute units.
- Negative eigenvalues are clipped (mass reported per row).
- Unconstrained directions get zero variance (ridge minimum norm).
- Samples are clipped to [0, 1] for display and MMD. This changes the moments, so moment checks are also reported unclipped.

**RFF: exploratory.** g ~ N(μ_g, C), then the pre-image x* = argmin_{x∈[0,1]¹⁹⁶} ‖φ(x) − g‖².

- Projected Adam: 300 steps, lr 0.02, from the mean image, identical for every row.
- A Gaussian g is generally off the image of φ, so the optimiser, init and box shape the result.
- The achieved relative residual is exported.
- A "real data, pre-image round trip" row shows what the generator does to real digits.

**Rows**, all with the same seeds within a sheet: real data / true-moment generator (Σ_g: a same-generator oracle reference, not a universal ceiling) / weight-reconstructed generator / baseline generators (diag Σ, untrained-network readout).

**Metric:** unbiased squared MMD (Gaussian kernel, median bandwidth) to 1000 held-out digits, plus moment consistency.

Squared MMD to held-out digits (lower is better; real training digits vs held-out: −0.0001):

| sheet | true-moment | weight-reconstructed | diag | untrained |
|---|---|---|---|---|
| pixels θ0.45 (3 parts) | 0.0124 | 0.0345 | 0.0147 | 0.40 |
| pixels θ0.75 (7 parts) | 0.0124 | 0.0296 | 0.0147 | 0.40 |
| pixels θ0.9 (12 parts) | 0.0124 | 0.0273 | 0.0147 | 0.40 |
| rff256 θ0.1 (2 parts) | 0.050 | 0.084 | 0.094 | 0.35 |
| rff1024 θ0.1 (2 parts) | 0.020 | 0.051 | 0.046 | 0.35 |

**Pixels.** The weight-reconstructed generator carries only 21–35% of the true variance (trace ratio), almost all of it on the parts' pixels.

- Its samples look like the mean digit plus a few flickering blobs.
- It is worse than the diag baseline in MMD, and gets better as more parts are populated.
- The negative eigen-mass clipped is 13–18%.
- Moment check: sample covariances match their own targets (unclipped error 5–9%, i.e. sampling noise), but they are 80–85% away from the true Σ. That is the readout error from §4 showing up as samples.

**RFF.**

- The pre-image round trip of real digits is exact (residual 1e-6), so the inverse map itself is fine.
- Gaussian feature draws are not on the image of φ: pre-image residual 0.43–0.60 for the true covariance, 0.9–1.4 for the reconstructed one.
- The weight-reconstructed samples are blurry 0/1 blends, because the two learned parts are a "1" detector and a "0" detector (the two signed ends of the leading kernel-PCA axis split into nonnegative parts).
- They are better than diag at D = 256 and on par at D = 1024. Larger D helps the true-moment reference (0.050 → 0.020) more than it helps the reconstruction.

Blurry digit-like blobs are the honest ceiling of a second-moment generator here. The interesting comparison is reconstructed vs true per θ: the reconstruction improves as more parts are populated, but stays far from the true-moment generator.

## 6. Costs

| stage | compute | estimated cost |
|---|---|---|
| Pilots | 4 grids, 86 small units, about 0.4 L4-hours | ≈ $0.35 |
| Production N = 512 | 38 units on Modal CPU (8 cores), 1456 s | ≈ $0.15 |
| Production N = 2048, first attempt (float32 readouts; discarded, §7) | about 1.5 GPU-hours | ≈ $1.5 |
| Production N = 2048, final | 76 units on A10 (L4 queue was starved), 10,626 GPU-s | ≈ $3.3 |
| **Total** | | **≈ $5–6** |

Costs are per-unit timing × list price, excluding container start-up. Local post-processing (UMAP, exports, samples) ran on the VPS CPU.

## 7. Surprises, failures and fixes

1. **Extinction instead of fragmentation** (§3). This is the main qualitative result. The note's "θ sets how many modes are resolved" holds only up to about 12 parts, and only with a heavy-tailed init.
2. **N-dependence through ω and the lateral init.** At fixed w_min = 1e-3 and U(0, 0.2) lateral init, N = 2048 was extinct where N = 512 was alive. Rescaling both by 1/N fixed it; ω is the right variable (F5).
3. **float32 conjugate gradients diverged** on N = 2048 problems with about 4M lateral equations (relative residual 0.5–31 instead of 1e-7, giving negative cosines in 2 of 67 runs). All readouts were moved to float64 and every N = 2048 run was redone. Final readouts now all converge. 4 snapshot readouts (out of about 2700) have residual 1e-4 to 0.17 at the 300-iteration cap; they are flagged by `ro_info_cg_rel_res` in the curves.
4. **Infrastructure.** A stale Modal image ran old code for the first production launch; I cancelled it and wiped its outputs. L4 capacity was starved for about 20 minutes, so the N = 512 runs moved to Modal CPU and the N = 2048 runs to an L4/A10G/T4 fallback list (all landed on A10). Nine units hit a ZeroDivisionError when CG converged exactly in one step (one-part states); this was guarded and the units rerun.
5. **Clones.** As in the synthetic chapters, every cluster is a set of identical neurons. There is no diversity to make an "expansion code" from in this setting.

## 8. What the exports contain (for the page author)

- **Featured runs** (36 snapshots each), with full exports:
  - pixels N = 2048 heavy, θ ∈ {0.3, 0.45, 0.6, 0.75, 0.9} (headline θ = 0.9);
  - pixels N = 2048 generic θ = 0.75;
  - rff256 and rff1024 N = 2048 heavy θ = 0.1.
- **Every featured run has:** curves, similarity matrices H Σ Hᵀ (256 neurons, fixed order), UMAP frames, graph edges, cluster parts / response-weighted averages / top digits / tuning, and readout spectra. Ĉ arrays exist for B ≤ 256.
- **All 114 runs** have metadata and final metrics in `index.json`.
- **Samples:** sheets for pixels θ = 0.45/0.75/0.9 and rff256/rff1024 θ = 0.1.
- **Filmstrip:** 6 frames of the headline run.
- **The neuron subsample is stratified** (every cluster has at least min(size, 4) members, the rest proportional to size). Show true cluster sizes from `clusters[].size`, not counts in the sample.
- **The dead pool** is a real cluster (flagged `dead`). It dominates the neuron count at high θ, and the page should say what it is: neurons with every synapse at the floor, floating at the floor-weighted centroid.
- **UMAP** uses the Hellinger metric (the natural metric for probability vectors; it is √h in Euclidean space). It is computed at the final snapshot and warm-started backwards, with each frame Procrustes-aligned to the next. Boundary nodes are not included in the map: each e_b is equidistant from every other under Hellinger, so their UMAP placement would be arbitrary. For the pixel lift, the graph instead offers the **Tutte layout** H·(pixel xy), the literal harmonic embedding of each neuron into the image plane.
- **Mid-training readouts** in `curves` are instantaneous-weight readouts and violate the equilibrium assumption. Label them as such.

## 9. Not done

- No third lift (reason in §1).
- No hinge penalty for clipped edges, and no fitted-θ offset in the contrast setting. The contrast score projects out 11ᵀ instead.
- The parts-sum estimator is reported only in this document (§4), not in `index.json`.
- Sampling uses the final readout only, not the better mid-training readout.
- The filmstrip's first frame is step 1, not step 0 (the log-spaced pick); the full series in `similarity.u8` starts at step 0.
