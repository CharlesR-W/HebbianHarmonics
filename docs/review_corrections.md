# Corrections after independent review (GPT-6.1 Sol, round 1)

These override `findings.md` (F), `findings_private_noise.md` (P) and `nonlin.md` (N)
wherever they differ. Page text must use the corrected wording.

Verified correct by the reviewer (independent scripts): F0 shift identity; F1 part
equation, active-support Jacobian, corollaries 1, 2, 4, extinction bound; F10 parts-sum
equations (restricted as below); N dual-activation formula, monotone and convex.
Independent numbers: stability rates −0.333/+1.000 vs predicted −1/3/+1; two-block
transitions bracket 0.1 and 0.6; active-edge errors < 1e-15 at γ = 1.4; clone spread
4e-16; lone-neuron allocation exponent 2.76 vs F6's 2.8.

1. **P F11 (multiplicative noise) — claim withdrawn pending rerun.** The integrator
   applied the deterministic drift, then the multiplicative factor, then the clip. A
   floored synapse's pre-noise weight is negative (e.g. −0.0024 at dt = 0.05), so no
   positive multiplier can lift it: it is numerically impossible, not exponentially
   rare, for it to leave the floor. "Multiplicative noise never hops at the default
   floor" is therefore an artefact as measured. The stationary density quoted is for a
   reflected Itô SDE that the code does not integrate (exp-multiplication adds an Itô
   drift term). Status: RESOLVED 2026-11-17 — rerun with validated reflected / log-space Itô schemes; `findings_private_noise.md` is the revised text. At the default floor multiplicative noise still never equilibrates, now for a verified reason (ln escape time 112 to 16,000); additive and input-noise results were redone with a reflected scheme (beta moves by 0.0-0.4; conclusions unchanged).
2. **N §2.2 — "the identifiable set is unchanged" is false.** With rectified output
   the equations also involve boundary variances C_bb. Example: one cloned cluster
   h = (1/2, 1/2), C = I, two I–B observations and one within-cluster I–I observation:
   Jacobian rank 3 (rectified) vs 2 (linear). Rectification can reveal covariance
   directions invisible to the linear readout. Unknown θ and unknown λ/η are not simple
   gauge ambiguities there. Say: identifiability must be assessed by the local Jacobian
   rank; it is at least not smaller in the example checked.
3. **F F1 corollary 3 — amended.** Vertex b is a part iff Σ_bb > θ AND Σ_ab ≤ θ for
   every a ≠ b. The stability test "c is the top eigenvalue of A_SS and (Ah)_b < 0
   strictly off S" is sufficient (the nondegenerate case), not necessary: with
   Σ = diag(2, 1), θ = 0 the dominant vertex is stable although its off-support margin
   is zero.
4. **P F12 — qualified.** The Rayleigh quotient is a Lyapunov function for the
   zero-floor projected flow; with a positive floor it can decrease slightly (measured
   dR/dt = −2e-6 in an example), not only by round-off. The 1-D barrier formula is a
   linearisation near a vertex with the resident frozen, not an exact quasi-potential.
   "The deepest well is the loudest part" is an empirical observation in the tested
   families, not a deduction.
5. **F F3 — heuristic, say so everywhere.** θ_c = γ·κ0 is an init-dependent heuristic
   crossover (expected initial lateral drift), which predicted all tested transitions
   within 0.03–0.08; it is not a derived phase boundary. κ0 assumes independent
   exchangeable initial embeddings (with dense lateral init the mean initial
   cross-covariance differs: 0.358 vs 1/3 in a small check). The packing inequality is
   justified when inter-cluster edges are clipped.
6. **F F4 / F10 — hypotheses.** "Any allocation is a fixed point" holds for mutually
   compatible, disconnected parts in the zero-floor limit; a finite floor couples
   clusters and makes equilibria weakly allocation-dependent (O(ω/c)). The parts-sum
   estimator satisfies the active equations for disjoint zero-floor parts; its rank is
   at most K + 1 (K after removing the θ11ᵀ term).
7. **F F7, P F14 — scope.** Permutation symmetry proves identical neurons stay
   identical, not that all members become clones; clones are an empirical result in
   every tested condition (spread < 1e-5). The power-law / Boltzmann allocation fits and
   the θ-independence under input noise are fits on the tested families and three
   thresholds, not general laws.
8. **Note's claim about p(stimulus).** Even perfect recovery of the centred covariance
   does not recover the distribution: two-atom mixtures with weights p and 1 − p have
   the same centred feature covariance up to the factor p(1 − p) for any feature map,
   and without the mean the atoms' locations are lost. Covariance recovery and
   distribution recovery are separate; the latter needs the mean and further assumptions.
9. **F F10 sparse wiring on signed power-law.** Chapter `many` measured an improvement
   (0.50 → 0.71; 0.87 → 0.90) where F10 says "does not improve". Treat as
   setting-dependent; quote only what the page computes live, with its scope.
