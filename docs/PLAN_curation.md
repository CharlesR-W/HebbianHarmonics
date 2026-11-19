# Plan: curated explorer (2026-11-19)

## The user's request (paraphrased, with their answers to clarifying questions)

- Make the story cleaner and more curated. Main highlights: (1) the harmonic embedding,
  (2) recovering the input statistics from the weights, (3) the Perron-eigenvector
  interpretation of clusters, (4) whatever was achieved in the nonlinear case.
- Story spine (user's choice): **linear first, then nonlinear.** Main path: harmonic embedding ->
  Hebb's covariance rule gives clusters on Perron vectors of Sigma - theta 11^T -> reading input
  statistics back from the weights (exact for the linear distance rule) -> nonlinear neurons:
  what carries over, what we got. Annoying / secondary results go to an appendix.
- Rename "The talk" tab to **"Primer"** and update it to the new story.
- **Every graphic must be clearly either neuron space or input (vector) space.**
  Neuron-space graphics (network diagrams: nodes = neurons, edges = synapses) place nodes at
  random (seeded, fixed) positions, so physical layout is visibly unrelated to function.
  Vector-space graphics (simplex for B = 3, MDS/projection for B > 3, heatmaps of H) keep the
  current organised layouts.
- Precompute big runs with many more neurons than inputs (browser cannot run them), replayed
  in the browser: (a) B = 3, N ~ 300 films; (b) B = 8-16, N ~ 300-500; (c) nonlinear vs linear.
- Drop MNIST entirely (synthetic data only). Done for the VPS data (/var/tmp/hebb-mnist,
  2.4 GB deleted); remove remaining MNIST text from the site.

## Tab structure after the change

Main path (in order):
1. `primer` — renamed `talk` (files `talk_*.{html,js}` may be renamed `primer_*`; keep `#talk`
   as an alias that redirects to `#primer`).
2. `idea` — the question, the linear resistor network, the harmonic embedding H (rows are
   probability vectors over inputs; simplex picture), both rules stated: the note's covariance
   rule, and the distance rule as the energy gradient (envelope theorem). Linear only; the
   on-site potential is introduced in the nonlinear chapter, not here.
3. `perron` (new, built mostly from `old.html`'s first half + the v1 live sim `ch_old.js`):
   covariance rule; without pruning one cluster (full-support Perron vector); clusters are
   parts = non-negative Perron vectors of principal submatrices of A = Sigma - theta 11^T;
   worked 3-input cases; many inputs (blocks, hierarchy, ring: parts vs eigenvectors), with the
   new precomputed N ~ 300-500 runs. Clones / allocation-from-transient summarised in two
   sentences with an appendix link.
4. `readout` — recovering input statistics from the weights. Linear distance rule: exact
   recovery of P C P with enough live edges (existing readout chapter, trimmed); one paragraph
   on the covariance rule's readout being a sketch of the parts. New big-N results (rank and
   cosine vs N at fixed B).
5. `nonlinear` (new) — on-site potentials as transfer functions; response functions as the
   clustered objects; what carries over (exact fixed point w = (b - D^2/2)_+/alpha, clusters
   in function space, readout of a partial distance geometry -> MDS); what we got and didn't
   (diversity within clusters is small, ~2% of variance; no allocation law). Precomputed
   linear-vs-rectifier big runs side by side.

Appendix (visually separated in the tab bar, e.g. a divider and smaller "Appendix" label):
- `fusion` (two-neuron bistability, N-neuron fusion details), `many` (hysteresis, resolution
  law), `diversity`, `allocation`, `old` (renamed "More on the covariance rule": clones,
  allocation from the transient, rectified output on synthetic data only; MNIST paragraphs
  removed), `method` (choices & caveats).
- Each appendix chapter starts with one line saying what it adds and linking back to the main path.

## Graphics convention (applies to all chapters and the primer)

- Two kinds of panel, each tagged with a small badge in the panel title: "neuron space" or
  "input space" (consistent colour/tint per kind, defined once in CSS tokens).
- Neuron space: network diagram, seeded random node positions (uniform in the panel with a
  minimum spacing; inputs drawn as squares, also at random positions), edges = synapses with
  width ~ conductance. Never position nodes by embedding.
- Input space: points at their embedding (simplex corners = inputs for B = 3; classical MDS of
  D for nonlinear or B > 3, stated). Synapse lines are not drawn here by default (cluster
  membership by colour); if kept, a toggle labelled "show synapses".
- Where both matter, show them side by side with linked hover/selection (hover a neuron in one,
  highlight it in the other) and the same cluster colours.

## Precomputed big runs (site/data/big/, scripts in hebb/big/)

All synthetic. Each run's JSON carries its config, seed, convergence residual, and the script
that produced it. Films are quantised (uint8/uint16 or float16 in base64 or .bin) with ~60-120
frames on a log-time axis; budget ~10-15 MB total.

A. B = 3, N = 300 films:
   - covariance rule (v1 engine `hebb/core.py`): independent inputs at theta in {0.2, 0.4};
     correlated pair (rho12 = 0.6) at theta = 0.5. Per frame: H (N x 3), lateral weights above
     floor (sparse edge list), cluster labels at the end.
   - distance rule (`hebb/onsite.py`), linear and rectifier (klo = 3, sigma_t = 0.7), b in
     {small, percolation, fused}, alpha = 1 (and one alpha = 0.1). Per frame: positions
     (linear: H rows; rectifier: best-linear-fit coefficients in the simplex AND an MDS of D,
     stated which is drawn), live edge list.
B. B = 12-16, N = 400:
   - covariance rule: blocks (3 x 4), hierarchy (8 or 16), ring (16) at 2-3 theta each. Output:
     H at the end (+ ~40 frames), parts (cluster centroids), Perron vectors of A_SS on each
     support, top eigenvectors of Sigma, cosine tables, cluster sizes.
   - distance rule, linear vs rectifier, blocks and power-law spectra, 3 b values: clusters,
     diameters, readout (operator rank, cosine to P C_hat P and P Sigma P), nonlinear distance
     MDS, within-cluster diversity measures as in `hebb/sweeps2/diversity.py`.
C. Readout vs N: B = 12, N in {40, 80, 160, 320, 500}, linear distance rule, 3 seeds, 3 b.

Compute: the linear runs are cheap locally. The rectifier at N = 400 costs ~1-4 s per step on
the VPS (batched N x N solves per sample), so nonlinear runs go to Modal (CPU fan-out, or GPU
float64 batched solves), per the user's standing instruction for anything over ~10 minutes.

## Work split

1. Precompute (one agent): `hebb/big/`, Modal, `site/data/big/` + `site/data/big/README.md`
   documenting formats; findings appended to `docs/findings_v2.md` under a new heading.
2. Shell + graphics (one agent, in parallel): tab rename/reorder/appendix nav, `#talk` alias,
   a shared `netview.js` (neuron-space network diagram with seeded random layout, linked
   highlighting API) and an input-space badge convention; convert every existing figure that
   draws a network to the convention; a generic film player for the big-run data format.
3. Writing (after 1 and 2): primer slides; idea, perron, readout, nonlinear chapters; appendix
   intros; MNIST removal; README/SPEC/DESIGN updates.
4. Review: an independent reviewer checks claims against findings and the page in a headless
   browser (desktop + mobile), then fixes.

## Revisions after plan review (supersede the sections above where they differ)

R1. **Rectifier readout is not a covariance readout.** Covariance rank/cosine only in the linear
    arm. For the rectifier: distances implied by the weights (live: D^2 = 2(b - alpha w); dead:
    D^2 >= 2b) vs directly measured response distances; number of pairs identified; MDS of the
    measured distances labelled "response-function space" and any completed-distance MDS
    labelled an approximation. No nonlinear simplex on the main path.
R2. **Matched linear-vs-rectifier protocol.** Same seeds, attachments, initial W, training and
    evaluation banks, same b schedule. Report both same-b comparisons and comparisons at matched
    edge density / component diameter, labelled as such. Keep a homogeneous-rectifier control
    (sigma_t = 0) in the data even if it is not a headline film. "About 2%" stays attached to the
    controlled rank-one experiment; new runs report their own numbers.
R3. **Perron runs.** Floor scaled with N so that the total floor conductance per neuron
    omega = N * w_min matches the live demo (~0.009): w_min = 0.009 / N; check sensitivity at
    omega/3. Support of a part = inputs with above-floor input synapses (not nonzero H).
    Report per cluster: member spread, Perron fit (cosine of centroid with the Perron vector of
    A_SS on its support, top-eigenvalue test, off-support violations), best eigenvector of Sigma.
    The one-cluster statement is for entrywise-positive A only.
R4. **Pilot first, byte budget.** Pilot one near-percolation and one fused rectifier run on Modal
    and one covariance-rule film locally; record time, memory, residuals, encoded bytes; only
    then fan out. Chunk evaluation-bank solves. Export positions, weights and responses from the
    same state (post-update weights with post-update equilibrium). Display edges are bounded
    (<= 1500 per frame, the strongest above floor); full weights stay in analysis outputs, not
    on the site. Total site/data/big budget <= 15 MB; films lazy-loaded.
R5. **Readout wording.** "Exact identifiability of the training-bank contrast second moment
    P C_hat P, for stationary noiseless weights, known attachments and rule parameters, and a
    full-rank, well-conditioned live-edge operator." N-sweep plots rank and conditioning and
    separates fixed-point, inverse and sampling error. State the rule change (covariance rule ->
    distance rule; which synapses are plastic) at the start of the readout chapter.

## Film data format (shared contract between exporter and player)

`site/data/big/index.json`: list of `{id, title, family, rule, nonlin, B, N, file, bytes}`.
Each film: `site/data/big/<id>.json` (metadata, small arrays) + `site/data/big/<id>.bin`
(little-endian typed arrays at the byte offsets given in the JSON).

```
{ "id", "title", "rule": "cov"|"dist", "nonlin": "linear"|"rect"|"rect_homog",
  "B", "N", "seed", "config": {...all knobs...}, "script": "hebb/big/....py",
  "space": "simplex" | "pca",          // simplex: coords are H rows (B = 3, linear only)
                                         // pca: function-space PCA, axes fixed by the final frame
  "space_label": "input space (harmonic embedding)" | "response-function space (PCA)",
  "frames": [{"t": float, "residual": float, "n_live": int, "n_components": int}],
  "arrays": {
    "coords": {"offset", "dtype": "uint16", "shape": [F, N, D], "lo": [..D], "hi": [..D]},
    "H":      {"offset", "dtype": "uint8",  "shape": [F, N, B]}  // linear only; value/255
    "edges":  {"offset", "dtype": "uint16", "shape": [F, E, 2], "count": [F ints]},
    "edge_w": {"offset", "dtype": "uint8",  "shape": [F, E], "wmax": float}  // w = wmax*(v/255)^2
  },
  "final": { "labels": [N ints], "clusters": [{"size", "support": [..], "centroid": [..B],
             "spread", "perron_cos", "top_eig_ok", "offsupport_violation", "best_eig_cos"}],
             "metrics": {...} },
  "network_layout_seed": int            // the player draws neuron space with this seed
}
```

## Story (shared by all writers; 2026-11-19, after the big runs)

One sentence per step; chapters expand, the primer compresses to one or two slides each.

1. **Question.** Hebb's rule is local; what does the network compute? Model: a linear resistor
   network, inputs clamped.
2. **Harmonic embedding** (idea). Each interior neuron's voltage is u_i = h_i · g; the row h_i
   of the discrete Poisson kernel H is a probability vector over the inputs, set by the
   weights alone. It is a point in the simplex of inputs (input space), unrelated to where the
   neuron "is" in the network (neuron space). Covariance between neurons is an inner product
   of embeddings, so Hebb's rule acts on embeddings.
3. **Perron parts** (perron). The note's covariance rule with a threshold θ: a cluster is a
   non-negative Perron vector of a principal submatrix of A = Σ − θ11ᵀ (a "part"), not an
   eigenvector of Σ. θ sets the support size. Big runs (N = 300-400, B up to 16, blocks,
   hierarchy, ring): every populated cluster fits its predicted Perron vector to cosine
   ≥ 0.9999 and passes the stability test; the best Σ eigenvector reaches only 0.49-0.87 at B = 12-16 (B = 3 independent: Σ = I).
   Members are clones; which parts are populated depends on the start (appendix).
4. **Reading the weights** (readout). Change of rule, stated: the distance rule
   dw = b − αw − D²/2 is the gradient of the network's own equilibrium energy plus a weight
   cost (envelope theorem), and it is a close cousin of the covariance rule (same Hebbian
   term minus each neuron's power). Its fixed point w = (b − D²/2)₊/α makes each live
   synapse a measurement of a distance between embeddings, hence a linear equation on Σ.
   With stationary noiseless weights, known attachments and rule parameters and a full-rank,
   well-conditioned operator, the training-bank contrast second moment P Ĉ P is identified
   exactly. More neurons than inputs is what makes it work: at B = 12, rank 66/66 is reached
   at N = 80-320 depending on b, conditioning improves with N, and recovery is 1.000 with
   errors separated (fixed point ≤ 2e-9, inverse ≤ 1.4e-3, sampling ~0.09). Limits: only
   contrasts (P C P), the uncentred bank moment, not the input distribution; one full-rank
   but ill-conditioned run recovers only 0.987. The covariance rule's readout is only a
   sketch of the parts (one paragraph, appendix link).
5. **Nonlinear neurons** (nonlinear). On-site potential = transfer function; response
   functions F_i(g) replace embeddings; the distance rule and its exact fixed point carry
   over, now in response-function space. What we got: live synapses still state exact
   response distances (training bank, error ≤ 5e-6; 1.6-7.1% median on held-out inputs; 58 rectifier runs), dead
   pairs give lower bounds (≥ 99.95% hold), so the weights fix a partial response geometry
   (MDS good once fused, poor at percolation). Within-cluster diversity is real but small
   (~1-2% of variance in compact components with heterogeneous thresholds vs ~0.1-0.3% linear) and fusion
   linearises neurons; threshold spread delays fusion. No allocation law (appendix). Honest
   bottom line: nonlinearity keeps the geometry and readout of distances, adds a little
   diversity, and does not deliver the expansion code or a spectral filter in the basic model.
6. **What we get / open** (primer end, idea box): harmonic embeddings as the natural
   variables; clusters as Perron parts of the thresholded covariance; input statistics from
   the weights; plasticity parameters as clustering knobs; open: directed networks,
   allocation, noise, diversity.
