# site/data/big: precomputed big runs (N >> B)

Synthetic data only. All files here are produced by the scripts in `hebb/big/` (2026-11-19).
Total size: about 7.4 MB (budget 15 MB). The numbers and what they show are in
`docs/findings_v2.md`, section "Big runs (N ≫ B), 2026-11-19".

| file | what | script |
|---|---|---|
| `index.json` | list of films `{id, title, family, rule, nonlin, B, N, file, bytes}` (`bytes` = .json + .bin) | written by every exporter |
| `<id>.json` + `<id>.bin` | one film (18 films, list below) | `cov_runs.py` (rule `cov`), `dist_runs.py` + `dist_export.py` (rule `dist`) |
| `cov_all.json` | covariance rule: every run (films and non-film runs, three seeds for B = 3, floor sensitivity), with the per-cluster Perron report | `hebb/big/cov_runs.py` |
| `dist_A.json`, `dist_B.json` | distance rule, families A (B = 3, N = 300) and B (B = 12, N = 400): every run, the b-grid sweeps, matched comparisons | `hebb/big/dist_runs.py` (runs), `hebb/big/dist_export.py` (analysis) |
| `readout_vs_N.json` | family C: covariance readout vs N at B = 12 | `hebb/big/readout_vs_n.py` |

Regenerate: `uv run python -m hebb.big.cov_runs all`; `uv run python -m hebb.big.dist_runs local all`
(linear units) plus the Modal fan-out for the rectifier units (`hebb/big/modal_app.py`, see its
docstring; config built from `dist_runs.units()`), then `uv run python -m hebb.big.dist_export`;
`uv run python -m hebb.big.readout_vs_n`. Raw outputs (full weight matrices, responses on both
banks) go to `scratch/out/big/` (gitignored), not to the site.

## Film format

`<id>.json` (metadata and small arrays) and `<id>.bin` (little-endian typed arrays at byte
offsets given in `json.arrays`; every offset is a multiple of 8).

```
{ "id", "title", "family": "A"|"B", "rule": "cov"|"dist", "nonlin": "linear"|"rect"|"rect_homog",
  "B", "N", "seed", "config": {...all knobs, including Sigma...}, "script",
  "space": "simplex" | "pca",
  "space_label": "input space (harmonic embedding)" | "response-function space (PCA)",
  "pca": null | {"explained_var": [3 floats], "total_var": float, "features": "..."},
  "frames": [{"t", "step", "residual", "n_live", "n_components"}],
  "bin": "<id>.bin",
  "arrays": {
    "coords": {"offset", "dtype": "uint16", "shape": [F, N, 3], "lo": [3], "hi": [3]},   x = lo + (hi - lo) v / 65535
    "H":      {"offset", "dtype": "uint8",  "shape": [F, N, B]},                          h = v / 255   (linear only)
    "edges":  {"offset", "dtype": "uint16", "shape": [F, E, 2], "count": [F ints]},       neuron index pairs i < j
    "edge_w": {"offset", "dtype": "uint8",  "shape": [F, E], "wmax": float},              w = wmax (v / 255)^2
    "in_w":   {"offset", "dtype": "uint8",  "shape": [F_in, N, B], "wmax": float}         w = wmax (v / 255)^2
  },
  "edge_threshold": "...",
  "final": {"labels": [N ints], "clusters": [...], "metrics": {...}},
  "network_layout_seed": int }
```

- **Frames.** `t` is model time (`step * dt`), `step` the Euler step. Frames are roughly
  log-spaced from t = 0 to the final (converged) state; the last frame is the final state.
  `residual` is the projected stationarity residual max|dw/dt| of that frame's state (cov rule:
  `core.stationarity`; dist rule: floor-clipped `b - alpha w - D^2/2`). `n_live` = number of
  live synapses (cov: lateral w above the floor, `w > w_min (1 + 1e-3)`; dist: `w > 1e-9`),
  `n_components` = connected components of that live graph.
- **coords.** `space = "simplex"`: coords are the rows of H (barycentric, sum 1; B = 3 linear
  runs). `space = "pca"`: function-space PCA with axes fixed by the final frame. The features
  are chosen so that Euclidean distance = RMS response distance D_ij, so the 3-D coordinates
  are classical MDS of the measured D at the final frame (cov rule: `x_i = Sigma^{1/2} h_i`;
  dist rule: `x_i = F_i / sqrt(S)` on the training bank, S = 256). `pca.explained_var` /
  `total_var` says how much of the geometry the three axes hold (0.62-1.0 here). Every frame
  uses the same axes and mean, so motion is real motion.
- **Same state.** In every frame the coordinates, H, edges and residual come from one state:
  the weights W_n and their own equilibrium (R4).
- **edges.** The strongest live synapses, at most 1500 per frame, strongest first; only the
  first `count[f]` rows of frame f are valid (the rest are zero padding). Full weights are not on
  the site (`scratch/out/big/`).
- **in_w** (addition to the plan's contract; the player may ignore it): input-synapse
  conductances. Cov rule: the plastic W_IB per frame (`F_in = F`). Dist rule: the fixed
  attachments A (`F_in = 1`).
- **final.labels.** Clusters at the final frame, labelled 0, 1, ... by decreasing size;
  `-1` = dead neuron (cov rule: every input synapse at the floor). Cov rule: single linkage on H
  in L1, cut at the largest gap (`core.clusters_auto`), dead neurons excluded. Dist rule:
  connected components of the live graph.
- **final.clusters** (cov rule): per cluster `size`, `support` (inputs whose input synapse is
  above the floor, `w > w_min (1 + 1e-3)`, in a majority of members; `support_agree` = fraction of
  members agreeing), `centroid` (mean H row), `spread` / `spread_mean` (max / mean L1 distance of a
  member to the centroid), `c_input_conductance` (mean summed input conductance), `perron_cos`
  (cosine of the centroid with the Perron vector of `A_SS`, A = Sigma - theta 11^T, embedded in
  R^B; `perron_cos_on_support` on the support only), `perron_positive`, `top_eig_ok` (the
  eigenvector of A_SS closest to the centroid is the top one, it is positive, the top eigenvalue
  is simple and positive), `eig_gap`, `c_rayleigh` / `c_top` (Rayleigh quotient of the centroid
  on A_SS / top eigenvalue), `offsupport_violation` = max over b not in S of (A h)_b (a part needs
  it <= 0; positive = violated), `offsupport_mass` (H mass off the support, from the floor),
  `best_eig_cos` (largest |cosine| of the centroid with an eigenvector of Sigma; degenerate
  eigenvalues grouped, cosine with the projection on the eigenspace), `best_eig_rank` (0 = top) and
  `best_eig_multiplicity`, `nearest_stable_part_cos` / `_support` (closest stable part from the
  exact enumeration of all parts, `common.enumerate_parts`).
- **final.clusters** (dist rule): the 30 largest components: `size`, `diam_max` / `diam_mean`
  (max / mean RMS distance D of a member to the prototype, training bank), `span` (largest
  pairwise D inside the component), linear arm also `centroid` (mean H row).
- **final.metrics**: run metrics (convergence: `converged`, `residual`, `steps`, `t`; cov: K,
  dead count, enumerated and populated stable parts; dist: K, live edges, density, diameters,
  `span_over_range` = span / sqrt(2b) (> 1 marks a chain), fixed-point deviations, diversity).
- **network_layout_seed**: seed for the neuron-space (random) layout in the player.

## Films

All converged (projected residual < 1e-6; actual ≤ 1e-7 cov, ≤ 1e-9 dist).

| id | rule | model | frames | space |
|---|---|---|---|---|
| `A_cov_ind_th0.4` | cov | B = 3 independent unit inputs, theta = 0.4, N = 300 | 59 | simplex |
| `A_cov_ind_th0.2` | cov | same, theta = 0.2 (bound population, one corner) | 60 | simplex |
| `A_cov_pair_th0.5` | cov | rho12 = 0.6, theta = 0.5 | 61 | simplex |
| `A_dist_ind3_lin_{fragmented,percolation,fused}` | dist | linear, B = 3 independent, N = 300, b = 0.0003 / 0.0017 / 0.01, alpha = 1 | 21-28 | simplex |
| `A_dist_ind3_rect_{fragmented,percolation,fused}` | dist | rectifier (klo = 3, khi = 0, sigma_t = 0.7), same seed, attachments, banks and b | 20-26 | pca |
| `A_dist_ind3_{lin,rect}_percolation_a0.1` | dist | as above at b = 0.0017, alpha = 0.1 | 45-46 | simplex / pca |
| `B_cov_blocks12_th0.35` | cov | 12 inputs, 3 blocks of 4 (within 0.5, across 0.1), theta = 0.35, N = 400 | 34 | pca |
| `B_cov_hier16_th0.35` | cov | 16 inputs, binary hierarchy (0.8 / 0.5 / 0.25 / 0.05), theta = 0.35 | 34 | pca |
| `B_cov_ring16_th0.5` | cov | ring of 16, exp(-d^2 / (2 * 1.5^2)), theta = 0.5 | 35 | pca |
| `B_dist_{blocks,power}_{lin,rect}_percolation` | dist | B = 12, N = 400, alpha = 1, b = 0.018 (blocks) / 0.045 (power) | 23-31 | pca |

Regime names (fragmented / percolation / fused) are those of the **linear** arm at that b
(fragmented: almost all neurons singletons or tiny groups; percolation: the largest component
holds 1/3-1/2 of the neurons; fused: one or a few components). The rectifier at the same b can
be in a different regime (it is in family A: see `dist_A.json` `matched`).

The distance-rule films have fewer frames than planned (20-46, not 60-120): from W = 0 the flow
converges in 20-80 Euler steps (dt = 0.5), and every step is a frame until then.

## Covariance rule (`cov_all.json`)

Engine: `hebb/core.py` (`init_state`, `step`, `stationarity`, `harmonic`, `clusters_auto`),
mean-field (exact Sigma each step), dense masks, lambda = eta = gamma = 1, dt = 0.05,
W_IB ~ Exp(0.2), W_II starting at the floor, seed s = `np.random.default_rng(s)`.
Floor scaled with N (R3): `w_min = omega / N`, omega = 0.009 (the live demo's 90 x 1e-4), and a
sensitivity copy of every setting at omega / 3. Stop when the residual < 1e-7; equilibrium
claimed when < 1e-6.

Runs: B = 3, N = 300: (ind3, theta 0.4), (ind3, 0.2), (pair3, 0.5), seeds 0-2 at omega and seed 0
at omega/3. B = 12-16, N = 400, seed 0, omega and omega/3: blocks12 at theta 0.2 / 0.35 / 0.6;
hier16 at 0.2 / 0.35 / 0.6; ring16 at 0.3 / 0.5 / 0.7.

`runs[]`: `name` (preset), `theta`, `N`, `omega`, `seed`, `config`, `clusters` (as in
final.clusters above), `metrics`: K (all clusters), `K_major` (clusters with ≥ max(2, 2% N)
members), `n_dead`, `sizes`, `clusters_robust` and `cluster_cut_L1`, `n_parts` /
`n_stable_parts` (exact enumeration), `n_stable_populated` (distinct stable parts matched by a
major cluster at cosine > 0.98), `frac_neurons_on_stable_part`, `max_spread`, `kappa0` (mean
entry of Sigma), convergence fields, `seconds`, `Sigma_eigenvalues`, `stable_parts` (support,
h, c, margin, best eigenvector cosine; at most 64), `film` and `film_bytes` if a film was made.

## Distance rule (`dist_A.json`, `dist_B.json`)

Engine: `hebb/big/dist_engine.py` (`lin_run` numpy, `rect_run` torch float64, Cholesky Newton),
same semantics as `onsite.run` and validated against `hebb/onsite.py` to 1e-15 (W, U, step
counts identical; `hebb/big/validate.py`, `scratch/out/big/validate_cpu.json`). Projected Euler,
dt = min(0.5, 1/alpha), cold start W = 0, stop when max|dW|/dt < 1e-9 (max 40000 steps),
equilibrium claimed when the projected residual < 1e-6 (all 177 units: ≤ 9.6e-10).

Matched protocol (R2): for a (family, Sigma, N, seed) all three arms share the attachments
(s = 3 random inputs, Dirichlet(1) shares, total 1; stream [seed,0]), W = 0, the training bank
(S = 256, [seed,1]), the evaluation bank (4096, [seed,2]) and the b values; thresholds from
[seed,3]. Arms: `linear` (klo = khi = 0), `rect` (klo = 3, khi = 0, sigma_t = 0.7),
`rect_homog` (klo = 3, khi = 0, t = 0; control).

Units: family A (Sigma = I_3, N = 300): every arm on a 13-point b grid 3e-4 ... 0.3 (seed 0); at
the three regime b values seeds 0-2; at b = 0.0017 also alpha = 0.1. Family B (N = 400, B = 12,
Sigma `blocks` = `oa.sigma_blocks(12)` and `power` = `oa.sigma_power(12)`, as in
`site/data/sweeps2`): every arm on a 13-point grid 1e-3 ... 1 (seed 0); at the regime b values
seeds 0-1. Regimes: A: 0.0003 / 0.0017 / 0.01; B blocks: 0.003 / 0.018 / 0.1; B power: 0.003 /
0.045 / 0.18.

Top level: `meta` (scripts, grid, regimes, arms, `missing_units` (none)), `films` (unit name ->
film id and bytes), `matched[sigma]`, `runs[]`.

`runs[]` fields: `name`, `fam`, `sigma`, `N`, `arm`, `b`, `alpha`, `seed`, `converged`,
`residual`, `steps`, `t`, `seconds`, `device`, `equil_gmax` (max node-equation residual of the
final equilibrium), `K`, `K_multi`, `largest`, `largest_frac`, `n_live`, `edge_density`
(live edges / all pairs), `diam_max`, `diam_mean`, `span_over_range` (largest span over all components / sqrt(2b)),
`span_largest_over_range` (the largest component's span / sqrt(2b)), `fp_dev_live` (max over
live edges |alpha w - (b - D^2/2)|), `fp_slack_dead` (min over dead pairs D^2 - 2b), `components`
(30 largest, as in the films), and:

- `diversity` (evaluation bank, all arms; logic of `hebb/sweeps2/diversity.py`, its
  `cluster_measures` imported unchanged): `affine_err_mean` / `_max` (held-out relative error of
  an affine fit of each response from the full input: per-neuron nonlinearity),
  `frac_in_multi`, size-weighted over multi-member components `proto_err_w` (held-out relative
  error of member ≈ a + c prototype), `deff_w`, `rank_w`, `deff_resid_w`, `proto_err_max`, `proto_err_w_compact` (the same weighted proto_err over compact
  components only, span <= 2 sqrt(2b); null if none), `frac_in_compact`, `n_compact`, `n_multi_noncompact`, and
  `top_clusters` (the 10 largest components' measures). Responses are stored as float32, so
  linear-arm values below ~1e-6 are rounding.
- `readout` (linear arm, family B only): `hebb/sweeps2/readout.analyse` unchanged, fields as in
  `site/data/sweeps2/readout.json`: `n_live`, `dim` (66), `rank_1e8`, `rank_1e4`, `sv_max`,
  `sv_min_nonzero`, `condition` (= sv_max / sv_min_nonzero), `err_fixed_point`, `err_sampling`,
  `err_inverse_identifiable`, `identifiable_frac(_1e8)`, recoveries and baselines
  (`<method>_vs_<target>` with cos / rel_err / bures_rel; targets Chat = P C_hat P, Sigma =
  P Sigma P), and noise tests `noise0.01_...`, `noise0.0001_...`.
- `distance_readout` (rect and rect_homog; R1, no covariance readout): `frac_exact` (pairs whose
  D the weights give exactly: live edges), `frac_upper_bounded` (same component, not an edge:
  shortest-path bound), `frac_lower_only` (different components: only D ≥ sqrt(2b)),
  `edge_abs_err_train_max` (max |sqrt(2(b - alpha w)) - D_train| on live edges),
  `edge_rel_err_eval_median` / `_max` (same against the evaluation bank: sampling),
  `dead_lower_bound_holds_train` / `_eval` (fraction of cross-component pairs with D ≥ sqrt(2b)),
  `sp_over_eval_median` / `_max` (shortest path through live edges / evaluation-bank D, inside
  components), `mds_completed_vs_measured_eval_disparity` (Procrustes disparity of 3-D classical
  MDS of the completed distances vs of the measured evaluation-bank distances, largest component;
  0 = same up to similarity) and `mds_component_size`. The completed-distance MDS is an
  approximation; the measured-distance MDS is "response-function space".

`matched[sigma]`: `grid[arm]` (seed-0 b grid: K, K_multi, largest_frac, edge_density, diam_max,
span_over_range, span_largest_over_range, proto_err_w, proto_err_w_compact, deff_w, affine_err_mean, converged) and `matched[]`: for every
rectifier grid point, `linear_b_same_density` / `linear_b_same_largest` (the b at which the
linear arm has the same edge density / largest-component fraction, log-interpolated on the
monotone envelope; null outside the grid), `nearest_linear_by_density` (the linear grid point
closest in log edge density, with its structure and diversity) and `this_arm` (the same for the
rectifier point); `regime_match[]`: for each regime run (seed 0, rect and rect_homog) its edge density, the
linear b with the same density (interpolated at that density) and `ratio` = b / that linear b. Same-b comparisons are the `runs` at equal b; matched-density comparisons
are labelled as such.

## Readout vs N (`readout_vs_N.json`)

Linear distance rule, B = 12, `oa.sigma_blocks(12)`, alpha = 1, N in {40, 80, 160, 320, 500},
seeds 0-2, b in {0.01, 0.036, 0.1}, cold start, s = 3 random attachments; all 45 runs converged
(residual ≤ 1e-9). `runs[]`: N, seed, b, convergence, live edges, `rank_1e8` / `rank_1e4` (of 66),
`sv_max`, `sv_min_nonzero`, `condition`, the three errors (`err_fixed_point`, `err_sampling`,
`err_inverse_identifiable`, `identifiable_frac`), cluster structure (K, K_multi, largest,
diam_max, span_over_range), cosines (`ridge_psd_cos_Chat`, `..._Sigma`, `hinge_psd_*`,
baselines `base_identity_*`, `base_diag_*`, `base_untrained_*`), noise tests (`noise1e-4_cos`,
`noise1e-2_cos`). `summary[]`: per (b, N), [mean, sd] over seeds of n_live, rank, rank_1e4,
cos_Chat, cos_Sigma, log10_condition, err_sampling, err_inverse, K, largest; max fixed-point
error; all_converged. What is claimed (R5): exact identifiability of the training-bank contrast
second moment P C_hat P, for stationary noiseless weights, known attachments and rule
parameters, and a full-rank, well-conditioned live-edge operator.

## Compute record

- Covariance rule, linear distance-rule units, family C: local (4 vCPU VPS), about 1.5, 1 and 5
  minutes of wall clock.
- Rectifier units (118: rect and rect_homog): Modal app `crw-hebb-big`, CPU fan-out (2 cores,
  12 GiB per container, torch float64 on CPU; the A100 path exists but the workspace's GPU
  slots were occupied by another project's jobs). Pilot run `20261119-102033` (two units, R4),
  production run `20261119-102221` (116 units, 0 failed), about 5 minutes wall clock, 66
  container-minutes of unit time (longest unit 236 s; peak RSS 5.8 GB). Estimated cost under
  $0.50.
- Pilot (R4): blocks N = 400 rectifier at b = 0.018 (near percolation): 30 steps, 23 s,
  residual 4.1e-10, 5.2 GB, film 140 kB; at b = 0.1 (fused): 73 steps, 42 s, residual 7.2e-10,
  5.8 GB. Covariance film (B = 3, theta = 0.4, N = 300): 400 steps, 2.3 s, 73 MB, 664 kB.
