# site/data/big: precomputed big runs (N >> B)

Synthetic data only. All files here are produced by the scripts in `hebb/big/` (2026-11-19).
Total size: about 4.4 MB (budget 15 MB). The numbers and what they show are in
`docs/findings_v2.md`, section "Big runs (N ≫ B), 2026-11-19".

| file | what | script |
|---|---|---|
| `index.json` | list of films `{id, title, family, rule, nonlin, B, N, file, bytes}` (`bytes` = .json + .bin) | written by every exporter |
| `<id>.json` + `<id>.bin` | one film (6 films, list below) | `hebb/big/cov_runs.py` |
| `cov_all.json` | covariance rule: every run (films and non-film runs, three seeds for B = 3, floor sensitivity), with the per-cluster Perron report | `hebb/big/cov_runs.py` |
| `readout_vs_N.json` | family C: covariance readout vs N at B = 12 | `hebb/big/readout_vs_n.py` |

Regenerate: `uv run python -m hebb.big.cov_runs all`;
`uv run python -m hebb.big.readout_vs_n`. Raw outputs (full weight matrices, responses on both
banks) go to `scratch/out/big/` (gitignored), not to the site.

## Film format

`<id>.json` (metadata and small arrays) and `<id>.bin` (little-endian typed arrays at byte
offsets given in `json.arrays`; every offset is a multiple of 8).

```
{ "id", "title", "family": "A"|"B", "rule": "cov", "nonlin": "linear",
  "B", "N", "seed", "config": {...all knobs, including Sigma...}, "script",
  "space": "simplex" | "pca",
  "space_label": "input space (harmonic embedding)" | "response-function space (PCA)",
  "pca": null | {"explained_var": [3 floats], "total_var": float, "features": "..."},
  "frames": [{"t", "step", "residual", "n_live", "n_components"}],
  "bin": "<id>.bin",
  "arrays": {
    "coords": {"offset", "dtype": "uint16", "shape": [F, N, 3], "lo": [3], "hi": [3]},   x = lo + (hi - lo) v / 65535
    "H":      {"offset", "dtype": "uint8",  "shape": [F, N, B]},                          h = v / 255
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
  `residual` is the projected stationarity residual max|dw/dt| of that frame's state (`core.stationarity`). `n_live` = number of live synapses
  (lateral w above the floor, `w > w_min (1 + 1e-3)`),
  `n_components` = connected components of that live graph.
- **coords.** `space = "simplex"`: coords are the rows of H (barycentric, sum 1; B = 3 linear
  runs). `space = "pca"`: function-space PCA with axes fixed by the final frame. The features
  are chosen so that Euclidean distance = RMS response distance D_ij, so the 3-D coordinates
  are classical MDS of the measured D at the final frame (`x_i = Sigma^{1/2} h_i`). `pca.explained_var` /
  `total_var` says how much of the geometry the three axes hold (0.62-1.0 here). Every frame
  uses the same axes and mean, so motion is real motion.
- **Same state.** In every frame the coordinates, H, edges and residual come from one state:
  the weights W_n and their own equilibrium (R4).
- **edges.** The strongest live synapses, at most 1500 per frame, strongest first; only the
  first `count[f]` rows of frame f are valid (the rest are zero padding). Full weights are not on
  the site (`scratch/out/big/`).
- **in_w** (addition to the plan's contract; the player may ignore it): input-synapse
  conductances: the plastic W_IB per frame (`F_in = F`).
- **final.labels.** Clusters at the final frame, labelled 0, 1, ... by decreasing size;
  `-1` = dead neuron (every input synapse at the floor). Single linkage on H
  in L1, cut at the largest gap (`core.clusters_auto`), dead neurons excluded.
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
- **final.metrics**: run metrics (convergence: `converged`, `residual`, `steps`, `t`; K,
  dead count, enumerated and populated stable parts).
- **network_layout_seed**: seed for the neuron-space (random) layout in the player.

## Films

All converged (projected residual < 1e-6; actual ≤ 1e-7).

| id | rule | model | frames | space |
|---|---|---|---|---|
| `A_cov_ind_th0.4` | cov | B = 3 independent unit inputs, theta = 0.4, N = 300 | 59 | simplex |
| `A_cov_ind_th0.2` | cov | same, theta = 0.2 (bound population, one corner) | 60 | simplex |
| `A_cov_pair_th0.5` | cov | rho12 = 0.6, theta = 0.5 | 61 | simplex |
| `B_cov_blocks12_th0.35` | cov | 12 inputs, 3 blocks of 4 (within 0.5, across 0.1), theta = 0.35, N = 400 | 34 | pca |
| `B_cov_hier16_th0.35` | cov | 16 inputs, binary hierarchy (0.8 / 0.5 / 0.25 / 0.05), theta = 0.35 | 34 | pca |
| `B_cov_ring16_th0.5` | cov | ring of 16, exp(-d^2 / (2 * 1.5^2)), theta = 0.5 | 35 | pca |

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
error; all_converged.
`runs[]` also has relative Frobenius errors for ridge, hinge and baseline recoveries vs Chat and Sigma; `summary[]` has `relerr_Chat` (mean, sd), `relerr_Chat_min` and `relerr_Chat_max`.
What is claimed (R5): exact identifiability of the training-bank contrast
second moment P C_hat P, for stationary noiseless weights, known attachments and rule
parameters, and a full-rank, well-conditioned live-edge operator.

## Compute record

- Covariance rule and family C readout: local (4 vCPU VPS), about 1.5 and 5 minutes of wall clock, respectively.
- Covariance film (B = 3, theta = 0.4, N = 300): 400 steps, 2.3 s, 73 MB, 664 kB.
