# `site/data/nonlin/`: the "linear circuit, rectified output" extension

Produced by `hebb/nonlin/export.py` from Modal runs of `hebb/nonlin/train.py`. The theory,
method, decisions and numbers are in `docs/nonlin.md`. Everything is static and follows the
`site/data/mnist` conventions. Binary files are raw little-endian arrays with no header:
load them with `new Float32Array(buf)` / `new Uint8Array(buf)` and reshape with the `shape`
stored next to the file in `index.json`. Paths are relative to this directory.

Model in one line: u = H g (Kirchhoff, unchanged); the rule correlates rates
r_i = c·relu(u_i − E u_i − τ sd u_i) (interior) and s_b = c·relu(g_b − μ_b − τ σ_b)
(boundary) instead of voltages. g is the raw MNIST pixels (14×14, scaled to mean pixel
variance 1). `rule = "linear"` is the base model on the same inputs. `input = "gauss"`
replaces the pixels with Gaussian samples of the same mean and covariance.

## Files

| file | contents |
|---|---|
| `index.json` | everything small: theta tables, every run's configuration and summaries, featured-run file lists, the sample sheet. See below. |
| `dual.json` | the dual activation and the empirical scatter (export item: "dual-activation curve with the empirical scatter"). See below. |
| `synth_ring.json` | synthetic Gaussian check of the theory (`hebb/nonlin/synth.py`): ring covariance B = 24, N = 96, mean-field, linear vs rectified at the same pairwise correlation threshold rho*. `runs[]`: `{rule, tau, rho_star, theta, c, seed, residual, K, K_major, sizes, width_mean, energy, parts (K x 24), readout{known_relfro, contrast_cos}, readout_fit_theta{theta_true, theta_fit, ...}}`. Also `rho_grid` and `ktilde[tau]`. |
| `sigma14.f32` | float32 `[196, 196]`: Sigma_g of the scaled 14×14 pixels (natural pixel order, b = row·14 + col). |
| `mu14.f32` | float32 `[196]`: mean of the scaled pixels. |
| `samples.png` | export item D, the sample sheet. Rows are described in `index.samples`. |
| `runs/<name>/parts_final.png` | every run: final live cluster parts (centroid h of the members) as 14×14 (or 28×28) images, largest first, at most 20, each on its own max (`magma_r`). Layout is in `index.runs[i].parts_final`. |
| `runs/<name>/…` (featured runs only) | see "Featured runs". |

Sprite layout (same as `site/data/mnist`): `{tile_w, tile_h, pad, ncol, nrow, n, cmap}`. Tile i
is at x = pad + (i % ncol)·(tile_w + pad), y = pad + floor(i / ncol)·(tile_h + pad).

## `index.json`

```
{ version, description,
  theta_table14, theta_table28,     // per rule: theta at quantiles q of the boundary covariance the rule sees, the gain c,
                                    // and for the rectified rule the Gaussian surrogate's own-quantile theta (gauss_own_theta)
  featured: {<key>: <run name>},    // see below
  runs: [RunInfo],                  // 89 runs
  files: {dual, sigma, mu, synth_ring},
  samples: SampleSheet }
```

### RunInfo (every run)
- Configuration: `name, exp, q, rule ("rect" | "linear"), tau (null for linear), input ("pixels" | "gauss"),
  res (14 | 28), B, N, theta, c (rate gain; 1 for linear), seed, wmin, gamma, steps, dt, batch`.
- `exp` labels the experiment:
  - `A_lin`, `A_rect`: real pixels.
  - `B_lin`, `B_rect`: Gaussian surrogate with the same theta and c as A.
  - `Bown_rect`: Gaussian surrogate with theta at the surrogate's own quantile q.
  - `G0_*`: the same with gamma = 0 (no lateral plasticity).
  - `A28_*`: 28×28 pixels, N = 3920.
- `q`: the quantile of the off-diagonal boundary covariance at which theta sits. The q grid is the sibling's linear
  theta {0.3, 0.45, 0.6, 0.75, 0.9} expressed as quantiles of Sigma_g: 0.946, 0.967, 0.980, 0.987, 0.992.
- `final`: structure of the time-averaged final weights.
  - `n_clusters`: the headline count, single-linkage components of L1(h_i, h_j) < 0.1, as in `site/data/mnist`.
  - Also `n_clusters_tol05`, `n_clusters_tol20`, `n_major`, `n_singletons`, `n_active_components`, `within_l1`, `eff_rank`,
    `floor_ii`, `floor_ib`.
  - Residuals: `res_max`, `res_mean`, `res_active_mean`, `res_rel` (= res_max / max weight), `res_max_strict`, `w_max`.
  - Computed here: `n_dead` (every synapse ≤ 2 wmin), `n_live_clusters`, `n_live_major` (≥ max(2, 1% N)),
    `n_singletons_live`, `within_l1_live` (mean L1 to the centroid among live neurons in live clusters of size ≥ 2),
    `live_neurons`.
  - Residual definition: the projected stationarity residual of the EXACT expected drift. The expectation is over the
    full 60k training set for rectified/pixels, closed form for rectified/gauss, and H Sigma H^T for linear. Edges with
    w ≤ 2 wmin and negative drift count 0. `res_max_strict` uses the floor tolerance wmin(1+1e-6).
- `allocation`: major live clusters only.
  - `sizes`, `energy_volt` (c^T Sigma c of the part), `energy_rate` (unit-gain rate variance of the part's own rate
    over the training digits), `part_width` (participation ratio 1/sum h^2, in pixels).
  - `beta_volt`, `beta_rate`: log-log slope of size vs energy, only when there are ≥ 3 major clusters (rarely
    meaningful here; see docs/nonlin.md §4.4).
- `rule_cov_quantiles`: `{q, ii, ib}`, quantiles of the expected covariance the rule actually correlates at the end:
  rates (×c²) for rect, voltages for linear. `ii` is over interior pairs and `ib` over interior–pixel pairs. Use it
  to read theta.
- `gap`: the higher-order gap. The empirical rate covariance over the 60k training digits (E) is compared with the
  Gaussian dual-activation prediction from the exact voltage covariance (P), at the rule's tau (tau = 0 post hoc for
  linear-rule nets).
  - Rates are unit gain. For networks trained on the surrogate the gap is still measured on REAL digits.
  - Per family (`ii`, `ib`, `bb` = pixel pairs):
    - `rel_fro_gap` = ||E − P||/||E||; `corr_emp_pred`; `mean_emp`, `mean_pred`; `emp_quantiles`.
    - `rate_corr_rel_gap` and `rate_corr_mean_abs_gap` compare the empirical rate correlation with k~_tau(rho).
      This removes variance mismatch.
    - `active_*` restricts to active edges; `active_mean_ratio_emp_over_pred` is the median E/P there.
  - `var_ratio_interior_q10_50_90` and `var_ratio_pixel_q10_50_90`: Var_emp(r)/(k_tau(1) Var(u)).
  - `gap_init`: the same on the untrained network.
- `readout`, one metric dict per variant (the metrics follow `site/data/mnist`: `known_relfro, known_bw, scale_cos,
  scale_bw, contrast_cos, contrast_bw, overlap_k5, overlap_k10, overlap_kclust, neg_mass_full, neg_mass_contrast`,
  plus `info` with the misfit).
  - rect variants:
    - `gauss_both` / `gauss_ii` / `gauss_ib`: the Gaussian inversion of the weights, by edge family.
    - `chord_both`: LS readout of Q/k_tau(1), i.e. the nonlinearity ignored.
    - `gauss_expected_cov`: the same active edges, with targets = the exact expected rate covariance (removes SGD
      noise and non-equilibrium; what remains is the Gaussian-closure bias).
    - `gauss_fit_theta`: theta fitted; `info.theta_fit`.
    - `init_gauss_both`: the untrained control.
  - linear variants: `ls_both` / `ls_ii` / `ls_ib` and `init_ls_both`.
  - gamma = 0 runs have no `*_ii` variants (lateral edges carry no information).
- `readout_baselines`: `{identity, diag, rank_k_truncation}`.
- `probes`: test accuracy of a multinomial logistic probe (z-scored features, L-BFGS, wd 1e-4, 60k train / 10k test).
  - Feature sets: `pixels` (196);
    `trained_volt_all|sub` and `trained_rate_t{0,0.5,1}_all|sub` (all N neurons or a random subset of B = 196; rates
    at that tau, post hoc for the linear rule); `init_*` (the same for the untrained network);
    `randrelu_t{0,0.5,1}_all|sub` (Gaussian random projections of the centred pixels, rectified at tau sd).
  - `probes_frac_active`: the fraction of (digit, neuron) pairs with a nonzero rate.
- `parts` (14×14 runs): up to 8 major live clusters, largest first.
  - Shape: `size, width, volt_var`; `skew` and `excess_kurtosis` of the part's voltage c·g over the training digits.
  - Rates: `rate_var`, `rate_var_over_gauss` (= Var_emp(rate)/(k_tau(1) volt_var)), `frac_active`.
  - Coding: `tuning[10]` (mean z-scored test response per digit), `pref_digit`, `probe_1feature` (logistic probe on
    this single feature).
  - `cos_to_linear_part`: the cosine to the largest live part of the linear-rule run with the same gamma, seed and q.
  - `energy_ratio_<variant>` = c^T C_hat c / c^T Sigma c for the stored readout matrices (parameters known; 1 =
    unbiased).
- `parts_final`: `{png, layout, sizes, note}`.
- `timing`: seconds (GPU type recorded).

### Featured runs (`index.featured`; 14×14, gamma = 1, seed 0)
| key | experiment |
|---|---|
| `A_lin` | linear rule, pixels, theta = 0.6 (q = 0.980, the sibling's theta) |
| `A_rect_t0` | rectified tau = 0, pixels, same q |
| `A_rect_t1` | rectified tau = 1, pixels, same q |
| `Bown_rect_t0` | rectified tau = 0, Gaussian surrogate, theta at the surrogate's own q = 0.980 |
| `A_rect_t0_q0.946` | rectified tau = 0, pixels, q = 0.946 |
| `B_rect_t0_q0.946` | rectified tau = 0, Gaussian surrogate, SAME theta and c as `A_rect_t0_q0.946` (at q = 0.967 the surrogate net loses 931 of 980 neurons; at q ≥ 0.98 it dies) |

Extra RunInfo fields: `snapshot_steps[10]`, `n_shown` (256), `shown_ids` (neuron ids in display order: clusters
by size with dead clusters last, within a cluster by L1 distance to the centroid), `shown_cluster_rank`,
`shown_dead`, and `files`:
- `similarity_volt` → `similarity_volt.u8`, uint8 `[10, 256, 256]`: H Sigma_g H^T for the shown neurons at each
  snapshot, one fixed order. Decode as value = −a + 2a·byte/255, with `a` given.
- `similarity_rate` → `similarity_rate.u8`: the rate covariance c² Cov(r_i, r_j) over the 60k training digits, same
  encoding. For the linear rule this is post hoc (tau = 0, c of tau = 0).
- `parts_over_time` (PNG): rows are the final live clusters with members among the shown neurons, columns are the
  snapshots; each tile is the mean h of those members at that snapshot.
- `neurons_over_time` (PNG): 16 shown neurons × 10 snapshots (`positions` index into the display order, `neuron_ids`).
- `umap` → `umap.f32`, float32 `[N, 2]`: a Hellinger UMAP of ALL final embeddings in neuron-id order (same metric
  and settings as `site/data/mnist`). It is degenerate when one cluster holds every neuron.
  - `umap_cluster_rank.u8` `[N]`: the cluster size rank (0 = largest live, 250 = rank ≥ 250, 255 = dead).
  - `umap_pref_digit.u8` `[N]`: the preferred digit (argmax of the mean z-scored test response: rates for rect,
    voltages for linear).
- `chat_<variant>.f32`, float32 `[196, 196]`: C_hat in absolute units (parameters known).
  - rect: `gauss_both`, `chord_both`, `gauss_expected_cov`; linear: `ls_both`.
  - Compare with `sigma14.f32`.
- `curves` → `curves.json`: per-snapshot arrays (24 snapshots).
  - `step, time, n_clusters, n_major, n_singletons, n_active_components, within_l1, eff_rank, floor_ii, floor_ib`.
  - Residuals: `res_max, res_mean, res_active_mean, res_rel, res_max_strict` (instantaneous weights, so SGD noise is
    included).
  - `cov_q10/50/90/99`: quantiles of the rule's expected interior covariance (rates ×c² for rect).
  - `dense_cols` and `dense`: every 10 steps, [step, mean |dW_II|/dt, mean |dW_IB|/dt, mean W_II, mean W_IB].

### SampleSheet (`index.samples`, export item D)
`{png, layout, rows[{label, psd_negative_mass_clipped}], generator}`. The rows, 16 samples each (14×14, gray,
clipped to [0, 1]):
1. real test digits;
2. N(mu, Sigma_g) with the true moments;
3. the linear rule's LS readout;
4. the rectified tau = 0 Gaussian inversion of the weights;
5. the rectified chord readout;
6. the rectified inversion of the exact rate covariance;
7. the Gaussian-surrogate-trained rectified net (own-quantile theta), Gaussian inversion.

All rows share the same z. mu is the training-data mean: this is an ASSUMPTION, since the weights do not determine
the mean. C_hat has negative eigenvalues clipped and is in absolute units.

## `dual.json`
- `rho[201]` on [−1, 1].
- `ktilde[tau]` = k_tau(rho)/k_tau(1), the rate correlation predicted from the voltage correlation, for
  tau ∈ {0, 0.5, 1, 2}.
- `kraw[tau]` = k_tau(rho) (unit-variance rate covariance) and `dk[tau]` = k_tau'(rho) = P(both active).
- `scatter[<featured key>]`: `{run, tau, ii, ib, bb}`, each `{rho[], rate_corr[]}` (≤ 1500 random pairs). These are
  the empirical voltage correlation and the empirical rate correlation over the 60k training digits, to plot over
  ktilde[tau].
  - `ii` = neuron pairs (degenerate at rho = 1 when one cluster holds every neuron), `ib` = neuron–pixel pairs,
    `bb` = pixel pairs.
