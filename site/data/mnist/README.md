# `site/data/mnist/` — chapter 3 data (Hebbian resistor network on kernel-lifted MNIST)

Produced by `hebb/mnist/export.py` from Modal runs of `hebb/mnist/train.py` (method,
decisions and numbers: `docs/mnist.md`). Everything is static. Binary files are raw
little-endian arrays with no header: load them with `new Float32Array(buf)` or
`new Uint8Array(buf)` and reshape using the `shape` given next to the file in `index.json`.
Every file path in `index.json` is relative to this directory.

## `index.json`

```
{
  version, description, headline: <run name>,
  filmstrip: {png, steps[6], run},              // headline run, 6 frames of H Sigma H^T, fixed colour scale
  lifts: { <liftKey>: LiftInfo },               // liftKey: "pixels" | "rff64" | "rff256" | "rff1024" | "rff256bw0.5" | "rff256bw2"
  runs: [ RunInfo, ... ],                       // every run (featured or not)
  samples: [ SampleSheet, ... ]                 // export item 6
}
```

### LiftInfo
- `kind`, `B`, `scale` (lifted = scale * raw features; chosen so the mean feature variance is 1, i.e. tr Sigma_g = B),
  `sigma` (RFF Gaussian-kernel bandwidth, 14x14 pixel units), `bw_mult` (multiplier on the median heuristic).
- `sigma_stats`: off-diagonal quantiles of Sigma_g (`q`, `value`), `frac_negative`, `mean`, `diag_mean`, `diag_max`,
  `max_offdiag`, `diag` (quantiles 0, .25, .5, .75, 1 of the diagonal). Use these to put theta in context.
- `offdiag_hist`: `{counts[60], edges[61]}`, a histogram of the off-diagonal entries of Sigma_g.
- `files.sigma`: Sigma_g, float32 `[B, B]`, natural boundary order (only if B <= 256).
- `files.sigma_contrast_spectrum`: eigenvalues of P Sigma_g P (P = I - 11^T/B), float32 `[B]`, descending.
- `files.eigen_images` (pixels only): PNG sprite of the top 16 eigenvectors of P Sigma P as 14x14 images (red +, blue -).
- `files.eigen_sta`: the same 16 eigen-directions, rendered as response-weighted average stimuli (see below).
- `files.eigen_top_digits`: 16 rows x 8 test digits (28x28) with the largest response to each eigen-direction.
- `eigen_tuning`: `[16][10]`, the mean z-scored response of each eigen-direction per digit class.

Sprite `layout`: `{tile_w, tile_h, pad, ncol, nrow, n, cmap}`. Tile i is at
x = pad + (i % ncol)·(tile_w + pad), y = pad + floor(i / ncol)·(tile_h + pad).

### RunInfo (every run)
- Configuration: `name, lift, B, N, theta, wmin, gamma, init ("generic" | "heavy"), ib_shape, ii_scale, mode ("sgd" | "mf"),
  seed, steps, dt, batch, featured`. `theta_percentile` is the fraction of off-diagonal Sigma_g entries below theta.
- `final`: the cluster/structure summary computed on the final weights (time-averaged over the last 20% of steps for
  sgd). Fields:
  - `n_clusters`: the headline cluster count. Clusters are the single-linkage components of "L1(h_i, h_j) < 0.1".
  - `n_clusters_tol05` and `n_clusters_tol20`: the same count at tolerance 0.05 and 0.2.
  - `n_leader`: `hebb.core.clusters` leader clustering, for comparison.
  - `n_major`: clusters with at least 1% of N neurons.
  - `n_dead`: neurons with every synapse at the floor.
  - `n_live_clusters` and `n_live_major`: the counts excluding the dead pool.
  - `n_active_components`: connected components of the active I-I graph.
  - `within_l1`: the mean L1 distance from a neuron to its cluster centroid.
  - `eff_rank` of H.
  - `floor_ii` and `floor_ib`: the fraction of synapses at the floor.
  - `res_max`, `res_mean`, `res_active_mean`: the projected mean-field stationarity residual |dw/dt| (SPEC §1). `res_rel` is res_max divided by the max weight.
  - `averaged_over_steps`.
- `cluster_sizes[K]`, `cluster_energy[K]` (h_k^T Sigma_g h_k of the cluster centroid part), `dead_clusters` (ids).
- `readout`: `{ls_both, ls_ii, ls_ib, naive_ii, naive_ib, init_ls_both, instant_ls_both}`, each a metric dict:
  - parameters known: `known_relfro` (||C - Sigma||_F / ||Sigma||_F), `known_bw` (BW^2 / tr Sigma, C not rescaled);
  - unknown scale: `scale_cos` (Frobenius cosine), `scale_bw` (BW^2/tr Sigma with C optimally rescaled = 1 - F^2/(trC trSigma));
  - unknown theta ("contrast-subspace score", P C P vs P Sigma P): `contrast_cos`, `contrast_relfro_scaled`, `contrast_bw`,
    `overlap_k5/k10/k20/kclust` (mean cos^2 of principal angles between top-k eigenspaces, 1 = identical);
  - `neg_mass_full`, `neg_mass_contrast`: negative spectral mass clipped before BW.
  `init_ls_both` = readout applied to the untrained network, and `instant_ls_both` = readout on the final instantaneous (not averaged) weights.
  Both are controls in which the equilibrium assumption is violated (for init) or noisy (for instant).
- `baselines`: `{identity, diag, rank_k_truncation}` (k = n_clusters), with the same metric dict.
- `readout_info`: number of active equations per family (`n_ii`, `n_ib`), CG iterations, ridge alpha, and the data misfits.
- `measurement_rank`: rank and condition number of the cluster-level measurement operator on Sym(B), per edge family
  (`ii`, `ib`, `both`). `dim_sym` = B(B+1)/2 is the number of unknowns.
- `support`: readout quality restricted to the boundary nodes covered by live cluster parts (`n_supported`,
  `cos_on_support`, `relerr_on_support_known`, `part_energy_relerr_median`, `sigma_contrast_mass_on_support_block`).
- `timing`: seconds on the GPU.

### RunInfo, featured runs only: `files`, `n_shown`, `snapshot_steps[S]`
S = 36 log-spaced snapshots (step 0 = init). n = `n_shown` <= 256 displayed neurons, in ONE fixed order (clusters by size,
dead pool last; within a cluster by distance to its centroid).
- `curves` -> `runs/<name>/curves.json`: per-snapshot arrays `step, time, n_clusters, n_live_clusters, n_dead, ...,
  eff_rank, floor_ii, floor_ib, res_*`, plus `ro_<family>_<metric>` for the readout at every snapshot. These are
  instantaneous-weight readouts, so the equilibrium assumption is violated before convergence.
  Also: `dense` (`cols`, `rows`; every ~10 steps), `hist_edges[50]`, `hist_ii[S][49]`, `hist_ib[S][49]` (weight histograms,
  bin 0 = [0, 1e-5), then log-spaced up to 10), `cluster_sizes_snap[S][*]`, `cluster_energy_snap[S][*]`.
- `similarity` -> `similarity.u8`, uint8 `[S, n, n]`: H Sigma_g H^T of the shown neurons, decoded as value = -a + 2a·byte/255
  (`a` given). The range is symmetric so that 0 maps to byte 127.5.
- `neurons` -> `neurons.json`:
  - `neurons`: `{ids, cluster, cluster_rank, dead, preferred_digit}`, each in display order.
  - `clusters`: `[{id, rank, size, dead, energy, support, entropy_eff, tuning[10], preferred_digit}]`, in size order with dead clusters last.
  - `snapshot_steps`.
  The sample is stratified: each cluster has at least min(size, 4) members and the rest are proportional to size. Use the cluster `size` field, not counts in the sample.
- `cluster_part_arrays` -> `cluster_parts.f32`, float32 `[K, B]`: cluster centroid parts in `clusters` order.
- `cluster_parts` (pixels only, PNG): parts as 14x14 images, each tile on its own max (`clusters` lists ids).
- `cluster_sta` (PNG): the response-weighted average stimulus E[r(x)(x - xbar)]/sd(r), with r(x) = c_k^T (g(x) - mu_g) over the
  test set. Pixel units, red = brighter than average, each tile on its own symmetric scale.
- `cluster_top_digits` (PNG): one row per cluster (same order as `cluster_sta`) of the 8 test digits with the largest response.
- `neuron_tiles` (pixels only, PNG): 16 rows (the shown-neuron positions in `neurons`) x S columns (snapshots); h_i as 14x14.
- `umap` -> `umap.f32`, float32 `[S, n, 2]`: Hellinger-metric UMAP of the shown neurons' embeddings. It is computed at the last
  snapshot, then backwards, with each frame initialised at the next frame and Procrustes-aligned to it.
- `graph` -> `graph_edges.f32`, float32 `[sum n_edges, 3]` = (src, dst, weight).
  - The edge list is concatenated over `snapshots`; each snapshot has `{step, umap_snapshot, n_edges, tutte_xy?}`.
  - Node ids: 0..n-1 are shown neurons (in `neurons.ids` order); n..n+B-1 are boundary pixels, located at `boundary_xy`
    (x = column, y = row) for the pixel lift only.
  - Only active edges are included (w > 2 w_min), at most 3000 per snapshot, strongest first.
  - Layout: use UMAP frame `umap_snapshot`, or for pixels `tutte_xy`. `tutte_xy` is the harmonic (Tutte) embedding H @ pixel_xy plus N(0, 0.12 px) jitter.
- `readout` -> `readout.json`:
  - `order`: a permutation of the boundary nodes. Supported nodes come first, grouped by owning cluster; the rest follow, sorted by the leading eigenvector. Apply it to rows and columns of C_hat AND Sigma_g.
  - `n_supported`.
  - `spectra`: `{ls_both, ls_ii, ls_ib, naive_ii, naive_ib, init_ls_both, sigma_contrast}`: eigenvalues of P C P, descending. Compare shapes after scaling.
- `chat_<variant>` -> `chat_<variant>.f32`, float32 `[B, B]` (B <= 256), natural order, absolute units (parameters known).

### SampleSheet (export item 6: moment-based sample synthesis)
- `{run, png, layout, rows[], generator, mmd}`. The PNG has one row per entry of `rows`, with 16 samples each, shown as 14x14, clipped to [0,1].
- Row labels: "real data", ("real data, pre-image round trip" for RFF), "true-moment generator",
  "weight-reconstructed generator", "baseline generator: diag", "baseline generator: untrained net".
- Each row has metrics: `mmd2_to_heldout`, moment-consistency errors, and `psd_negative_mass_clipped`. RFF rows also have `preimage_rel_residual_median`.
- `generator` states every assumption: the mean source, scale, completion, PSD repair and pre-image settings.
- `mmd.reference_real_train_vs_heldout` is the MMD^2 of real training digits against the held-out digits, i.e. the noise floor.
