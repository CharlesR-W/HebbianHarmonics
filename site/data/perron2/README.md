# site/data/perron2: Perron parts, second pass (2026-11-19)

Synthetic data only. The model is the v1 covariance rule (`hebb/core.py`, mean-field, dense masks,
lambda = eta = 1, dt = 0.05, floor w_min = omega / N with omega = 0.009, as `hebb/big/cov_runs.py`).
Scripts are in `hebb/perron2/`; raw outputs (full weights per theta, unit JSONs) in `scratch/out/perron2/`
(gitignored). Total size here: about 3.7 MB.

Regenerate:

```
uv run python -m hebb.perron2.parts all            # C1: selftest, diagrams, edge cases, random trees (~20 min on 4 cores)
uv run python -m hebb.perron2.ramps run            # C2: up + down ramps, P2_GAMMAS=1,0.5,0.25,0.1
uv run python -m hebb.perron2.ramps down2          # C2: down ramps from a cold start at 0.822 (P2_GAMMAS=1,0.5,0.1)
uv run python -m hebb.perron2.floor_test 1 && uv run python -m hebb.perron2.floor_test 2   # C2 split tests
uv run python -m hebb.perron2.cold units           # C3 units -> Modal (hebb/perron2/modal_app.py), see its docstring
uv run python -m hebb.perron2.cold extra           # C3 extra: block345, local
uv run python -m hebb.perron2.export c1|ramps      # site JSON
uv run python -m hebb.perron2.export cold <unit dirs...>
uv run python -m hebb.perron2.ramps export 1.0:up 1.0:down2 0.1:up 0.1:down2   # films
```

Definitions used everywhere. A = Sigma - theta 11^T. A part is a support S with a positive eigenvector
h_S of A_SS (normalised to sum 1), eigenvalue c > 0, and (A h)_b <= 0 for b not in S;
margin = theta - max_{b not in S} Sigma_bS h_S; gap = c - (next eigenvalue of A_SS).
Stable: c is the top eigenvalue of A_SS with gap > 1e-9 and margin > 1e-9. Enumeration is exact over
all 2^B - 1 supports (`parts.enumerate_fast`, vectorised; its stable parts agree exactly with
`hebb/big/common.enumerate_parts`, checked at 20 (Sigma, theta) pairs). G_theta is the threshold graph
{a != b : Sigma_ab > theta}. "Populated": a cluster with >= 1% of the neurons whose support (inputs whose
input synapse is above the floor in most members) equals a stable part and whose centroid has cosine
> 0.98 with it. Coverage = populated stable parts / stable parts.

## Files

| file | what |
|---|---|
| `part_diagrams.json` | C1: stable parts vs theta (200 values) for 11 covariances |
| `c1_summary.json` | C1: per covariance (16, including noise seeds 1-2) agreement with cliques / components of G_theta, edge cases (theta exactly at / 1e-7 around each level and variance), random-tree sweep |
| `ramps.json` | C2: quasi-static ramps on hier16, gammas 1, 0.5, 0.25, 0.1; split tests; hysteresis |
| `cold_starts.json` | C3 table (hier16, block12, block345) and C4 (ramp vs cold start at equal theta) |
| `index.json`, `ramp_hier16_*.json/.bin` | four films (format below) |

### part_diagrams.json

`meta` (grid, field order), `diagrams[]`: `name`, `kind` (`ultra`: unit-variance ultrametric;
`ultra_var`: ultrametric matrix with unequal diagonal; `drd`: D R D with R ultrametric;
`nonultra`; `nonultra_ultra` = block12, which is in fact ultrametric), `desc`, `B`, `Sigma` (4 decimals),
`kappa0` (mean entry), `theta[200]` (theta_k = (k + 1/2)/200 * max variance), and per theta:
`parts[k]` = list of stable parts `[support, c, margin, gap, h on the support]`, sorted by decreasing c;
`n_parts_all` (all parts, stable or not; parts hidden in degenerate eigenspaces can be missed; never
stable); `n_covered` (inputs in some stable part); `cliques[k]` / `components[k]` (maximal cliques /
connected components of G_theta whose A_SS has a positive top eigenvalue; `null` where they equal the
stable supports); `eq_cliques[k]`, `eq_components[k]` (1 if the stable supports are exactly those sets).
Covariances: hier16 (binary, 0.8/0.5/0.25/0.05), hier12_322 (branching 3,2,2; 0.7/0.4/0.1), irreg14
(irregular tree, see `desc`), randultra16 (random tree), hier16_ultravar (hier16 off the diagonal,
variances 0.85 / 1.4), hier16_drd (variances U(0.5, 1.5)), hier16 + uniform symmetric noise +-0.01 /
0.03 / 0.1 (seed 100; seeds 101-102 in c1_summary only), block12, ring16.

### c1_summary.json

`families[]`: `eq_cliques` / `eq_components` (number of the 200 thetas where the stable set equals the
maximal cliques / components), `stable_not_clique_total`, `clique_not_stable_total` (summed over theta),
`theta_with_uncovered_inputs`, `n_mismatch`, `mismatches_first` (first 3). `edge_cases[]` (ultrametric
families): `level`, `where` (below / at / above, +-1e-7), stable supports, cliques, `eq_cliques`, and up to
two marginal parts (margin or gap ~ 0) at that theta. `random_ultrametric`: 24 random trees (B = 8-14),
120 thetas each, unit variance (`unit`) and an ultrametric matrix with unequal diagonal (`ultra_var`):
`n_fail` = thetas where stable set != cliques or != components.

### ramps.json

`meta` (protocol). Protocol: grid = 80 values linspace(0.03, 0.95); N = 300, B = 16 (hier16).
Up: cold start (core.init_state, seed 0) at 0.03, then at each grid theta a relative kick
(all above-floor weights x (1 + 1e-3 xi), xi ~ N(0, 1)) and integration from the previous equilibrium to
projected residual < 1e-7 (checked every 50 steps, cap 40000 steps = t 2000). Down: continues from the
end of up (0.95 -> 0.03). Down2: cold start at 0.822 (all 16 singleton parts populated), then down to 0.03.
The kick is needed: without it the mean-field flow keeps exactly symmetric clone clusters on saddles (a
part that just lost stability) with residual < 1e-7 for several grid steps (log in
`scratch/out/perron2/log_ramps_nokick.txt`).

`runs[]` per gamma: `up`, `down`, `down2` (gammas 1, 0.5, 0.1), each with per-theta arrays `theta`,
`steps`, `residual`, `converged` (< 1e-6), `K_major`, `n_dead`, `sizes`, `supports`, `perron_cos` (centroid
vs Perron vector of A_SS on the cluster's support), `matches` (cluster populates a stable part), `populated`,
`populated_sizes`, `n_stable`, `n_populated`, `coverage`, `frac_on_stable`, `lateral_between_max` (largest
mean lateral weight between two major clusters; the floor is 3e-5), `lateral_between` (K x K mean lateral
weights, diagonal = within), `n_live_lateral`, `n_lateral_components`; `transitions` (consecutive thetas
where the cluster supports change, with the levels crossed); `n_not_converged`. `hysteresis[]`: per theta,
populated sets and supports on the up and down branches, `same`, coverages.
`split_tests`: restart from the up-ramp equilibrium at `theta_from`, floor set to `omega / N`, kick, integrate
at `theta_to` (cap 200000 steps); `sizes_before` -> `sizes`.

### cold_starts.json

`table[]` per (sigma, gamma, theta): 5 seeds (block345: 8), N = 300, cold start as cov_runs, integrate to
residual < 1e-7 (cap 120000 steps), then a saddle test (kick 1e-3 and re-integrate, up to 3 times):
`n_stable`, `n_populated`, `K_major`, `coverage_mean`, `frac_on_stable`, `bound_one_cluster`,
`n_distinct_populated_sets`, `populated_sets`, `populated_c_ranks` (0 = largest c; with equal c the rank is
arbitrary), `supports`, `sizes`, `converged`, `residual_max`, `changed_by_kick`, `pre_kick_n_populated`,
`gamma_kappa0`. Thetas: 30 values = ramp grid indices round(linspace(0, 79, 30)). Gammas: 1 and 0.5;
hier16 also 0.25 and 0.1 (for C4). block345 (blocks of 3, 4, 5; within 0.5, across 0.1): theta 0.15 /
0.2 / 0.25 at gamma 1 and theta 0.1 (exactly the across level: no stable part) at gamma 0.5.
`c4_ramp_vs_cold[]`: per gamma and theta, up-ramp populated count, sizes and supports vs the cold starts,
`ramp_set_in_cold_sets`.

## Films

Format of `site/data/big/README.md` ("Film format"), with these differences: one frame per grid theta
(`frames[f].theta`; `t` / `step` are cumulative model time / Euler steps over the ramp; also
`steps_this_theta`, `converged`, `K_major`, `sizes`, `supports`, `populated`, `n_stable`, `coverage`,
`lateral_between_max`); an extra array `labels` (int16, shape [F, N]: cluster of each neuron at that frame,
0 = largest, -1 = dead); no `in_w`; at most 150 edges per frame; PCA axes (x_i = Sigma^{1/2} h_i) fixed by
all frames pooled, not the final frame; `family` = "ramp"; `final` = last frame.

| id | what | frames |
|---|---|---|
| `ramp_hier16_g1.0_up` | up ramp, gamma = 1 | 80 |
| `ramp_hier16_g1.0_down2` | down from a cold start at 0.822, gamma = 1 | 69 |
| `ramp_hier16_g0.1_up` | up ramp, gamma = 0.1 (best splitting gamma) | 80 |
| `ramp_hier16_g0.1_down2` | down from a cold start at 0.822, gamma = 0.1 | 69 |

The loop's own down branch (from the end of up) is one cluster for every gamma and is in `ramps.json`
only.
