"""Train one configuration and save everything downstream code needs into one .npz +
one .json (scalars).  Runs on CUDA (Modal) or CPU (pilot).

    python -m hebb.mnist.train --lift pixels --N 256 --theta 0.3 --steps 1500 --out /var/tmp/hebb-mnist/pilot
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hebb.mnist import analysis as A
from hebb.mnist import readout as R
from hebb.mnist.data import load_mnist, make_lift, lifted_stats, offdiag_quantiles
from hebb.mnist.model import Cfg, init_weights, step, clip, harmonic, residual_stats


def snapshot_steps(steps: int, n: int) -> list[int]:
    s = np.unique(np.round(np.geomspace(1, steps, n - 1)).astype(int))
    return [0] + s.tolist()


def run_name(c: Cfg) -> str:
    """e.g. pixels_N2048_th0.75_nw0.5_heavy_sgd_s0  (nw = N * w_min)."""
    if c.lift == "pixels":
        lift = "pixels"
    else:
        lift = f"{c.lift}{c.B}" + ("" if c.bw_mult == 1 else f"bw{c.bw_mult:g}")
    init = "generic" if c.ib_shape == 1.0 else f"heavy" if c.ib_shape == 0.1 else f"gamma{c.ib_shape:g}"
    if abs(c.ii_scale - 0.2 * 512 / c.N) > 1e-9:
        init += f"-ii{c.ii_scale:g}"
    return f"{lift}_N{c.N}_th{c.theta:g}_nw{c.N * c.wmin:.3g}_{init}_{c.mode}_s{c.seed}"


def train(cfg: Cfg, out_dir: Path, device=None, verbose=True) -> dict:
    t_start = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32 if device == "cuda" else torch.float64
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(1000 + cfg.seed)
    name = run_name(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights").mkdir(exist_ok=True)

    # ---- data and exact statistics
    x_tr, y_tr = load_mnist("train")
    lift = make_lift(cfg.lift, cfg.B, cfg.lift_seed, cfg.bw_mult, x_tr)
    mu_g, Sigma_np = lifted_stats(lift, x_tr)
    Sigma = torch.as_tensor(Sigma_np, device=device, dtype=dt)
    X = torch.as_tensor(x_tr, device=device, dtype=dt)
    scorer = R.Scorer(torch.as_tensor(Sigma_np, device="cpu"))

    # ---- init
    Wii, Wib = init_weights(cfg, device, dt)
    Wii, Wib = clip(Wii, Wib, cfg.wmin)
    Wii0, Wib0 = Wii.clone(), Wib.clone()
    rng = np.random.default_rng(cfg.seed)
    sub = np.sort(rng.choice(cfg.N, min(cfg.n_sub, cfg.N), replace=False))
    subt = torch.as_tensor(sub, device=device)

    snaps = snapshot_steps(cfg.steps, cfg.n_snap)
    gpos = np.unique(np.round(np.geomspace(1, len(snaps) - 1, 6)).astype(int)).tolist()
    gpos = sorted(set([0] + gpos[1:] + [len(snaps) - 1]))[-6:] if len(gpos) >= 6 else gpos
    graph_snaps = set(snaps[i] for i in gpos)
    avg_start = cfg.steps - int(cfg.avg_frac * cfg.steps)
    sum_ii = torch.zeros_like(Wii); sum_ib = torch.zeros_like(Wib); n_avg = 0

    H_sub, Wii_g, Wib_g, g_steps = [], [], [], []
    snap_rows, hist_ii, hist_ib, sizes_l, energy_l = [], [], [], [], []
    dense = []
    dense_every = max(1, cfg.steps // 300)
    t_dyn = t_meas = 0.0

    def measure(t, Wii, Wib, H):
        Hs = H.float()
        summ, lab, cent, energy = A.cluster_summary(Hs, Wii, cfg.wmin, Sigma.float(), with_leader=cfg.N <= 1024, Wib=Wib)
        row = dict(step=t, time=t * cfg.dt, **summ, eff_rank=A.eff_rank(H))
        row["floor_ii"], row["floor_ib"] = A.floor_frac(Wii, Wib, cfg.wmin)
        row["mean_wii"] = float(Wii.sum() / (cfg.N * (cfg.N - 1))); row["mean_wib"] = float(Wib.mean())
        row["max_w"] = float(max(Wii.max(), Wib.max()))
        row.update(residual_stats(Wii, Wib, Sigma, cfg))
        if cfg.readout_every_snap:
            # readouts always in float64 (float32 CG lost conjugacy on N=2048 problems)
            Mii, Yii, Mib, Yib = R.equations(Wii.double(), Wib.double(), cfg.theta, cfg.lam, cfg.eta, cfg.gamma, cfg.wmin)
            for fam in ("both",):
                C, info = R.ls_readout(H.double(), Mii, Yii, Mib, Yib, fam, iters=cfg.cg_iters)
                sc = scorer.score(C.cpu(), summ["n_clusters"])
                row.update({f"ro_{fam}_{k}": v for k, v in sc.items() if not k.startswith("_")})
                if fam == "both":
                    row.update({f"ro_info_{k}": v for k, v in info.items()})
        snap_rows.append(row)
        hist_ii.append(A.weight_hist(Wii, True)); hist_ib.append(A.weight_hist(Wib, False))
        sizes_l.append(np.bincount(lab)); energy_l.append(energy.cpu().numpy())
        H_sub.append(H[subt].to(torch.float16).cpu().numpy())
        if t in graph_snaps:
            g_steps.append(t)
            Wii_g.append(Wii[subt][:, subt].to(torch.float16).cpu().numpy())
            Wib_g.append(Wib[subt].to(torch.float16).cpu().numpy())
        if verbose:
            print(f"[{name}] t={t:5d} K={summ['n_clusters']:4d} major={summ['n_major']:3d} "
                  f"erank={row['eff_rank']:.1f} floor_ii={row['floor_ii']:.3f} floor_ib={row['floor_ib']:.3f} "
                  f"res={row['res_max']:.3g} ro_cos={row.get('ro_both_contrast_cos', float('nan')):.3f}", flush=True)

    snapset = set(snaps)
    for t in range(cfg.steps + 1):
        if t in snapset:
            tm = time.time()
            H = harmonic(Wii, Wib)
            measure(t, Wii, Wib, H)
            t_meas += time.time() - tm
        if t == cfg.steps:
            break
        td = time.time()
        if cfg.mode == "mf":
            Wii_n, Wib_n, _ = step(Wii, Wib, cfg, Sigma=Sigma)
        else:
            idx = torch.randint(0, X.shape[0], (cfg.batch,), device=device, generator=gen)
            G = lift.apply_torch(X[idx], torch)
            Wii_n, Wib_n, _ = step(Wii, Wib, cfg, G=G)
        if t % dense_every == 0:
            dense.append([t, float((Wii_n - Wii).abs().mean() / cfg.dt), float((Wib_n - Wib).abs().mean() / cfg.dt),
                          float(Wii_n.sum() / (cfg.N * (cfg.N - 1))), float(Wib_n.mean())])
        Wii, Wib = Wii_n, Wib_n
        if t + 1 > avg_start:
            sum_ii += Wii; sum_ib += Wib; n_avg += 1
        if device == "cuda" and t % 100 == 0:
            torch.cuda.synchronize()
        t_dyn += time.time() - td

    # ---- final analysis on time-averaged weights (sgd) / final weights (mf)
    tm = time.time()
    Aii, Aib = (sum_ii / n_avg, sum_ib / n_avg) if cfg.mode == "sgd" else (Wii, Wib)
    Aii, Aib = Aii.double(), Aib.double()          # final analysis and readouts in float64
    Wii0, Wib0, Wii, Wib = Wii0.double(), Wib0.double(), Wii.double(), Wib.double()
    Ha = harmonic(Aii, Aib)
    summ, lab, cent, energy = A.cluster_summary(Ha.float(), Aii.float(), cfg.wmin, Sigma.float(), with_leader=True, Wib=Aib.float())
    res_avg = residual_stats(Aii, Aib, Sigma.double(), cfg)
    Mii, Yii, Mib, Yib = R.equations(Aii, Aib, cfg.theta, cfg.lam, cfg.eta, cfg.gamma, cfg.wmin)
    final_ro, C_store, ro_info = {}, {}, {}
    for fam in ("both", "ii", "ib"):
        C, info = R.ls_readout(Ha, Mii, Yii, Mib, Yib, fam, iters=2 * cfg.cg_iters)
        final_ro[f"ls_{fam}"] = scorer.score(C.cpu(), summ["n_clusters"], spectra=True)
        C_store[f"ls_{fam}"] = C; ro_info[f"ls_{fam}"] = info
    Cn_ii, Cn_ib = R.naive_readout(Ha, Aii, Aib, cfg.theta, cfg.lam, cfg.eta, cfg.gamma)
    final_ro["naive_ii"] = scorer.score(Cn_ii.cpu(), summ["n_clusters"], spectra=True)
    final_ro["naive_ib"] = scorer.score(Cn_ib.cpu(), summ["n_clusters"], spectra=True)
    C_store["naive_ii"], C_store["naive_ib"] = Cn_ii, Cn_ib
    # untrained-network control (equilibrium assumption violated)
    H0 = harmonic(Wii0, Wib0)
    M0 = R.equations(Wii0, Wib0, cfg.theta, cfg.lam, cfg.eta, cfg.gamma, cfg.wmin)
    C0, info0 = R.ls_readout(H0, *M0, "both", iters=2 * cfg.cg_iters)
    final_ro["init_ls_both"] = scorer.score(C0.cpu(), summ["n_clusters"], spectra=True)
    C_store["init_ls_both"] = C0
    # instantaneous final weights (no averaging), for comparison
    Hf = harmonic(Wii, Wib)
    Mf = R.equations(Wii, Wib, cfg.theta, cfg.lam, cfg.eta, cfg.gamma, cfg.wmin)
    Cf, _ = R.ls_readout(Hf, *Mf, "both", iters=2 * cfg.cg_iters)
    final_ro["instant_ls_both"] = scorer.score(Cf.cpu(), summ["n_clusters"])
    base = scorer.baselines(summ["n_clusters"])
    mrank = R.measurement_rank(cent, lab, Mii, Mib)
    t_final = time.time() - tm

    spectra = {k: v.pop("_spectrum") for k, v in final_ro.items() if "_spectrum" in v}
    spectra["sigma_contrast"] = scorer.ev_Sp.numpy()
    sizes = np.bincount(lab)
    res = dict(name=name, cfg=cfg.to_dict(), lift=lift.describe(), device=device,
               sigma_stats=offdiag_quantiles(Sigma_np), snaps=snaps, graph_steps=g_steps,
               snap_rows=snap_rows, final=dict(**summ, eff_rank=A.eff_rank(Ha), **res_avg,
                                               floor_ii=A.floor_frac(Aii, Aib, cfg.wmin)[0],
                                               floor_ib=A.floor_frac(Aii, Aib, cfg.wmin)[1],
                                               averaged_over_steps=n_avg if cfg.mode == "sgd" else 0),
               cluster_sizes=sizes.tolist(), cluster_energy=energy.cpu().numpy().tolist(),
               readout=final_ro, readout_info=ro_info, baselines=base, measurement_rank=mrank,
               timing=dict(dynamics_s=t_dyn, measure_s=t_meas, final_s=t_final, total_s=time.time() - t_start,
                           gpu=torch.cuda.get_device_name() if device == "cuda" else "cpu"),
               dense_cols=["step", "mean_abs_dwii_dt", "mean_abs_dwib_dt", "mean_wii", "mean_wib"])
    f16 = (lambda T: T.to(torch.float16).cpu().numpy()) if cfg.B > 256 else (lambda T: T.float().cpu().numpy())
    np.savez_compressed(
        out_dir / f"{name}.npz",
        sub=sub, H_sub=np.stack(H_sub), Wii_g=np.stack(Wii_g), Wib_g=np.stack(Wib_g), g_steps=np.array(g_steps),
        hist_ii=np.stack(hist_ii), hist_ib=np.stack(hist_ib), hist_edges=A.HIST_EDGES,
        dense=np.array(dense), labels=lab, cent=cent.float().cpu().numpy(), H_final=Ha.to(torch.float16).cpu().numpy(),
        Sigma=Sigma_np.astype(np.float32), mu_g=mu_g.astype(np.float32),
        sizes_snap=np.array([np.pad(s, (0, max(0, 4096 - len(s))))[:4096] for s in sizes_l], dtype=np.int32),
        **{f"C_{k}": f16(v) for k, v in C_store.items()},
        **{f"spec_{k}": v for k, v in spectra.items()},
    )
    # full weight matrices: separate file, kept on the Modal Volume (not fetched by default)
    np.savez_compressed(out_dir / "weights" / f"{name}_weights.npz",
                        Wii_avg=Aii.to(torch.float16).cpu().numpy(), Wib_avg=Aib.to(torch.float16).cpu().numpy(),
                        Wii_init=Wii0.to(torch.float16).cpu().numpy(), Wib_init=Wib0.to(torch.float16).cpu().numpy())
    energy_snap = [e.tolist() for e in energy_l]
    res["cluster_energy_snap"] = energy_snap
    (out_dir / f"{name}.json").write_text(json.dumps(res, default=float))
    if verbose:
        f = res["final"]; ro = final_ro["ls_both"]
        print(f"[{name}] DONE K={f['n_clusters']} major={f['n_major']} res={f['res_max']:.3g} "
              f"contrast_cos={ro['contrast_cos']:.3f} bw={ro['contrast_bw']:.3f} "
              f"timing={res['timing']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    for k, v in Cfg().to_dict().items():
        if k == "extra":
            continue
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else lambda s: s.lower() in ("1", "true"), default=v)
    ap.add_argument("--out", default="/var/tmp/hebb-mnist/pilot")
    a = vars(ap.parse_args())
    out = Path(a.pop("out"))
    train(Cfg(**a), out)


if __name__ == "__main__":
    main()
