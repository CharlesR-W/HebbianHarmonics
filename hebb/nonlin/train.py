"""Train one configuration of the rectified-output (or linear) rule on raw MNIST pixels or the
Gaussian surrogate, then analyse it.  Writes <name>.npz + <name>.json.

    PYTHONPATH=. python -m hebb.nonlin.train --N 256 --theta 0.2 --steps 1500 --out /var/tmp/hebb-nonlin/pilot
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hebb.mnist import analysis as SA          # sibling: headline cluster definition (L1 < 0.1)
from hebb.mnist import readout as SR           # sibling: linear LS readout + Scorer
from hebb.nonlin import analysis as NA
from hebb.nonlin import readout as NR
from hebb.nonlin.model import Cfg, Data, init_weights, clip, harmonic, step, residual, run_name


def snapshot_steps(steps, n):
    s = np.unique(np.round(np.geomspace(1, steps, n - 1)).astype(int))
    return [0] + s.tolist()


def _f16(T):
    return T.detach().to(torch.float16).cpu().numpy()


def train(cfg: Cfg, out_dir: Path, D: Data | None = None, device=None, verbose=True) -> dict:
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32
    name = run_name(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    D = D or Data(cfg.res, device, dt)
    gen = torch.Generator(device=device).manual_seed(1000 + cfg.seed)
    if cfg.rule == "rect":
        c = cfg.c if cfg.c is not None else D.pixel_rate_gain(cfg.tau)
    else:
        c = 1.0
    Sigma64 = torch.as_tensor(D.Sigma)
    scorer = SR.Scorer(Sigma64)

    Wii, Wib = init_weights(cfg, device, dt)
    Wii, Wib = clip(Wii, Wib, cfg.wmin)
    Wii0, Wib0 = Wii.clone(), Wib.clone()
    rng = np.random.default_rng(cfg.seed)
    sub = np.sort(rng.choice(cfg.N, min(cfg.n_sub, cfg.N), replace=False))
    subt = torch.as_tensor(sub, device=device)
    snaps = snapshot_steps(cfg.steps, cfg.n_snap)
    avg_start = cfg.steps - int(cfg.avg_frac * cfg.steps)
    sum_ii = torch.zeros_like(Wii); sum_ib = torch.zeros_like(Wib); n_avg = 0
    rows, H_sub, dense = [], [], []
    dense_every = max(1, cfg.steps // 300)
    t_dyn = t_meas = 0.0

    for t in range(cfg.steps + 1):
        if t in snaps:
            tm = time.time()
            st, Cii, Cib, H = residual(Wii, Wib, D, cfg, c)
            summ, lab, cent, energy = SA.cluster_summary(H.float(), Wii, cfg.wmin, D.Sigma_t.float(), with_leader=False)
            iu = torch.triu_indices(cfg.N, cfg.N, 1, device=device)
            cv = Cii[iu[0], iu[1]].float()
            q = torch.quantile(cv[torch.randperm(len(cv), device=device)[:1_000_000]],
                               torch.tensor([.1, .5, .9, .99], device=device)).tolist()
            row = dict(step=t, time=t * cfg.dt, **summ, **st, eff_rank=SA.eff_rank(H),
                       cov_q10=q[0], cov_q50=q[1], cov_q90=q[2], cov_q99=q[3])
            row["floor_ii"], row["floor_ib"] = SA.floor_frac(Wii, Wib, cfg.wmin)
            rows.append(row)
            H_sub.append(_f16(H[subt]))
            if verbose:
                print(f"[{name}] t={t:5d} K={summ['n_clusters']:4d} major={summ['n_major']:3d} "
                      f"floor_ii={row['floor_ii']:.3f} floor_ib={row['floor_ib']:.3f} res={st['res_max']:.3g} "
                      f"cov q50/q90/q99={q[1]:.3f}/{q[2]:.3f}/{q[3]:.3f}", flush=True)
            t_meas += time.time() - tm
        if t == cfg.steps:
            break
        td = time.time()
        G = D.sample(cfg.input, cfg.batch, gen)
        Wii_n, Wib_n, _ = step(Wii, Wib, cfg, D, G, c)
        if t % dense_every == 0:
            dense.append([t, float((Wii_n - Wii).abs().mean() / cfg.dt), float((Wib_n - Wib).abs().mean() / cfg.dt),
                          float(Wii_n.sum() / (cfg.N * (cfg.N - 1))), float(Wib_n.mean())])
        Wii, Wib = Wii_n, Wib_n
        if t + 1 > avg_start:
            sum_ii += Wii; sum_ib += Wib; n_avg += 1
        t_dyn += time.time() - td

    # ------------------------------------------------ final analysis on time-averaged weights
    tm = time.time()
    Aii, Aib = sum_ii / n_avg, sum_ib / n_avg
    st_avg, Cii, Cib, Ha = residual(Aii, Aib, D, cfg, c)
    summ, lab, cent, energy = SA.cluster_summary(Ha.float(), Aii, cfg.wmin, D.Sigma_t.float(), with_leader=cfg.N <= 1024)
    tau_e = cfg.tau if cfg.rule == "rect" else 0.0
    v_en, r_en = NA.cluster_rate_energy(cent, D, tau_e)
    H0 = harmonic(Wii0, Wib0)
    res = dict(name=name, cfg=cfg.to_dict(), c=c, pixel_scale=D.scale, device=device, snaps=snaps, snap_rows=rows,
               final=dict(**summ, **st_avg, eff_rank=SA.eff_rank(Ha), averaged_over_steps=n_avg,
                          floor_ii=SA.floor_frac(Aii, Aib, cfg.wmin)[0], floor_ib=SA.floor_frac(Aii, Aib, cfg.wmin)[1]),
               cluster_sizes=np.bincount(lab).tolist(), cluster_energy_volt=v_en.tolist(),
               cluster_energy_rate=r_en.tolist(),
               dense_cols=["step", "mean_abs_dwii_dt", "mean_abs_dwib_dt", "mean_wii", "mean_wib"])
    iu = torch.triu_indices(cfg.N, cfg.N, 1, device=device)
    cv = Cii[iu[0], iu[1]].float()
    qs = [.01, .1, .25, .5, .75, .9, .95, .99, .999]
    res["final_cov_quantiles"] = dict(q=qs, ii=torch.quantile(cv[torch.randperm(len(cv), device=device)[:2_000_000]],
                                                              torch.tensor(qs, device=device)).tolist(),
                                      ib=torch.quantile(Cib.flatten().float(), torch.tensor(qs, device=device)).tolist())

    # higher-order gap (always measured at the rule's tau; for linear-rule nets at tau = 0, post hoc)
    Mii_a = (Aii > 2 * cfg.wmin).float(); Mii_a.fill_diagonal_(0); Mib_a = (Aib > 2 * cfg.wmin).float()
    res["gap"] = NA.gap_analysis(Ha, D, tau_e, Mii_a, Mib_a, seed=cfg.seed)
    res["gap_init"] = NA.gap_analysis(H0, D, tau_e, None, None, n_pairs=1000, seed=cfg.seed)
    t_gap = time.time() - tm

    # weight-space readout
    C_store = {}
    if cfg.readout:
        tr = time.time()
        ro = {}
        Hd = Ha.double()
        g_eff = cfg.gamma if cfg.gamma != 0 else 1.0          # gamma = 0: lateral edges carry no information

        def no_ii(eq):
            if cfg.gamma != 0:
                return eq
            Mii, Qii, Mib, Qib = eq
            return torch.zeros_like(Mii), torch.zeros_like(Qii), Mib, Qib
        fams = ("both", "ii", "ib") if cfg.gamma != 0 else ("both", "ib")
        if cfg.rule == "rect":
            eq = no_ii(NR.rect_equations(Aii.double(), Aib.double(), cfg.theta, cfg.lam, cfg.eta, g_eff, cfg.wmin, c))
            for fam in fams:
                C, C0, info = NR.gauss_readout(Hd, *eq, tau=cfg.tau, family=fam, iters=cfg.nls_iters)
                ro[f"gauss_{fam}"] = dict(score=scorer.score(C.cpu(), summ["n_clusters"], spectra=(fam == "both")), info=info)
                C_store[f"gauss_{fam}"] = C
                if fam == "both":
                    ro["chord_both"] = dict(score=scorer.score(C0.cpu(), summ["n_clusters"]))
                    C_store["chord_both"] = C0
            # same active edges, targets = exact expected rate covariance (no stochastic/non-equilibrium error):
            # what remains is the Gaussian-closure bias
            Mii, _, Mib, _ = eq
            Ce, _, info = NR.gauss_readout(Hd, Mii, Cii.double() / (c * c), Mib, Cib.double() / (c * c),
                                           tau=cfg.tau, iters=cfg.nls_iters)
            ro["gauss_expected_cov"] = dict(score=scorer.score(Ce.cpu(), summ["n_clusters"], spectra=True), info=info)
            C_store["gauss_expected_cov"] = Ce
            # theta unknown (fitted): is theta still a gauge?
            eq0 = no_ii(NR.rect_equations(Aii.double(), Aib.double(), 0.0, cfg.lam, cfg.eta, g_eff, cfg.wmin, c))
            Cf, _, info = NR.gauss_readout(Hd, *eq0, tau=cfg.tau, iters=cfg.nls_iters, fit_theta=True,
                                           a_ii=1 / (g_eff * c * c), a_ib=1 / (c * c))
            ro["gauss_fit_theta"] = dict(score=scorer.score(Cf.cpu(), summ["n_clusters"]), info=info)
            # untrained control (equilibrium assumption violated)
            H0d = H0.double()
            eqi = no_ii(NR.rect_equations(Wii0.double(), Wib0.double(), cfg.theta, cfg.lam, cfg.eta, g_eff, cfg.wmin, c))
            Ci, _, info = NR.gauss_readout(H0d, *eqi, tau=cfg.tau, iters=min(cfg.nls_iters, 150))
            ro["init_gauss_both"] = dict(score=scorer.score(Ci.cpu(), summ["n_clusters"]), info=info)
        else:
            Mii, Yii, Mib, Yib = no_ii(SR.equations(Aii.double(), Aib.double(), cfg.theta, cfg.lam, cfg.eta, g_eff, cfg.wmin))
            for fam in fams:
                C, info = SR.ls_readout(Hd, Mii, Yii, Mib, Yib, fam, iters=800)
                ro[f"ls_{fam}"] = dict(score=scorer.score(C.cpu(), summ["n_clusters"], spectra=(fam == "both")), info=info)
                C_store[f"ls_{fam}"] = C
            Mi0 = no_ii(SR.equations(Wii0.double(), Wib0.double(), cfg.theta, cfg.lam, cfg.eta, g_eff, cfg.wmin))
            C0, info = SR.ls_readout(H0.double(), *Mi0, "both", iters=800)
            ro["init_ls_both"] = dict(score=scorer.score(C0.cpu(), summ["n_clusters"]), info=info)
        spectra = {}
        for k, v in ro.items():
            sp = v["score"].pop("_spectrum", None)
            if sp is not None:
                spectra[k] = sp
        res["readout"] = ro
        res["readout_baselines"] = scorer.baselines(summ["n_clusters"])
        res["sigma_contrast_spectrum"] = scorer.ev_Sp.numpy().tolist()
        res["readout_spectra"] = {k: np.asarray(v).tolist() for k, v in spectra.items()}
        res["t_readout"] = time.time() - tr

    if cfg.probes:
        tp = time.time()
        res["probes"] = NA.probe_suite(Ha, H0, D, cfg.B, seed=cfg.seed)
        res["t_probes"] = time.time() - tp

    res["timing"] = dict(dynamics_s=t_dyn, measure_s=t_meas, gap_s=t_gap, total_s=time.time() - t0,
                         gpu=torch.cuda.get_device_name() if device == "cuda" else "cpu")
    np.savez_compressed(
        out_dir / f"{name}.npz", sub=sub, H_sub=np.stack(H_sub), dense=np.array(dense), labels=lab,
        cent=cent.float().cpu().numpy(), H_final=Ha.float().cpu().numpy(), H_init=_f16(H0),
        Wii_avg=_f16(Aii), Wib_avg=Aib.float().cpu().numpy(),
        **{f"C_{k}": v.float().cpu().numpy() for k, v in C_store.items()})
    (out_dir / f"{name}.json").write_text(json.dumps(res, default=float))
    if verbose:
        f = res["final"]
        msg = f"[{name}] DONE K={f['n_clusters']} major={f['n_major']} res={f['res_max']:.3g} within_l1={f['within_l1']:.3f}"
        if "readout" in res:
            k0 = "gauss_both" if cfg.rule == "rect" else "ls_both"
            msg += f" ro_contrast_cos={res['readout'][k0]['score']['contrast_cos']:.3f}"
        if "probes" in res:
            p = res["probes"]; msg += f" probe px={p['pixels']['test']:.3f} rate_all={p['trained_rate_t0_all']['test']:.3f}"
        print(msg, f"timing={res['timing']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    for k, v in Cfg().to_dict().items():
        if k in ("extra", "B"):
            continue
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true"), default=v)
        elif v is None:
            ap.add_argument(f"--{k}", type=float, default=None)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--out", default="/var/tmp/hebb-nonlin/pilot")
    a = vars(ap.parse_args())
    out = Path(a.pop("out"))
    train(Cfg(**a), out)


if __name__ == "__main__":
    main()
