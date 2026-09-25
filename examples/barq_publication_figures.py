"""Generate publication-ready figures for the hardware-aware BARQ study.

The script consumes the matched-seed benchmark CSV, selects a representative
compatibility-aware seed (closest to the median gate-time lower bound), reruns
that seed with the same optimizer settings, and writes Figures 2--6 plus
machine-readable CSV/NPZ data.

Figures:
  2. Matched-seed benchmark (CFI, gate time, wall time, success rate)
  3. TTC versus optimized hardware gauge on one robust curve
  4. Three-level transmon leakage dynamics
  5. CFI--gate-time Pareto plane, with final leakage encoded
  6. Amplitude--detuning average-gate-infidelity maps
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from barq_hardware_aware_benchmark import (
    ANHARMONICITY,
    DELTA_MAX_HW,
    GAUGE_W_DETUNING,
    GAUGE_W_PHASE,
    GAUGE_W_SLEW,
    ITERS,
    LEAKAGE_ITERS,
    OMEGA_MAX_HW,
    PHASE_SLEW_MAX_HW,
    build_curve,
    make_optimizer,
    prepare_leakage_refinement,
    run_optimize_silent,
)
from qurveros import jax_controltools
from qurveros import transmon


OUT = Path(os.environ.get("BARQ_FIGURE_DIR", "publication_results"))
OUT.mkdir(parents=True, exist_ok=True)
CSV_PATH = Path(
    os.environ.get("BARQ_BENCHMARK_CSV", "barq_hardware_aware_benchmark.csv")
)
HEAT_N = int(os.environ.get("BARQ_HEATMAP_N", "41"))
CONTROL_N = int(os.environ.get("BARQ_FIGURE_CONTROL_POINTS", "1024"))


def _bootstrap_median_ci(values, n_boot=4000, seed=2026):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    med = float(np.median(values))
    if len(values) <= 1:
        return med, med, med
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    meds = np.median(draws, axis=1)
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return med, float(lo), float(hi)


def _wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    den = 1.0 + z*z/total
    centre = (p + z*z/(2*total)) / den
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return p, max(0.0, centre-half), min(1.0, centre+half)


def choose_representative_seed(df):
    comp = df[(df["variant"] == "compatibility") & (df["success"] == True)]
    target = comp["gate_time_ns"].median()
    idx = (comp["gate_time_ns"] - target).abs().idxmin()
    return int(comp.loc[idx, "seed"])


def optimize_representative(seed):
    baseline = build_curve(seed, "baseline")
    run_optimize_silent(baseline, make_optimizer(baseline.params), ITERS)

    compat = build_curve(seed, "compatibility")
    run_optimize_silent(compat, make_optimizer(compat.params), ITERS)

    leakage = build_curve(seed, "compatibility")
    run_optimize_silent(leakage, make_optimizer(leakage.params), ITERS)
    prepare_leakage_refinement(leakage)
    run_optimize_silent(
        leakage, make_optimizer(leakage.params), LEAKAGE_ITERS
    )

    for curve in (baseline, compat, leakage):
        curve.evaluate_frenet_dict(n_points=2048)
    return baseline, compat, leakage


def figure2(df):
    variants = ["baseline", "compatibility", "leakage"]
    labels = ["Baseline", "+ Compatibility", "+ Leakage\ncontinuation"]
    x = np.arange(3)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), constrained_layout=True)
    metrics = [
        ("cfi", "CFI"),
        ("gate_time_ns", r"$T_{\min}$ (ns)"),
        ("wall_time_s", "Optimization time (s)"),
    ]

    for ax, (metric, ylabel) in zip(axes.flat[:3], metrics):
        pivot = df.pivot(index="seed", columns="variant", values=metric)
        for _, row in pivot.iterrows():
            ax.plot(x, [row.get(v, np.nan) for v in variants],
                    marker="o", linewidth=0.6, alpha=0.20)
        meds, los, his = [], [], []
        for v in variants:
            vals = df[df["variant"] == v][metric].to_numpy()
            m, lo, hi = _bootstrap_median_ci(vals)
            meds.append(m); los.append(lo); his.append(hi)
        ax.errorbar(
            x, meds,
            yerr=[np.asarray(meds)-np.asarray(los),
                  np.asarray(his)-np.asarray(meds)],
            marker="D", linewidth=2.0, capsize=4,
        )
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    ax = axes.flat[3]
    rates, lows, highs = [], [], []
    for v in variants:
        sub = df[df["variant"] == v]
        p, lo, hi = _wilson_interval(int(sub["success"].sum()), len(sub))
        rates.append(100*p); lows.append(100*lo); highs.append(100*hi)
    ax.bar(x, rates)
    ax.errorbar(
        x, rates,
        yerr=[np.asarray(rates)-np.asarray(lows),
              np.asarray(highs)-np.asarray(rates)],
        fmt="none", capsize=4,
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success rate (%)")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Figure 2. Matched-seed BARQ benchmark")
    fig.savefig(OUT / "figure2_matched_seed_benchmark.png", dpi=240)
    fig.savefig(OUT / "figure2_matched_seed_benchmark.pdf")
    plt.close(fig)


def _optimized_control(curve):
    frenet = curve.get_frenet_dict()
    return jax_controltools.hardware_gauge_control(
        frenet,
        w_detuning=GAUGE_W_DETUNING,
        w_phase=GAUGE_W_PHASE,
        w_slew=GAUGE_W_SLEW,
        preserve_barq_gate=True,
    )


def figure3(compat):
    compat.evaluate_control_dict(control_mode="TTC", n_points=CONTROL_N)
    ttc = compat.get_control_dict()
    opt = _optimized_control(compat)
    opt = transmon.resample_control(opt, CONTROL_N)

    u = np.asarray(opt["times"])
    ttc_phi = np.unwrap(np.asarray(ttc["phi"]))
    ttc_slew = np.gradient(ttc_phi, np.asarray(ttc["times"]))
    opt_phi = np.unwrap(np.asarray(opt["phi"]))
    opt_slew = np.gradient(opt_phi, u)

    # Resample torsion to the same grid.
    geom = jax_controltools.geometry_control_arrays(compat.get_frenet_dict())
    torsion = np.interp(u, np.asarray(geom["times"]), np.asarray(geom["torsion"]))

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9.0), sharex=True,
                             constrained_layout=True)
    axes[0].plot(np.asarray(ttc["times"]), np.asarray(ttc["omega"]), label="TTC")
    axes[0].plot(u, np.asarray(opt["omega"]), "--", label="Optimized gauge")
    axes[0].set_ylabel(r"$T\Omega$")
    axes[0].legend()

    axes[1].plot(np.asarray(ttc["times"]), np.asarray(ttc["delta"]), label="TTC")
    axes[1].plot(u, np.asarray(opt["delta"]), "--", label="Optimized gauge")
    axes[1].set_ylabel(r"$T\Delta$")

    axes[2].plot(np.asarray(ttc["times"]), ttc_slew, label="TTC")
    axes[2].plot(u, opt_slew, "--", label="Optimized gauge")
    axes[2].set_ylabel(r"$T\dot{\Phi}$")

    ttc_tau_res = ttc_slew - np.asarray(ttc["delta"]) - torsion
    opt_tau_res = opt_slew - np.asarray(opt["delta"]) - torsion
    axes[3].plot(u, ttc_tau_res, label="TTC residual")
    axes[3].plot(u, opt_tau_res, "--", label="Optimized-gauge residual")
    axes[3].axhline(0.0, linewidth=0.8)
    axes[3].set_ylabel(r"$T\dot\Phi-T\Delta-T\tau$")
    axes[3].set_xlabel(r"Normalized time $u$")
    axes[3].legend()
    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle("Figure 3. Hardware-gauge redistribution at fixed geometry")
    fig.savefig(OUT / "figure3_hardware_gauge_waveforms.png", dpi=240)
    fig.savefig(OUT / "figure3_hardware_gauge_waveforms.pdf")
    plt.close(fig)

    data = pd.DataFrame({
        "u": u,
        "ttc_omega": np.interp(u, np.asarray(ttc["times"]), np.asarray(ttc["omega"])),
        "ttc_delta": np.interp(u, np.asarray(ttc["times"]), np.asarray(ttc["delta"])),
        "ttc_phase_slew": np.interp(u, np.asarray(ttc["times"]), ttc_slew),
        "optimized_omega": np.asarray(opt["omega"]),
        "optimized_delta": np.asarray(opt["delta"]),
        "optimized_phase_slew": opt_slew,
        "torsion": torsion,
        "ttc_identity_residual": ttc_tau_res,
        "optimized_identity_residual": opt_tau_res,
    })
    data.to_csv(OUT / "figure3_hardware_gauge_waveforms.csv", index=False)


def _transmon_history(curve, initial_level=0):
    control = _optimized_control(curve)
    sampled = transmon.resample_control(control, CONTROL_N)
    phase_slew = jnp.gradient(sampled["phi"], sampled["times"])
    gate_time = jax_controltools.minimum_gate_time(
        sampled["omega"], sampled["delta"], phase_slew,
        omega_max_hw=OMEGA_MAX_HW,
        delta_max_hw=DELTA_MAX_HW,
        phase_slew_max_hw=PHASE_SLEW_MAX_HW,
    )
    state = jnp.zeros((3,), dtype=jnp.complex128)
    state = state.at[initial_level].set(1.0 + 0.0j)
    _, history = transmon.propagate_transmon(
        sampled["times"], sampled["omega"], sampled["phi"], sampled["delta"],
        gate_time, ANHARMONICITY, state, levels=3,
    )
    return sampled, float(gate_time), np.asarray(history)


def figure4(compat, leakage):
    c_ctrl, c_time, c_hist = _transmon_history(compat)
    l_ctrl, l_time, l_hist = _transmon_history(leakage)
    u = np.asarray(c_ctrl["times"])
    pc = np.abs(c_hist)**2
    pl = np.abs(l_hist)**2

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for level in range(3):
        axes[0].plot(u, pc[:, level], label=fr"$P_{level}$")
        axes[1].plot(u, pl[:, level], label=fr"$P_{level}$")
    axes[0].set_title(f"Compatibility only ({c_time*1e9:.1f} ns)")
    axes[1].set_title(f"Leakage continuation ({l_time*1e9:.1f} ns)")
    axes[0].set_ylabel("Population")
    axes[0].legend()
    axes[1].legend()

    axes[2].semilogy(u, np.maximum(pc[:, 2], 1e-16), label="Compatibility")
    axes[2].semilogy(u, np.maximum(pl[:, 2], 1e-16), label="Leakage continuation")
    axes[2].set_title(r"Leakage population $P_2$")
    axes[2].set_ylabel(r"$P_{\rm leak}$")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel(r"Normalized time $u$")
        ax.grid(alpha=0.2)

    fig.suptitle("Figure 4. Three-level transmon leakage dynamics")
    fig.savefig(OUT / "figure4_transmon_leakage_dynamics.png", dpi=240)
    fig.savefig(OUT / "figure4_transmon_leakage_dynamics.pdf")
    plt.close(fig)

    pd.DataFrame({
        "u": u,
        "compat_p0": pc[:, 0], "compat_p1": pc[:, 1], "compat_p2": pc[:, 2],
        "leak_p0": pl[:, 0], "leak_p1": pl[:, 1], "leak_p2": pl[:, 2],
    }).to_csv(OUT / "figure4_transmon_leakage_dynamics.csv", index=False)


def _pareto_mask(x, y):
    # Minimize both x and y.
    order = np.argsort(x)
    mask = np.zeros(len(x), dtype=bool)
    best = np.inf
    for i in order:
        if y[i] < best:
            mask[i] = True
            best = y[i]
    return mask


def figure5(df):
    fig, ax = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    markers = {"baseline": "o", "compatibility": "s", "leakage": "^"}
    labels = {
        "baseline": "Baseline",
        "compatibility": "+ Compatibility",
        "leakage": "+ Leakage continuation",
    }
    all_leak = np.clip(df["leakage"].to_numpy(float), 1e-16, None)
    zmin, zmax = -np.log10(all_leak).min(), -np.log10(all_leak).max()
    # Use one common color scale: -log10(leakage), larger is better.
    sc_ref = None
    for variant in markers:
        sub = df[df["variant"] == variant]
        z = -np.log10(np.clip(sub["leakage"].to_numpy(float), 1e-16, None))
        sc = ax.scatter(
            sub["gate_time_ns"], sub["cfi"], c=z, cmap="viridis",
            marker=markers[variant], s=42, alpha=0.75, label=labels[variant],
            vmin=min(-np.log10(all_leak)), vmax=max(-np.log10(all_leak)),
        )
        sc_ref = sc
        mask = _pareto_mask(sub["gate_time_ns"].to_numpy(float),
                            sub["cfi"].to_numpy(float))
        frontier = sub.iloc[np.where(mask)[0]].sort_values("gate_time_ns")
        ax.plot(frontier["gate_time_ns"], frontier["cfi"], linewidth=1.2)
    cb = fig.colorbar(sc_ref, ax=ax)
    cb.set_label(r"$-\log_{10} P_{\rm leak}(T)$")
    ax.set_xlabel(r"Hardware-box $T_{\min}$ (ns)")
    ax.set_ylabel("CFI")
    ax.set_title("Figure 5. Matched-seed Pareto plane")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "figure5_pareto_front.png", dpi=240)
    fig.savefig(OUT / "figure5_pareto_front.pdf")
    plt.close(fig)


def _su2_infidelity(control, gate_time_s, amp_errors, detuning_hz):
    u = np.asarray(control["times"], dtype=float)
    omega = np.asarray(control["omega"], dtype=float)
    delta = np.asarray(control["delta"], dtype=float)
    phi = np.unwrap(np.asarray(control["phi"], dtype=float))

    du = np.diff(u)
    om = 0.5*(omega[:-1] + omega[1:])
    de = 0.5*(delta[:-1] + delta[1:])
    ph = 0.5*(phi[:-1] + phi[1:])

    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    ident = np.eye(2, dtype=complex)
    target = -1j * sx

    out = np.empty((len(detuning_hz), len(amp_errors)), dtype=float)
    for iz, dz_hz in enumerate(detuning_hz):
        dz = 2*np.pi*dz_hz*gate_time_s
        for ia, eps in enumerate(amp_errors):
            U = ident.copy()
            for h, o, d, p in zip(du, om, de, ph):
                hx = (1.0+eps)*o*np.cos(p)
                hy = (1.0+eps)*o*np.sin(p)
                hz = d + dz
                r = np.sqrt(hx*hx + hy*hy + hz*hz)
                if r < 1e-15:
                    step = ident
                else:
                    a = 0.5*h*r
                    Hn = (hx*sx + hy*sy + hz*sz)/r
                    step = np.cos(a)*ident - 1j*np.sin(a)*Hn
                U = step @ U
            tr = np.trace(target.conj().T @ U)
            fidelity = (2.0 + np.abs(tr)**2)/6.0
            out[iz, ia] = max(1.0-fidelity, 1e-16)
    return out


def figure6(curves):
    names = ["Baseline", "+ Compatibility", "+ Leakage continuation"]
    amp = np.linspace(-0.04, 0.04, HEAT_N)
    det = np.linspace(-342e3, 342e3, HEAT_N)

    maps = []
    for curve in curves:
        ctrl = _optimized_control(curve)
        ctrl = transmon.resample_control(ctrl, 512)
        slew = jnp.gradient(ctrl["phi"], ctrl["times"])
        gate_time = float(jax_controltools.minimum_gate_time(
            ctrl["omega"], ctrl["delta"], slew,
            omega_max_hw=OMEGA_MAX_HW,
            delta_max_hw=DELTA_MAX_HW,
            phase_slew_max_hw=PHASE_SLEW_MAX_HW,
        ))
        maps.append(_su2_infidelity(ctrl, gate_time, amp, det))

    logmaps = [np.log10(m) for m in maps]
    vmin = min(m.min() for m in logmaps)
    vmax = max(m.max() for m in logmaps)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    im = None
    for ax, name, z in zip(axes, names, logmaps):
        im = ax.imshow(
            z, origin="lower", aspect="auto",
            extent=[100*amp[0], 100*amp[-1], det[0]/1e3, det[-1]/1e3],
            vmin=vmin, vmax=vmax, cmap="magma",
        )
        ax.set_title(name)
        ax.set_xlabel("Amplitude error (%)")
        ax.set_ylabel("Detuning error (kHz)")
    cb = fig.colorbar(im, ax=axes)
    cb.set_label(r"$\log_{10}(1-F_{\rm avg})$")
    fig.suptitle("Figure 6. Amplitude--detuning robustness maps")
    fig.savefig(OUT / "figure6_amplitude_detuning_robustness.png", dpi=240)
    fig.savefig(OUT / "figure6_amplitude_detuning_robustness.pdf")
    plt.close(fig)

    np.savez(
        OUT / "figure6_amplitude_detuning_robustness.npz",
        amplitude_error=amp,
        detuning_hz=det,
        baseline=maps[0],
        compatibility=maps[1],
        leakage=maps[2],
    )


def main():
    df = pd.read_csv(CSV_PATH)
    # Pandas may read boolean columns as strings in some CSV dialects.
    for col in ["success", "geometric_success", "compatibility_success",
                "leakage_success"]:
        if col in df:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(df[col]).astype(bool)

    figure2(df)
    figure5(df)

    seed = choose_representative_seed(df)
    (OUT / "representative_seed.txt").write_text(str(seed) + "\n")
    baseline, compat, leakage = optimize_representative(seed)
    figure3(compat)
    figure4(compat, leakage)
    figure6((baseline, compat, leakage))

    print(f"Representative seed: {seed}")
    print(f"Figures written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
