"""Generate paper Figures 2--6 from the matched-seed BARQ benchmark.

The representative seed is selected before detailed plotting by the rule stored
in paper_statistics.json: among seeds where all three methods succeed, choose
the compatibility-aware run whose hardware-box gate time is closest to the
median.  This avoids choosing a visually favorable seed after inspection.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_benchmark as pb
from qurveros import jax_controltools
from qurveros import transmon


OUTDIR = pb.OUTDIR
FIGDIR = OUTDIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def optimize_seed(seed):
    """Reproduce all three optimized curves for one matched seed."""
    baseline = pb.build_curve(seed, compatibility_aware=False)
    baseline_elapsed = pb.optimize_quiet(baseline, pb.ITERS)

    compat = pb.build_curve(seed, compatibility_aware=True)
    compat_elapsed = pb.optimize_quiet(compat, pb.ITERS)

    # Capture compatibility curve before continuation mutates its parameters.
    compat_snapshot = pb.build_curve(seed, compatibility_aware=True)
    compat_snapshot.set_params(
        jax.tree.map(lambda x: jnp.array(x), compat.params)
    )

    pb.prepare_leakage_refinement(compat)
    refine_elapsed = pb.optimize_quiet(compat, pb.LEAKAGE_ITERS)

    return {
        "baseline": (baseline, baseline_elapsed),
        "compatibility": (compat_snapshot, compat_elapsed),
        "leakage": (compat, compat_elapsed + refine_elapsed),
    }


def capture_curve(curve, n_curve=4096, n_control=1024, n_transmon=512):
    curve.evaluate_frenet_dict(n_points=n_curve)
    frenet = curve.get_frenet_dict()

    hw = jax_controltools.hardware_gauge_control(
        frenet,
        w_detuning=pb.GAUGE_W_DETUNING,
        w_phase=pb.GAUGE_W_PHASE,
        w_slew=pb.GAUGE_W_SLEW,
        preserve_barq_gate=True,
    )

    hw_np = {
        k: np.asarray(v)
        for k, v in hw.items()
        if k in (
            "times", "omega", "torsion", "phase_slew",
            "phi", "delta", "total_length"
        )
    }

    gate_time = float(jax_controltools.minimum_gate_time(
        hw["omega"],
        hw["delta"],
        hw["phase_slew"],
        omega_max_hw=pb.OMEGA_MAX_HW,
        delta_max_hw=pb.DELTA_MAX_HW,
        phase_slew_max_hw=pb.PHASE_SLEW_MAX_HW,
    ))
    hw_np["gate_time"] = gate_time

    # Official TTC reconstruction for a direct gauge comparison.
    curve.evaluate_control_dict("TTC", n_points=n_control)
    ttc = {
        k: np.asarray(v)
        for k, v in curve.get_control_dict().items()
        if k in ("times", "omega", "phi", "delta")
    }
    ttc["phase_slew"] = np.gradient(ttc["phi"], ttc["times"])

    # State-|0> transmon dynamics under the optimized hardware gauge.
    sampled = transmon.resample_control(hw, n_transmon)
    phase_slew_sampled = jnp.gradient(
        sampled["phi"], sampled["times"]
    )
    tm_gate_time = float(jax_controltools.minimum_gate_time(
        sampled["omega"],
        sampled["delta"],
        phase_slew_sampled,
        omega_max_hw=pb.OMEGA_MAX_HW,
        delta_max_hw=pb.DELTA_MAX_HW,
        phase_slew_max_hw=pb.PHASE_SLEW_MAX_HW,
    ))
    state0 = jnp.array([1.0 + 0j, 0.0j, 0.0j])
    final_state, history = transmon.propagate_transmon(
        sampled["times"],
        sampled["omega"],
        sampled["phi"],
        sampled["delta"],
        tm_gate_time,
        pb.ANHARMONICITY,
        state0,
        levels=3,
    )
    history = np.asarray(history)
    pops = np.abs(history)**2

    return {
        "frenet": {
            "x_values": np.asarray(frenet["x_values"]),
            "curve": np.asarray(frenet["curve"]),
            "frame": np.asarray(frenet["frame"]),
            "curvature": np.asarray(frenet["curvature"]),
            "torsion": np.asarray(frenet["torsion"]),
            "speed": np.asarray(frenet["speed"]),
        },
        "hardware": hw_np,
        "ttc": ttc,
        "transmon": {
            "times": np.asarray(sampled["times"]),
            "gate_time": tm_gate_time,
            "populations": pops,
            "final_state": np.asarray(final_state),
        },
    }


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
    })


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def common_success_dataframe(df, statistics):
    seeds = statistics["paired"]["common_successful_seeds"]
    if seeds:
        return df[df["seed"].isin(seeds)].copy(), seeds, True
    # Smoke tests and deliberately under-converged runs may have no seed for
    # which all three variants satisfy the final success threshold.  Fall back
    # to all matched seeds for plotting only; the success-rate panel still
    # reports the actual thresholds.
    seeds = sorted(df["seed"].unique().tolist())
    return df[df["seed"].isin(seeds)].copy(), seeds, False


def figure2_matched_seed_benchmark(df, statistics):
    """Figure 2: paired 50-seed statistics."""
    plot_df, seeds, all_success = common_success_dataframe(df, statistics)
    order = ["baseline", "compatibility", "leakage"]
    labels = ["Baseline", "+ Compatibility", "+ Leakage\ncontinuation"]
    x = np.arange(3)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    metric_info = [
        ("cfi", "CFI", axes[0, 0]),
        ("gate_time_ns", r"$T_{\min}$ (ns)", axes[0, 1]),
        ("wall_time_s", "Optimization time (s)", axes[1, 0]),
    ]

    for metric, ylabel, ax in metric_info:
        pivot = plot_df.pivot(index="seed", columns="variant", values=metric)
        pivot = pivot.reindex(columns=order)
        for _, row in pivot.iterrows():
            ax.plot(x, row.values, marker="o", alpha=0.18, linewidth=0.7)
        med = pivot.median(axis=0).values
        ax.plot(x, med, marker="D", linewidth=2.2, label="Median")
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    ax = axes[1, 1]
    success_rates = []
    lows = []
    highs = []
    for variant in order:
        info = statistics["variants"][variant]
        success_rates.append(100 * info["success_rate"])
        lo, hi = info["success_rate_ci95"]
        lows.append(max(0.0, 100 * (info["success_rate"] - lo)))
        highs.append(max(0.0, 100 * (hi - info["success_rate"])))
    ax.bar(x, success_rates, yerr=[lows, highs], capsize=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success rate (%)")
    ax.grid(axis="y", alpha=0.2)

    subset_label = (
        f"common successful seeds: n={len(seeds)}"
        if all_success
        else f"all matched seeds shown; common-success n=0 (smoke mode), n={len(seeds)}"
    )
    fig.suptitle(f"Matched-seed BARQ benchmark ({subset_label})")
    save_figure(fig, "Figure2_matched_seed_benchmark")


def _interp_to(xnew, x, y):
    return np.interp(xnew, x, y)


def figure3_hardware_gauge(bundle):
    """Figure 3: TTC versus optimized hardware-gauge waveforms."""
    hw = bundle["hardware"]
    ttc = bundle["ttc"]

    u = hw["times"]
    omega_ttc = _interp_to(u, ttc["times"], ttc["omega"])
    delta_ttc = _interp_to(u, ttc["times"], ttc["delta"])
    slew_ttc = _interp_to(u, ttc["times"], ttc["phase_slew"])

    residual_hw = hw["phase_slew"] - hw["delta"] - hw["torsion"]
    residual_ttc = slew_ttc - delta_ttc - hw["torsion"]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.9), sharex=True)
    panels = [
        (axes[0, 0], omega_ttc, hw["omega"], r"$T\Omega(u)$"),
        (axes[0, 1], delta_ttc, hw["delta"], r"$T\Delta(u)$"),
        (
            axes[1, 0],
            slew_ttc,
            hw["phase_slew"],
            r"$T\dot{\Phi}(u)$",
        ),
    ]

    for ax, y_ttc, y_hw, ylabel in panels:
        ax.plot(u, y_ttc, label="TTC")
        ax.plot(u, y_hw, label="Optimized gauge")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(u, residual_ttc, label="TTC")
    ax.plot(u, residual_hw, label="Optimized gauge")
    ax.set_ylabel(
        r"$T\dot{\Phi}-T\Delta-T\tau$"
    )
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    for ax in axes[1, :]:
        ax.set_xlabel(r"Normalized time $u=t/T$")

    fig.suptitle(
        "Hardware-gauge redistribution at fixed SCQC geometry"
    )
    save_figure(fig, "Figure3_hardware_gauge")


def figure4_leakage_dynamics(bundles):
    """Figure 4: transmon population and leakage dynamics."""
    order = ["baseline", "compatibility", "leakage"]
    titles = ["Baseline", "+ Compatibility", "+ Leakage continuation"]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), sharex="col")
    summary_rows = []

    for j, variant in enumerate(order):
        tm = bundles[variant]["transmon"]
        u = tm["times"]
        pops = tm["populations"]
        p0, p1, p2 = pops[:, 0], pops[:, 1], pops[:, 2]
        pleak = np.maximum(1.0 - p0 - p1, 1e-16)

        ax = axes[0, j]
        ax.plot(u, p0, label=r"$P_0$")
        ax.plot(u, p1, label=r"$P_1$")
        ax.plot(u, p2, label=r"$P_2$")
        ax.set_title(
            f"{titles[j]}\nT={1e9*tm['gate_time']:.1f} ns"
        )
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        if j == 0:
            ax.set_ylabel("Population")
        ax.legend(frameon=False, ncol=3, loc="upper center")

        ax = axes[1, j]
        ax.plot(u, pleak)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.set_xlabel(r"Normalized time $u=t/T$")
        if j == 0:
            ax.set_ylabel(r"$P_{\mathrm{leak}}$")
        ax.axhline(pleak[-1], linestyle="--", linewidth=1.0)

        summary_rows.append({
            "variant": variant,
            "gate_time_ns": 1e9*tm["gate_time"],
            "max_leakage": float(np.max(pleak)),
            "final_leakage_state0": float(pleak[-1]),
        })

    fig.suptitle(
        "Three-level transmon dynamics: transient excursion and final return"
    )
    save_figure(fig, "Figure4_transmon_leakage_dynamics")

    pd.DataFrame(summary_rows).to_csv(
        OUTDIR / "representative_leakage_dynamics_summary.csv",
        index=False,
    )


def nondominated_front(x, y):
    order = np.argsort(x)
    best_y = np.inf
    keep = []
    for idx in order:
        if y[idx] < best_y:
            keep.append(idx)
            best_y = y[idx]
    keep = np.array(keep, dtype=int)
    return keep[np.argsort(x[keep])]


def figure5_pareto(df, statistics):
    """Figure 5: gate-time/CFI Pareto scatter with leakage as marker size."""
    successful = df[df["success"]].copy()
    order = ["baseline", "compatibility", "leakage"]
    labels = {
        "baseline": "Baseline",
        "compatibility": "+ Compatibility",
        "leakage": "+ Leakage continuation",
    }

    fig, ax = plt.subplots(figsize=(7.3, 5.4))

    for variant in order:
        sub = successful[successful["variant"] == variant]
        x = sub["gate_time_ns"].to_numpy(float)
        y = sub["cfi"].to_numpy(float)
        leak = np.maximum(sub["leakage"].to_numpy(float), 1e-16)
        score = np.clip(-np.log10(leak), 4.0, 14.0)
        sizes = 18.0 + 6.0 * (score - 4.0)

        points = ax.scatter(
            x, y, s=sizes, alpha=0.55, label=labels[variant]
        )
        if len(sub) >= 2:
            front_idx = nondominated_front(x, y)
            ax.plot(
                x[front_idx], y[front_idx],
                linewidth=1.8, alpha=0.9,
            )

    rep = statistics["representative_seed"]
    rep_df = successful[successful["seed"] == rep]
    ax.scatter(
        rep_df["gate_time_ns"],
        rep_df["cfi"],
        marker="*",
        s=180,
        facecolors="none",
        linewidths=1.5,
        label=f"Representative seed {rep}",
    )

    ax.set_xlabel(r"Hardware-box $T_{\min}$ (ns)")
    ax.set_ylabel("CFI")
    ax.set_title(
        "Matched-seed Pareto plane\n(marker size increases as final leakage decreases)"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    save_figure(fig, "Figure5_pareto_front")


def _unitary_grid(times, omega, phi, delta, gate_time,
                  amp_errors, detuning_hz):
    """Vectorized two-level midpoint propagator using analytic SU(2) steps."""
    times = np.asarray(times, float)
    omega = np.asarray(omega, float) / gate_time
    delta = np.asarray(delta, float) / gate_time
    phi = np.asarray(phi, float)

    ea, dd = np.meshgrid(
        np.asarray(amp_errors, float),
        np.asarray(detuning_hz, float),
        indexing="ij",
    )
    eflat = ea.ravel()
    dflat = dd.ravel()
    ngrid = len(eflat)

    U = np.broadcast_to(np.eye(2, dtype=complex), (ngrid, 2, 2)).copy()

    dt = np.diff(times) * gate_time
    om_mid = 0.5 * (omega[:-1] + omega[1:])
    de_mid = 0.5 * (delta[:-1] + delta[1:])
    ph_mid = np.angle(
        np.exp(1j*phi[:-1]) + np.exp(1j*phi[1:])
    )

    for j in range(len(dt)):
        om = (1.0 + eflat) * om_mid[j]
        de = de_mid[j] + 2.0*np.pi*dflat
        hx = om * np.cos(ph_mid[j])
        hy = om * np.sin(ph_mid[j])
        hz = de
        rr = np.sqrt(hx*hx + hy*hy + hz*hz)
        half = 0.5 * rr * dt[j]
        cc = np.cos(half)
        qq = np.zeros_like(rr, dtype=complex)
        nonzero = rr > 1e-30
        qq[nonzero] = -1j * np.sin(half[nonzero]) / rr[nonzero]
        qq[~nonzero] = -0.5j * dt[j]

        step = np.empty((ngrid, 2, 2), dtype=complex)
        step[:, 0, 0] = cc + qq*hz
        step[:, 0, 1] = qq*(hx - 1j*hy)
        step[:, 1, 0] = qq*(hx + 1j*hy)
        step[:, 1, 1] = cc - qq*hz
        U = np.einsum("nij,njk->nik", step, U)

    return U.reshape(len(amp_errors), len(detuning_hz), 2, 2)


def _average_gate_infidelity(Uref, U):
    ref_dag = np.conjugate(Uref.T)
    overlap = np.einsum("ij,...ji->...", ref_dag, U)
    fidelity = (np.abs(overlap)**2 + 2.0) / 6.0
    return np.maximum(1.0 - fidelity, 1e-16)


def _average_fidelity_to_x(U):
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    overlap = np.trace(np.conjugate(x_gate.T) @ U)
    return (np.abs(overlap)**2 + 2.0) / 6.0


def figure6_robustness_maps(bundles):
    """Figure 6: amplitude-error / detuning-error robustness maps."""
    amp_errors = np.linspace(-0.04, 0.04, 41)
    detuning_hz = np.linspace(-342e3, 342e3, 41)
    order = ["baseline", "compatibility", "leakage"]
    titles = ["Baseline", "+ Compatibility", "+ Leakage continuation"]

    maps = {}
    ideal_x_infidelity = {}

    for variant in order:
        hw = bundles[variant]["hardware"]
        # Resample to a fixed grid for the robustness scan.
        u = np.linspace(0.0, 1.0, 512)
        omega = np.interp(u, hw["times"], hw["omega"])
        delta = np.interp(u, hw["times"], hw["delta"])
        cphi = np.interp(u, hw["times"], np.cos(hw["phi"]))
        sphi = np.interp(u, hw["times"], np.sin(hw["phi"]))
        phi = np.arctan2(sphi, cphi)
        gt = float(hw["gate_time"])

        U0_grid = _unitary_grid(
            u, omega, phi, delta, gt,
            np.array([0.0]), np.array([0.0])
        )
        U0 = U0_grid[0, 0]
        Ugrid = _unitary_grid(
            u, omega, phi, delta, gt,
            amp_errors, detuning_hz
        )
        infid = _average_gate_infidelity(U0, Ugrid)
        maps[variant] = infid
        ideal_x_infidelity[variant] = float(
            1.0 - _average_fidelity_to_x(U0)
        )

    all_logs = np.concatenate([
        np.log10(np.maximum(maps[v].ravel(), 1e-16))
        for v in order
    ])
    vmin = max(float(np.quantile(all_logs, 0.01)), -16.0)
    vmax = min(float(np.quantile(all_logs, 0.99)), -1.0)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharex=True, sharey=True)
    image = None
    extent = [
        detuning_hz[0]/1e3,
        detuning_hz[-1]/1e3,
        100*amp_errors[0],
        100*amp_errors[-1],
    ]

    for ax, variant, title in zip(axes, order, titles):
        image = ax.imshow(
            np.log10(np.maximum(maps[variant], 1e-16)),
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Detuning error (kHz)")
        ax.axhline(0.0, linewidth=0.7, alpha=0.5)
        ax.axvline(0.0, linewidth=0.7, alpha=0.5)

    axes[0].set_ylabel("Amplitude error (%)")
    cbar = fig.colorbar(image, ax=axes, shrink=0.88)
    cbar.set_label(r"$\log_{10}(1-F_{\mathrm{avg}})$")
    fig.suptitle(
        "Two-dimensional robustness around each pulse's zero-error realization"
    )
    fig.subplots_adjust(wspace=0.12, right=0.92, top=0.82)
    fig.savefig(
        FIGDIR / "Figure6_amplitude_detuning_robustness.png",
        bbox_inches="tight",
        dpi=300,
    )
    fig.savefig(
        FIGDIR / "Figure6_amplitude_detuning_robustness.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    np.savez(
        OUTDIR / "robustness_maps.npz",
        amplitude_error=amp_errors,
        detuning_hz=detuning_hz,
        baseline=maps["baseline"],
        compatibility=maps["compatibility"],
        leakage=maps["leakage"],
    )
    with (OUTDIR / "robustness_map_metadata.json").open("w") as handle:
        json.dump(
            {
                "ideal_x_infidelity": ideal_x_infidelity,
                "reference": (
                    "Heatmap infidelity is relative to each pulse's own "
                    "zero-error unitary, isolating noise sensitivity."
                ),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def save_representative_raw_data(seed, bundles):
    arrays = {"seed": np.array([seed], dtype=int)}
    for variant, bundle in bundles.items():
        hw = bundle["hardware"]
        arrays[f"{variant}_u"] = hw["times"]
        arrays[f"{variant}_omega"] = hw["omega"]
        arrays[f"{variant}_delta"] = hw["delta"]
        arrays[f"{variant}_phase_slew"] = hw["phase_slew"]
        arrays[f"{variant}_phi"] = hw["phi"]
        arrays[f"{variant}_torsion"] = hw["torsion"]
        tm = bundle["transmon"]
        arrays[f"{variant}_tm_u"] = tm["times"]
        arrays[f"{variant}_tm_populations"] = tm["populations"]
    np.savez(OUTDIR / "representative_seed_detailed_data.npz", **arrays)


def main():
    setup_style()
    df = pd.read_csv(OUTDIR / "matched_seed_benchmark.csv")
    with (OUTDIR / "paper_statistics.json").open() as handle:
        statistics = json.load(handle)

    figure2_matched_seed_benchmark(df, statistics)
    figure5_pareto(df, statistics)

    seed = int(statistics["representative_seed"])
    curves = optimize_seed(seed)
    bundles = {
        variant: capture_curve(curve)
        for variant, (curve, _) in curves.items()
    }

    save_representative_raw_data(seed, bundles)
    figure3_hardware_gauge(bundles["compatibility"])
    figure4_leakage_dynamics(bundles)
    figure6_robustness_maps(bundles)

    with (OUTDIR / "figure_manifest.json").open("w") as handle:
        json.dump(
            {
                "representative_seed": seed,
                "figures": [
                    "Figure2_matched_seed_benchmark",
                    "Figure3_hardware_gauge",
                    "Figure4_transmon_leakage_dynamics",
                    "Figure5_pareto_front",
                    "Figure6_amplitude_detuning_robustness",
                ],
            },
            handle,
            indent=2,
        )

    print(f"Generated Figures 2--6 for representative seed {seed}.")


if __name__ == "__main__":
    main()
