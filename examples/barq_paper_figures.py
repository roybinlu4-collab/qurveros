"""Generate publication-style Figures 2--6 from the matched BARQ benchmark.

Expected inputs are produced by barq_hardware_aware_benchmark.py:
  barq_hardware_aware_benchmark.csv
  representative_baseline.npz
  representative_compatibility.npz
  representative_leakage.npz
"""

from pathlib import Path
import csv
import math

import numpy as np
import matplotlib.pyplot as plt


OUT = Path("paper_figures")
OUT.mkdir(exist_ok=True)

CSV_PATH = Path("barq_hardware_aware_benchmark.csv")
VARIANTS = ("baseline", "compatibility", "leakage")
LABELS = {
    "baseline": "Baseline BARQ",
    "compatibility": "+ Compatibility",
    "leakage": "+ Leakage continuation",
}
COLORS = {
    "baseline": "#4c78a8",
    "compatibility": "#f58518",
    "leakage": "#54a24b",
}


def load_rows():
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    numeric = {
        "seed", "wall_time_s", "cfi", "gate_time_ns", "leakage",
        "state0_leakage", "max_state0_leakage",
        "compatibility_endpoint_loss", "compatibility_barrier",
        "tantrix_area_sq", "closed_curve_sq",
    }
    booleans = {
        "success", "geometric_success", "compatibility_success",
        "leakage_success",
    }
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
        row["seed"] = int(row["seed"])
        for key in booleans:
            if key in row:
                row[key] = row[key].lower() == "true"
    return rows


def by_variant(rows):
    return {
        v: sorted([r for r in rows if r["variant"] == v],
                  key=lambda r: r["seed"])
        for v in VARIANTS
    }


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure2(rows):
    grouped = by_variant(rows)
    common_seeds = sorted(
        set(r["seed"] for r in grouped["baseline"])
        & set(r["seed"] for r in grouped["compatibility"])
        & set(r["seed"] for r in grouped["leakage"])
    )
    maps = {
        v: {r["seed"]: r for r in grouped[v]} for v in VARIANTS
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    metrics = [
        ("cfi", "CFI", axes[0, 0]),
        ("gate_time_ns", r"$T_{\min}$ (ns)", axes[0, 1]),
        ("wall_time_s", "Optimization time (s)", axes[1, 0]),
    ]
    xpos = np.arange(3)

    for key, ylabel, ax in metrics:
        for seed in common_seeds:
            vals = [maps[v][seed][key] for v in VARIANTS]
            ax.plot(xpos, vals, color="0.78", lw=0.7, alpha=0.55, zorder=1)
            ax.scatter(
                xpos, vals,
                c=[COLORS[v] for v in VARIANTS],
                s=15, alpha=0.72, zorder=2,
            )
        med = [np.median([maps[v][s][key] for s in common_seeds]) for v in VARIANTS]
        ax.plot(xpos, med, color="black", lw=2.2, marker="o", ms=5, zorder=3)
        ax.set_xticks(xpos, [LABELS[v] for v in VARIANTS], rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)

    ax = axes[1, 1]
    success = [
        100.0 * np.mean([maps[v][s]["success"] for s in common_seeds])
        for v in VARIANTS
    ]
    ax.bar(xpos, success, color=[COLORS[v] for v in VARIANTS], alpha=0.88)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Task success (%)")
    ax.set_xticks(xpos, [LABELS[v] for v in VARIANTS], rotation=12, ha="right")
    for x, y in zip(xpos, success):
        ax.text(x, y + 2, f"{y:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.22)

    fig.suptitle(
        f"Figure 2 | Matched-seed benchmark ({len(common_seeds)} common seeds)",
        fontsize=13,
    )
    savefig(fig, "figure2_matched_seed_benchmark")


def load_snapshot(variant):
    return np.load(f"representative_{variant}.npz")


def normalized_torsion(snapshot):
    x = snapshot["frenet_x"]
    speed = snapshot["frenet_speed"]
    tau = snapshot["torsion"]
    dx = np.diff(x)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (speed[:-1] + speed[1:]) * dx)])
    L = s[-1]
    u = s / L
    return u, L * tau


def figure3():
    s = load_snapshot("compatibility")
    u = s["u"]
    tau_u, tau_hat = normalized_torsion(s)
    tau_interp = np.interp(u, tau_u, tau_hat)

    ttc_u = s["ttc_u"]
    ttc_tau = np.interp(ttc_u, tau_u, tau_hat)

    fig, axes = plt.subplots(4, 1, figsize=(9.2, 9.5), sharex=False,
                             constrained_layout=True)

    axes[0].plot(ttc_u, s["ttc_omega_hat"], label="TTC", lw=1.6)
    axes[0].plot(u, s["omega_hat"], label="Optimized gauge", lw=1.6)
    axes[0].set_ylabel(r"$T\Omega$")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.2)

    axes[1].plot(ttc_u, s["ttc_delta_hat"], label="TTC", lw=1.6)
    axes[1].plot(u, s["delta_hat"], label="Optimized gauge", lw=1.6)
    axes[1].set_ylabel(r"$T\Delta$")
    axes[1].grid(alpha=0.2)

    axes[2].plot(ttc_u, s["ttc_phase_slew_hat"], label="TTC", lw=1.6)
    axes[2].plot(u, s["phase_slew_hat"], label="Optimized gauge", lw=1.6)
    axes[2].set_ylabel(r"$T\dot{\Phi}$")
    axes[2].grid(alpha=0.2)

    res_ttc = s["ttc_phase_slew_hat"] - s["ttc_delta_hat"] - ttc_tau
    res_opt = s["phase_slew_hat"] - s["delta_hat"] - tau_interp
    axes[3].plot(ttc_u, res_ttc, label="TTC residual", lw=1.4)
    axes[3].plot(u, res_opt, label="Optimized residual", lw=1.4)
    axes[3].set_ylabel(r"$T\dot{\Phi}-T\Delta-T\tau$")
    axes[3].set_xlabel("Normalized time")
    axes[3].set_yscale("symlog", linthresh=1e-8)
    axes[3].grid(alpha=0.2)

    fig.suptitle(
        "Figure 3 | Hardware-gauge redistribution at fixed SCQC geometry",
        fontsize=13,
    )
    savefig(fig, "figure3_hardware_gauge")


def figure4():
    snaps = {v: load_snapshot(v) for v in VARIANTS}
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), constrained_layout=True)

    s = snaps["leakage"]
    axes[0].plot(s["u"], s["p0"], label=r"$P_0$", lw=1.6)
    axes[0].plot(s["u"], s["p1"], label=r"$P_1$", lw=1.6)
    axes[0].plot(s["u"], s["p2"], label=r"$P_2$", lw=1.6)
    axes[0].set_ylabel("Population")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.2)

    for v in VARIANTS:
        ss = snaps[v]
        axes[1].plot(
            ss["u"], np.maximum(ss["p2"], 1e-16),
            label=LABELS[v], lw=1.6, color=COLORS[v],
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Normalized time")
    axes[1].set_ylabel(r"$P_{\mathrm{leak}}(u)\simeq P_2(u)$")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2, which="both")

    fig.suptitle(
        "Figure 4 | Transmon population and transient leakage dynamics",
        fontsize=13,
    )
    savefig(fig, "figure4_transmon_leakage_dynamics")


def figure5(rows):
    grouped = by_variant(rows)
    fig, ax = plt.subplots(figsize=(8.5, 6.4), constrained_layout=True)

    for v in VARIANTS:
        rr = grouped[v]
        x = np.array([r["gate_time_ns"] for r in rr])
        y = np.array([r["cfi"] for r in rr])
        leak = np.array([max(r["leakage"], 1e-16) for r in rr])
        size = np.clip(18 + 9 * (-np.log10(leak) - 6), 18, 95)
        ax.scatter(
            x, y, s=size, alpha=0.62,
            label=LABELS[v], color=COLORS[v],
            edgecolors="white", linewidths=0.45,
        )
        ax.scatter(
            [np.median(x)], [np.median(y)],
            s=150, marker="X", color=COLORS[v],
            edgecolors="black", linewidths=0.8, zorder=5,
        )

    ax.set_xlabel(r"Hardware-box $T_{\min}$ (ns)")
    ax.set_ylabel("CFI")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    ax.set_title(
        "Figure 5 | Pareto plane: shorter gate vs lower geometric complexity\n"
        "Marker size increases as final leakage decreases"
    )
    savefig(fig, "figure5_pareto_front")


def pauli_propagate(snapshot, amp_errors, detuning_hz):
    """Batch-propagate qubit unitaries using an analytic SU(2) step."""
    u = snapshot["u"]
    T = float(snapshot["gate_time_s"])
    omega = snapshot["omega_hat"] / T
    delta = snapshot["delta_hat"] / T
    phi = snapshot["phi"]

    eps_grid, det_grid = np.meshgrid(amp_errors, detuning_hz, indexing="xy")
    eps = eps_grid.reshape(-1)
    dz = (2 * np.pi * det_grid.reshape(-1))

    n = eps.size
    U = np.tile(np.eye(2, dtype=np.complex128), (n, 1, 1))

    dt = np.diff(u) * T
    for j, step_dt in enumerate(dt):
        om = 0.5 * (omega[j] + omega[j + 1])
        ph = np.angle(np.exp(1j * phi[j]) + np.exp(1j * phi[j + 1]))
        de0 = 0.5 * (delta[j] + delta[j + 1])

        ox = (1.0 + eps) * om * np.cos(ph)
        oy = (1.0 + eps) * om * np.sin(ph)
        oz = de0 + dz
        rr = np.sqrt(ox * ox + oy * oy + oz * oz)
        half = 0.5 * rr * step_dt

        c = np.cos(half)
        fac = np.where(rr > 0.0, -1j * np.sin(half) / rr, -0.5j * step_dt)

        S00 = c + fac * oz
        S11 = c - fac * oz
        S01 = fac * (ox - 1j * oy)
        S10 = fac * (ox + 1j * oy)

        a = S00 * U[:, 0, 0] + S01 * U[:, 1, 0]
        b = S00 * U[:, 0, 1] + S01 * U[:, 1, 1]
        c0 = S10 * U[:, 0, 0] + S11 * U[:, 1, 0]
        d = S10 * U[:, 0, 1] + S11 * U[:, 1, 1]
        U[:, 0, 0], U[:, 0, 1] = a, b
        U[:, 1, 0], U[:, 1, 1] = c0, d

    return U.reshape(len(detuning_hz), len(amp_errors), 2, 2)


def figure6():
    amp = np.linspace(-0.04, 0.04, 41)
    det_hz = np.linspace(-342e3, 342e3, 41)

    maps = {}
    all_logs = []
    for v in VARIANTS:
        s = load_snapshot(v)
        Ugrid = pauli_propagate(s, amp, det_hz)
        # Ideal unitary for this control is the central grid point.
        U0 = Ugrid[len(det_hz)//2, len(amp)//2]
        overlap = np.einsum("ab,...ab->...", np.conjugate(U0), Ugrid)
        fidelity = (np.abs(overlap)**2 + 2.0) / 6.0
        infid = np.clip(1.0 - fidelity, 1e-12, 1.0)
        logs = np.log10(infid)
        maps[v] = logs
        all_logs.append(logs)

    vmin = max(-12.0, min(np.min(x) for x in all_logs))
    vmax = min(0.0, max(np.max(x) for x in all_logs))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    im = None
    for ax, v in zip(axes, VARIANTS):
        im = ax.imshow(
            maps[v],
            origin="lower",
            aspect="auto",
            extent=[100*amp[0], 100*amp[-1], det_hz[0]/1e3, det_hz[-1]/1e3],
            vmin=vmin, vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(LABELS[v])
        ax.set_xlabel("Amplitude error (%)")
        ax.grid(False)
    axes[0].set_ylabel("Detuning error (kHz)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label(r"$\log_{10}(1-F_{\rm avg})$")
    fig.suptitle(
        "Figure 6 | Amplitude–detuning robustness landscape",
        fontsize=13,
    )
    savefig(fig, "figure6_robustness_heatmaps")

    # Long-form CSV for independent reuse.
    with (OUT / "figure6_heatmap_data.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "amp_error", "detuning_hz", "log10_infidelity"])
        for v in VARIANTS:
            for iy, d in enumerate(det_hz):
                for ix, a in enumerate(amp):
                    writer.writerow([v, a, d, maps[v][iy, ix]])


def main():
    rows = load_rows()
    figure2(rows)
    figure3()
    figure4()
    figure5(rows)
    figure6()


if __name__ == "__main__":
    main()
