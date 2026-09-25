"""Paper-scale matched-seed benchmark for hardware-aware BARQ.

This research script is intentionally separate from the lightweight upstream PR
CI.  It reproduces the official X-gate BARQ construction, compares matched
random seeds, reports paired statistics, and writes data used by Figures 2--6.

Variants:
  baseline       : original BARQ geometry objective
  compatibility  : BARQ + local state/geometry feasibility barrier
  leakage        : continuation from the compatibility solution with a
                   differentiable transmon gate-leakage objective

Environment variables:
  PAPER_SEEDS                 default 50
  PAPER_ITERS                 default 10000
  PAPER_LEAKAGE_ITERS         default 2500
  PAPER_OPT_POINTS            default 512
  PAPER_LEAKAGE_WEIGHT        default 0.5
"""

import csv
import io
import json
import os
import time
import contextlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qutip
from scipy import stats

from qurveros import barqtools
from qurveros import compatibility
from qurveros import frametools
from qurveros import jax_controltools
from qurveros import losses
from qurveros import transmon
from qurveros.optspacecurve import BarqCurve
from qurveros.qubit_bench import quantumtools
from qurveros.settings import settings


SEEDS = int(os.environ.get("PAPER_SEEDS", "50"))
ITERS = int(os.environ.get("PAPER_ITERS", "10000"))
LEAKAGE_ITERS = int(os.environ.get("PAPER_LEAKAGE_ITERS", "2500"))
OPT_POINTS = int(os.environ.get("PAPER_OPT_POINTS", "512"))
LEAKAGE_WEIGHT = float(os.environ.get("PAPER_LEAKAGE_WEIGHT", "0.5"))

OMEGA_MAX_HW = 2.0 * jnp.pi * 20e6
DELTA_MAX_HW = 2.0 * jnp.pi * 20e6
PHASE_SLEW_MAX_HW = 2.0 * jnp.pi * 100e6
ANHARMONICITY = -2.0 * jnp.pi * 300e6

GAUGE_W_DETUNING = 4.0
GAUGE_W_PHASE = 1.0
GAUGE_W_SLEW = 0.01

OUTDIR = Path(os.environ.get("PAPER_OUTDIR", "research_outputs"))
OUTDIR.mkdir(parents=True, exist_ok=True)

settings.options["OPT_POINTS"] = OPT_POINTS

compat_endpoint_loss = compatibility.make_population_endpoint_loss(
    theta0=1e-3,
    beta0=jnp.pi / 2.0,
    target_theta=jnp.pi - 1e-3,
    pole_epsilon=1e-4,
)

leakage_loss = transmon.make_transmon_gate_leakage_loss(
    anharmonicity=ANHARMONICITY,
    omega_max_hw=OMEGA_MAX_HW,
    delta_max_hw=DELTA_MAX_HW,
    phase_slew_max_hw=PHASE_SLEW_MAX_HW,
    levels=3,
    n_steps=64,
    w_detuning=GAUGE_W_DETUNING,
    w_phase=GAUGE_W_PHASE,
    w_slew=GAUGE_W_SLEW,
    preserve_barq_gate=True,
)


def make_optimizer(params):
    labels = jax.tree.map(lambda _: True, params)
    labels["pgf_params"]["norm_value"] = False
    return optax.multi_transform(
        {
            True: optax.adam(learning_rate=0.001),
            False: optax.set_to_zero(),
        },
        param_labels=labels,
    )


@jax.jit
def xgate_pgf_mod(pgf_params, input_points):
    del input_points
    p = pgf_params.copy()
    n = p["norm_value"]
    p["left_tangent_fix"] = n
    p["left_tangent_aux"] = n
    p["left_binormal_fix"] = n
    p["right_binormal_fix"] = n
    p["right_tangent_aux"] = n
    p["right_tangent_fix"] = n
    p["left_binormal_aux"] = p["right_binormal_aux"]
    return p


def build_curve(seed, compatibility_aware):
    adj_target = quantumtools.calculate_adj_rep(qutip.sigmax())
    curve = BarqCurve(
        adj_target=adj_target,
        n_free_points=10,
        pgf_mod=xgate_pgf_mod,
    )
    init_pgf = barqtools.get_default_pgf_params_dict()
    init_pgf["norm_value"] = 0.25
    curve.initialize_parameters(
        seed=4531469 + seed,
        init_pgf_params=init_pgf,
    )

    terms = [
        [losses.tantrix_zero_area_loss, 1.0],
        [losses.max_amp_loss, 1e-2],
    ]
    if compatibility_aware:
        terms.append([
            compatibility.smoothstep_polar_compatibility_loss, 1.0
        ])
    curve.prepare_optimization_loss(*terms)
    return curve


def prepare_leakage_refinement(curve):
    curve.prepare_optimization_loss(
        [losses.tantrix_zero_area_loss, 1.0],
        [losses.max_amp_loss, 1e-2],
        [compatibility.smoothstep_polar_compatibility_loss, 1.0],
        [leakage_loss, LEAKAGE_WEIGHT],
    )


def optimize_quiet(curve, n_iter):
    opt = make_optimizer(curve.params)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        curve.optimize(opt, max_iter=n_iter)
    jax.block_until_ready(curve.opt_loss(curve.params))
    return time.perf_counter() - start


def validate(curve, variant, seed, elapsed):
    curve.evaluate_frenet_dict(n_points=2048)
    frenet = curve.get_frenet_dict()

    tan_area = frametools.calculate_tantrix_area(frenet)
    tan_area_sq = float(jnp.sum(tan_area**2))

    total_length = frametools.calculate_total_length(frenet)
    closed_vec = (
        frenet["curve"][-1] - frenet["curve"][0]
    ) / total_length
    closed_sq = float(jnp.sum(closed_vec**2))

    cfi = float(frametools.calculate_cfi_value(frenet))
    endpoint = float(compat_endpoint_loss(frenet))
    compat_barrier = float(
        compatibility.smoothstep_polar_compatibility_loss(frenet)
    )

    control = jax_controltools.hardware_gauge_control(
        frenet,
        w_detuning=GAUGE_W_DETUNING,
        w_phase=GAUGE_W_PHASE,
        w_slew=GAUGE_W_SLEW,
        preserve_barq_gate=True,
    )
    gate_time = float(jax_controltools.minimum_gate_time(
        control["omega"],
        control["delta"],
        control["phase_slew"],
        omega_max_hw=OMEGA_MAX_HW,
        delta_max_hw=DELTA_MAX_HW,
        phase_slew_max_hw=PHASE_SLEW_MAX_HW,
    ))

    tm_gate = transmon.gate_leakage_metrics_from_frenet(
        frenet,
        anharmonicity=ANHARMONICITY,
        omega_max_hw=OMEGA_MAX_HW,
        delta_max_hw=DELTA_MAX_HW,
        phase_slew_max_hw=PHASE_SLEW_MAX_HW,
        levels=3,
        n_steps=256,
        w_detuning=GAUGE_W_DETUNING,
        w_phase=GAUGE_W_PHASE,
        w_slew=GAUGE_W_SLEW,
        preserve_barq_gate=True,
    )
    tm_state = transmon.transmon_metrics_from_frenet(
        frenet,
        anharmonicity=ANHARMONICITY,
        omega_max_hw=OMEGA_MAX_HW,
        delta_max_hw=DELTA_MAX_HW,
        phase_slew_max_hw=PHASE_SLEW_MAX_HW,
        levels=3,
        n_steps=256,
        w_detuning=GAUGE_W_DETUNING,
        w_phase=GAUGE_W_PHASE,
        w_slew=GAUGE_W_SLEW,
        preserve_barq_gate=True,
    )

    leakage = float(tm_gate["leakage"])
    max_leakage = float(tm_state["max_leakage"])
    state0_final_leakage = float(tm_state["leakage"])

    geometric_success = bool(
        np.isfinite(cfi)
        and tan_area_sq < 1e-3
        and closed_sq < 1e-10
    )
    compatibility_success = bool(compat_barrier < 1e-2)
    leakage_success = bool(max_leakage < 5e-3)

    if variant == "baseline":
        success = geometric_success
    elif variant == "compatibility":
        success = geometric_success and compatibility_success
    else:
        success = (
            geometric_success
            and compatibility_success
            and leakage_success
        )

    return {
        "variant": variant,
        "seed": seed,
        "success": success,
        "geometric_success": geometric_success,
        "compatibility_success": compatibility_success,
        "leakage_success": leakage_success,
        "wall_time_s": elapsed,
        "cfi": cfi,
        "gate_time_ns": gate_time * 1e9,
        "leakage": leakage,
        "state0_final_leakage": state0_final_leakage,
        "max_state0_leakage": max_leakage,
        "compatibility_endpoint_loss": endpoint,
        "compatibility_barrier": compat_barrier,
        "tantrix_area_sq": tan_area_sq,
        "closed_sq": closed_sq,
        "tg_omega_max": float(jnp.max(jnp.abs(control["omega"]))),
        "tg_delta_max": float(jnp.max(jnp.abs(control["delta"]))),
        "tg_phase_slew_max": float(
            jnp.max(jnp.abs(control["phase_slew"]))
        ),
    }


def bootstrap_ci(values, statistic=np.mean, n_boot=10000, seed=314159):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.apply_along_axis(statistic, 1, values[idx])
    return [
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    ]


def wilcoxon_paired(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    try:
        result = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return float(result.pvalue)
    except ValueError:
        return 1.0


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = successes / total
    denom = 1.0 + z*z/total
    center = (p + z*z/(2*total)) / denom
    half = z * np.sqrt(
        p*(1-p)/total + z*z/(4*total*total)
    ) / denom
    return [float(center-half), float(center+half)]


def summarize(rows):
    variants = ("baseline", "compatibility", "leakage")
    summary = {
        "configuration": {
            "seeds": SEEDS,
            "iters": ITERS,
            "leakage_iters": LEAKAGE_ITERS,
            "opt_points": OPT_POINTS,
            "leakage_weight": LEAKAGE_WEIGHT,
            "omega_max_mhz": float(OMEGA_MAX_HW / (2*jnp.pi*1e6)),
            "delta_max_mhz": float(DELTA_MAX_HW / (2*jnp.pi*1e6)),
            "phase_slew_max_mhz": float(
                PHASE_SLEW_MAX_HW / (2*jnp.pi*1e6)
            ),
            "anharmonicity_mhz": float(
                ANHARMONICITY / (2*jnp.pi*1e6)
            ),
        },
        "variants": {},
        "paired": {},
    }

    by_variant = {}
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        by_variant[variant] = {int(r["seed"]): r for r in subset}
        successes = sum(bool(r["success"]) for r in subset)
        summary["variants"][variant] = {
            "runs": len(subset),
            "successful_runs": successes,
            "success_rate": successes / max(len(subset), 1),
            "success_rate_ci95": wilson_interval(successes, len(subset)),
        }
        for key in (
            "cfi",
            "gate_time_ns",
            "wall_time_s",
            "leakage",
            "max_state0_leakage",
            "compatibility_barrier",
        ):
            vals = np.array([float(r[key]) for r in subset], dtype=float)
            good_vals = np.array([
                float(r[key]) for r in subset if r["success"]
            ], dtype=float)
            summary["variants"][variant][key] = {
                "mean_all": float(np.mean(vals)),
                "median_all": float(np.median(vals)),
                "ci95_mean_all": bootstrap_ci(vals, np.mean),
                "mean_success": (
                    float(np.mean(good_vals)) if len(good_vals) else None
                ),
                "median_success": (
                    float(np.median(good_vals)) if len(good_vals) else None
                ),
            }

    common = sorted(
        set(by_variant["baseline"])
        & set(by_variant["compatibility"])
        & set(by_variant["leakage"])
    )
    common_success = [
        s for s in common
        if by_variant["baseline"][s]["success"]
        and by_variant["compatibility"][s]["success"]
        and by_variant["leakage"][s]["success"]
    ]
    summary["paired"]["common_successful_seeds"] = common_success
    summary["paired"]["n_common_success"] = len(common_success)

    comparisons = (
        ("baseline", "compatibility"),
        ("compatibility", "leakage"),
        ("baseline", "leakage"),
    )
    for a, b in comparisons:
        key_name = f"{a}_to_{b}"
        summary["paired"][key_name] = {}
        seeds = [
            s for s in common
            if by_variant[a][s]["success"] and by_variant[b][s]["success"]
        ]
        summary["paired"][key_name]["matched_success_seeds"] = seeds
        summary["paired"][key_name]["n"] = len(seeds)

        for metric in ("cfi", "gate_time_ns", "wall_time_s", "leakage"):
            xa = np.array([by_variant[a][s][metric] for s in seeds], float)
            xb = np.array([by_variant[b][s][metric] for s in seeds], float)
            rel = 100.0 * (xb - xa) / np.maximum(np.abs(xa), 1e-30)
            summary["paired"][key_name][metric] = {
                "mean_relative_change_percent": (
                    float(np.mean(rel)) if len(rel) else None
                ),
                "median_relative_change_percent": (
                    float(np.median(rel)) if len(rel) else None
                ),
                "ci95_mean_relative_change_percent": bootstrap_ci(
                    rel, np.mean
                ),
                "wilcoxon_pvalue": (
                    wilcoxon_paired(xa, xb) if len(seeds) else None
                ),
            }

    # Pre-register a representative seed without inspecting a desired outcome:
    # among seeds where all methods succeed, choose the compatibility run whose
    # gate-time is closest to the median compatibility gate-time.
    if common_success:
        gt = np.array([
            by_variant["compatibility"][s]["gate_time_ns"]
            for s in common_success
        ])
        med = float(np.median(gt))
        rep_idx = int(np.argmin(np.abs(gt - med)))
        representative_seed = int(common_success[rep_idx])
    else:
        representative_seed = int(common[0]) if common else 0

    summary["representative_seed"] = representative_seed
    return summary


def run_benchmark():
    rows = []

    for seed in range(SEEDS):
        baseline = build_curve(seed, compatibility_aware=False)
        elapsed = optimize_quiet(baseline, ITERS)
        row = validate(baseline, "baseline", seed, elapsed)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        compat = build_curve(seed, compatibility_aware=True)
        compat_elapsed = optimize_quiet(compat, ITERS)
        row = validate(compat, "compatibility", seed, compat_elapsed)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        prepare_leakage_refinement(compat)
        refinement_elapsed = optimize_quiet(compat, LEAKAGE_ITERS)
        row = validate(
            compat,
            "leakage",
            seed,
            compat_elapsed + refinement_elapsed,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    csv_path = OUTDIR / "matched_seed_benchmark.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    with (OUTDIR / "paper_statistics.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    with (OUTDIR / "representative_seed.json").open("w") as handle:
        json.dump(
            {"seed": summary["representative_seed"]},
            handle,
            indent=2,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return rows, summary


if __name__ == "__main__":
    run_benchmark()
